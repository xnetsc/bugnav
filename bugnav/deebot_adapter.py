"""Real-robot adapter built on the unofficial `deebot-client` library.

Verified against deebot-client 11.0.0 (the pinned PyPI release - its
wheels install cleanly in Docker; upstream git main is ahead of PyPI
and needs a Rust toolchain to build, so we do NOT track main):
  * Authenticator + create_rest_config + ApiClient.get_devices()
  * Device(device_info, authenticator); await device.initialize(mqtt)
  * PositionsEvent carries Position(x, y, a) - raw map units + heading;
    PositionType lives in deebot_client.events on this version
  * StateEvent / ErrorEvent for status and faults
  * Clean(CleanAction.PAUSE/RESUME/START/STOP) and
    CleanArea(CleanMode.CUSTOM_AREA, "x1,y1,x2,y2") - area is a
    comma-joined string on this version, not a list

UNVERIFIED on a physical T80s: whether the T80s (DEX62, China region)
is classified as a supported MQTT device at all, whether GetPos streams
at a useful rate, and whether CUSTOM_AREA commands are accepted.  The
runner's `observe` mode exists precisely to answer those questions
before any command is ever sent.  If the device lands in
`devices.not_supported`, this adapter says so loudly and exits.

The account password is read from the ECOVACS_PASSWORD environment
variable only - never from the config file, never logged.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass

from .geometry import Point
from .model import Command, CommandKind, RobotState, Sample
from .network import call_with_retry


@dataclass
class AdapterConfig:
    account_id: str  # CN accounts: the short ID (e.g. "EXXXXXXXX"), not phone
    country: str = "CN"
    device_id: str = "bugnav-host-001"  # must stay stable across runs
    device_name: str = ""  # substring match; empty = first device
    raw_per_meter: float = 1000.0  # CALIBRATE on the real robot (see README)
    origin_raw_x: float = 0.0
    origin_raw_y: float = 0.0
    invert_x: bool = False
    invert_y: bool = False
    clean_command_v2: bool = False
    # For aliased classes, replace the template's state capability:
    # state via GetChargeState + GetWorkState instead of GetCleanInfo[V2].
    # Newer firmwares dropped getCleanInfo entirely (the T80S fw 1.61.0
    # answers `20003 rcp not support` to both V1 and V2), which left the
    # work state (cleaning/paused/returning) frozen at whatever the
    # charge state last said - observed live 2026-11-11: state stayed
    # "docked" through a whole cleaning run. deebot-client's newest
    # hardware classes (c8rj4y etc.) source StateEvent exactly this way.
    work_state_patch: bool = True
    # Hardware classes deebot-client does not know yet, aliased to the
    # closest known class's capability set. The T80S (model
    # KEPLER_PLUS_WH_AUTO, class "nazy54", probed live 2026-11-11) is
    # unknown upstream. First template tried (ee23uv, V1 commands) got
    # `20003 rcp not support` for getCleanInfo from the T80S firmware
    # (fw 1.61.0) - it speaks the V2 command set. 9kpees carries the
    # full modern V2 set: CleanV2/CleanAreaV2, GetPos, GetError,
    # PlaySound (its state capability is patched per above).
    hardware_aliases: dict = None  # default set in __post_init__

    def __post_init__(self):
        if self.hardware_aliases is None:
            self.hardware_aliases = {"nazy54": "9kpees"}


_STATE_MAP = {}


def parse_device_clock(header, mono_now: float):
    """(device_epoch_s, mono_now, tz_offset_hours|None) from a
    response header, or None when the header carries no usable
    device timestamp.

    The firmware stamps every response with its own 'ts' (device
    epoch milliseconds, as a string) and 'tzm' (UTC offset minutes).
    The robot is the authority for both - the only sanity bounds
    reject absurd epochs (a device that has not time-synced yet);
    the value is NEVER clamped against the host clock, because the
    host clock is exactly what is not trusted."""
    if not header:
        return None
    try:
        epoch_s = float(header.get("ts")) / 1000.0
    except (TypeError, ValueError):
        return None
    if not (1_577_836_800 <= epoch_s <= 4_102_444_800):  # 2020..2100
        return None
    tz_hours = None
    tzm = header.get("tzm")
    if isinstance(tzm, (int, float)) and -720 <= tzm <= 840:
        tz_hours = tzm / 60.0
    return (epoch_s, mono_now, tz_hours)


class DeebotAdapter:
    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self._device = None
        self._pos_raw: tuple[float, float, float] | None = None
        self._state = RobotState.UNKNOWN
        self._error = 0
        self._last_pos_t = 0.0
        # Monotonic instant of the last PROOF the cloud link is
        # alive: a command response or an incoming event.  The
        # runner's link watchdog reads it (see network.LinkWatchdog).
        self._last_contact_t: float | None = None
        self.battery_pct: int | None = None  # latest BatteryEvent value
        # Robot-reported UTC offset in hours (response header 'tzm'),
        # captured at connect; None when the capture failed.
        self.device_tz_offset_hours: float | None = None
        # Robot clock basis: (device epoch s, monotonic s at capture).
        # device_now() projects it forward on the monotonic clock, so
        # the AI's sense of 'now' is the ROBOT's time, immune to host
        # wall-clock skew and jumps (user 2026-11-11: AI 的时钟以
        # 扫地机自己上报的时间和时区为准，不可以写死).
        self._device_clock: tuple[float, float] | None = None
        self.water_amount: int | None = None  # latest WaterInfoEvent amount 1..4
        self.mop_attached: bool | None = None

    # ------------------------------------------------------------ convert
    def to_metres(self, raw_x: float, raw_y: float) -> Point:
        x = (raw_x - self.cfg.origin_raw_x) / self.cfg.raw_per_meter
        y = (raw_y - self.cfg.origin_raw_y) / self.cfg.raw_per_meter
        if self.cfg.invert_x:
            x = -x
        if self.cfg.invert_y:
            y = -y
        return Point(x, y)

    def to_raw_rect(self, centre: Point, half_m: float) -> str:
        """Return the 'x1,y1,x2,y2' content string CleanArea expects."""
        def raw_x(x_m: float) -> float:
            if self.cfg.invert_x:
                x_m = -x_m
            return x_m * self.cfg.raw_per_meter + self.cfg.origin_raw_x

        def raw_y(y_m: float) -> float:
            if self.cfg.invert_y:
                y_m = -y_m
            return y_m * self.cfg.raw_per_meter + self.cfg.origin_raw_y

        xs = sorted([raw_x(centre.x - half_m), raw_x(centre.x + half_m)])
        ys = sorted([raw_y(centre.y - half_m), raw_y(centre.y + half_m)])
        return f"{int(xs[0])},{int(ys[0])},{int(xs[1])},{int(ys[1])}"

    # -------------------------------------------------------------- setup
    def _register_hardware_aliases(self, log) -> None:
        """Alias unknown hardware classes to a known capability set.

        deebot-client keeps one module per hardware class in
        hardware/deebot/<class>.py, registered in a DEVICES dict. For an
        unknown class we import the template class's module and register
        its StaticDeviceInfo under the unknown class's name too.
        """
        if not self.cfg.hardware_aliases:
            return
        import importlib

        from deebot_client.hardware.deebot import DEVICES

        for unknown, template in self.cfg.hardware_aliases.items():
            if unknown in DEVICES:
                continue
            importlib.import_module(f"deebot_client.hardware.deebot.{template}")
            if template in DEVICES:
                info = DEVICES[template]
                if self.cfg.work_state_patch:
                    info = self._patch_state_to_work_state(info)
                DEVICES[unknown] = info
                log(f"hardware alias: {unknown} -> {template} capabilities")
            else:
                log(f"WARNING: hardware template {template} not found")

    @staticmethod
    def _patch_state_to_work_state(info):
        """Return a copy of `info` whose state capability reads state via
        GetChargeState + GetWorkState (no GetCleanInfo[V2]).

        Mirrors how deebot-client's newest hardware classes declare
        their state CapabilityEvent. Everything else - clean, position,
        error, battery - stays exactly the template's, and the template
        class's own registration is left untouched (frozen dataclasses,
        so `dataclasses.replace` builds new objects).
        """
        import dataclasses

        from deebot_client.capabilities import CapabilityEvent
        from deebot_client.commands.json.charge_state import GetChargeState
        from deebot_client.events import StateEvent

        from .work_state import GetWorkState, register_work_state_message

        # GetWorkState does not exist in the pinned deebot-client
        # (11.1.0b2) - it is vendored in bugnav.work_state; registering
        # it also routes onWorkState pushes to its handler.
        register_work_state_message()
        state = CapabilityEvent(StateEvent, [GetChargeState(), GetWorkState()])
        capabilities = dataclasses.replace(info.capabilities, state=state)
        return dataclasses.replace(info, capabilities=capabilities)

    async def connect(self, log) -> None:
        import aiohttp
        from deebot_client.api_client import ApiClient
        from deebot_client.authentication import Authenticator, create_rest_config
        from deebot_client.device import Device
        from deebot_client.events import (
            BatteryEvent,
            ErrorEvent,
            PositionsEvent,
            StateEvent,
            WaterInfoEvent,
        )
        from deebot_client.models import State
        from deebot_client.events import PositionType
        from deebot_client.mqtt_client import MqttClient, create_mqtt_config

        global _STATE_MAP
        _STATE_MAP = {
            State.IDLE: RobotState.IDLE,
            State.CLEANING: RobotState.CLEANING,
            State.RETURNING: RobotState.RETURNING,
            State.DOCKED: RobotState.DOCKED,
            State.ERROR: RobotState.ERROR,
            State.PAUSED: RobotState.PAUSED,
        }

        password = os.environ.get("ECOVACS_PASSWORD")
        if not password:
            raise RuntimeError(
                "ECOVACS_PASSWORD env var is not set; refusing to start. "
                "Set it in the systemd unit / compose env, never in config.json."
            )
        password_hash = hashlib.md5(password.encode()).hexdigest()

        self._register_hardware_aliases(log)
        # This firmware's getWaterInfo/getLifeSpan payloads predate
        # nothing - they postdate the pinned library, whose strict
        # parsers die on them.  Swap in tolerant handlers.
        from .firmware_parsers import install_tolerant_parsers

        install_tolerant_parsers()
        session = aiohttp.ClientSession()
        rest_cfg = create_rest_config(
            session,
            device_id=self.cfg.device_id,
            alpha_2_country=self.cfg.country,
        )
        authenticator = Authenticator(rest_cfg, self.cfg.account_id, password_hash)
        api = ApiClient(authenticator)
        devices = await api.get_devices()
        if devices.not_supported:
            names = [d.get("deviceName", "?") for d in devices.not_supported]
            log(
                f"FATAL: device(s) {names} are classified NOT SUPPORTED by "
                "deebot-client (unknown hardware class). The T80s may need a "
                "hardware-capability patch like the China T90 Pro one. "
                "Verification step 1 FAILED - see README."
            )
            raise RuntimeError("device not supported by deebot-client")
        if not devices.mqtt:
            raise RuntimeError("no MQTT devices found on this account")

        info = devices.mqtt[0]
        if self.cfg.device_name:
            for d in devices.mqtt:
                if self.cfg.device_name in str(d.api.get("deviceName", "")):
                    info = d
                    break
        self._device = Device(info, authenticator)
        mqtt_cfg = create_mqtt_config(
            device_id=self.cfg.device_id, country=self.cfg.country
        )
        mqtt = MqttClient(mqtt_cfg, authenticator)
        await self._device.initialize(mqtt)
        log(f"connected to device: {info.api.get('deviceName')} class={info.api.get('class')}")

        # The robot carries its own clock and timezone and stamps
        # every response header with them ('ts' = device epoch ms,
        # 'tzm' = offset minutes).  Verified NOT an echo of the
        # request header: a probe declaring tzm=0 was still answered
        # 480 (2026-11-11).  Capture both as the time basis for
        # everything downstream.
        try:
            from deebot_client.commands.json.battery import GetBattery

            raw_resp = await self._device.execute_command(GetBattery())
            self._note_device_clock(raw_resp)
            if self._device_clock is not None:
                log(f"robot clock captured: epoch "
                    f"{self._device_clock[0]:.0f}, timezone UTC"
                    f"{self.device_tz_offset_hours:+g}")
        except Exception:  # noqa: BLE001 - best effort; fallbacks exist
            pass

        async def on_pos(event) -> None:
            self._last_contact_t = time.monotonic()
            for p in event.positions:
                if p.type == PositionType.DEEBOT:
                    self._pos_raw = (p.x, p.y, p.a)
                    self._last_pos_t = time.monotonic()

        async def on_state(event) -> None:
            self._last_contact_t = time.monotonic()
            self._state = _STATE_MAP.get(event.state, RobotState.UNKNOWN)

        async def on_error(event) -> None:
            self._last_contact_t = time.monotonic()
            self._error = int(event.code or 0)

        async def on_battery(event) -> None:
            self._last_contact_t = time.monotonic()
            self.battery_pct = int(event.value)

        async def on_water(event) -> None:
            self._last_contact_t = time.monotonic()
            # Spray-amount setting (level), not a remaining level - the
            # firmware exposes no tank level at all.  If this firmware
            # rejects getWaterInfo the event simply never fires and
            # water_amount stays None; the supervisor's virtual water
            # gauge then falls back to its default amount.
            self.water_amount = int(event.amount)
            self.mop_attached = event.mop_attached

        self._device.events.subscribe(PositionsEvent, on_pos)
        self._device.events.subscribe(StateEvent, on_state)
        self._device.events.subscribe(ErrorEvent, on_error)
        self._device.events.subscribe(BatteryEvent, on_battery)
        self._device.events.subscribe(WaterInfoEvent, on_water)
        self._PositionsEvent = PositionsEvent
        self._StateEvent = StateEvent
        self._BatteryEvent = BatteryEvent
        self._WaterInfoEvent = WaterInfoEvent

    @property
    def last_contact_t(self) -> float | None:
        """Monotonic instant of the last proven cloud contact."""
        return self._last_contact_t

    async def ping(self) -> bool:
        """One cheap cloud round-trip (a battery read - a query, so
        it is observe-safe).  True when the cloud answered.  False
        on ANY failure: a token expired during a long outage fails
        here too, and the watchdog's answer to that is the same
        reconnect-with-fresh-login.  Deliberately no retry inside:
        the watchdog owns the pacing."""
        if self._device is None:
            return False
        try:
            from deebot_client.commands.json import GetBattery

            raw = await self._device.execute_command(GetBattery())
        except Exception:  # noqa: BLE001 - verdict is binary
            return False
        self._note_device_clock(raw)
        return True

    def _note_device_clock(self, raw) -> None:
        """Adopt the robot's clock basis from a command response."""
        self._last_contact_t = time.monotonic()
        header = ((raw or {}).get("resp", {}) or {}).get("header")
        parsed = parse_device_clock(header, time.monotonic())
        if parsed is None:
            return
        epoch_s, mono_now, tz_hours = parsed
        self._device_clock = (epoch_s, mono_now)
        if tz_hours is not None:
            self.device_tz_offset_hours = tz_hours

    def device_now(self) -> float | None:
        """Current time on the ROBOT's clock (epoch s), or None if
        no response header has been captured yet.  Projects the
        captured basis forward on the monotonic clock, so it tracks
        the device even between responses and never inherits host
        wall-clock jumps."""
        if self._device_clock is None:
            return None
        epoch0, mono0 = self._device_clock
        return epoch0 + (time.monotonic() - mono0)

    def device_now_at(self, mono_ts: float) -> float | None:
        """The robot-clock epoch at ANY monotonic instant of this
        process - including instants BEFORE the basis was captured:
        the (epoch, monotonic) pair defines the whole timeline, so
        startup lines logged before the first header landed can
        still be stamped on the robot's clock once it exists."""
        if self._device_clock is None:
            return None
        epoch0, mono0 = self._device_clock
        return epoch0 + (mono_ts - mono0)

    def request_refresh(self) -> None:
        if self._device is not None:
            self._device.events.request_refresh(self._PositionsEvent)
            self._device.events.request_refresh(self._StateEvent)
            self._device.events.request_refresh(self._BatteryEvent)
            self._device.events.request_refresh(self._WaterInfoEvent)

    # ------------------------------------------------------------- sample
    def sample(self, t: float) -> Sample | None:
        if self._pos_raw is None:
            return None
        x, y, a = self._pos_raw
        return Sample(
            t=t,
            pos=self.to_metres(x, y),
            state=self._state,
            heading_deg=float(a) if a is not None else None,
            error_code=self._error,
        )

    # ------------------------------------------------------------ command
    async def execute(self, cmd: Command, log) -> None:
        from deebot_client.commands.json.clean import (
            Clean,
            CleanArea,
            CleanAreaV2,
            CleanV2,
        )
        from deebot_client.models import CleanAction, CleanMode

        assert self._device is not None

        async def _exec(command) -> None:
            async def _once() -> None:
                raw = await self._device.execute_command(command)
                self._note_device_clock(raw)

            # A cloud hiccup mid-command (proxy restart, router
            # reboot) is retried at a paced interval instead of
            # surfacing as a traceback or a lost command.
            await call_with_retry(
                _once, log=log,
                what=f"command {type(command).__name__}")

        # V2 firmwares (like the T80S) reject the V1 "clean" command the
        # same way they reject getCleanInfo - then EVERY action, pause
        # and resume included, must go through clean_V2.
        clean_cls = CleanV2 if self.cfg.clean_command_v2 else Clean
        if cmd.kind == CommandKind.PAUSE:
            await _exec(clean_cls(CleanAction.PAUSE))
            log("sent pause")
        elif cmd.kind == CommandKind.STOP:
            await _exec(clean_cls(CleanAction.STOP))
            log("sent stop")
        elif cmd.kind == CommandKind.RESUME_AUTO:
            await _exec(clean_cls(CleanAction.START))
            log("sent resume (auto clean start)")
        elif cmd.kind == CommandKind.RETURN_TO_DOCK:
            from deebot_client.commands.json import Charge

            await _exec(Charge())
            log("sent return-to-dock (charge)")
        elif cmd.kind == CommandKind.CLEAN_AREA and cmd.point is not None:
            rect = self.to_raw_rect(cmd.point, cmd.rect_half_m)
            cls = CleanAreaV2 if self.cfg.clean_command_v2 else CleanArea
            await _exec(cls(CleanMode.CUSTOM_AREA, rect, 1))
            log(f"sent CUSTOM_AREA rect={rect}")
        elif cmd.kind == CommandKind.PLAY_SOUND:
            from deebot_client.commands.json import PlaySound

            # PlaySound is the robot's built-in locate/alert tone
            # (sid 30, the only sound deebot-client addresses; the
            # firmware offers no custom voice - see README).  Repeat
            # with pauses so it registers as an alarm, not a blip.
            n = max(1, cmd.count)
            for i in range(n):
                await _exec(PlaySound())
                if i + 1 < n:
                    await asyncio.sleep(1.5)
            log(f"played robot alert sound x{n}")
        elif cmd.kind == CommandKind.NOTIFY:
            log(f"NOTIFY: {cmd.message}")
