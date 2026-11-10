"""Real-robot adapter built on the unofficial `deebot-client` library.

Verified against deebot-client's source (Nov 2026):
  * Authenticator + create_rest_config + ApiClient.get_devices()
  * Device(device_info, authenticator); await device.initialize(mqtt)
  * PositionsEvent carries Position(x, y, a) - raw map units + heading
  * StateEvent / ErrorEvent for status and faults
  * Clean(CleanAction.PAUSE/RESUME/START/STOP) and
    CleanArea(CleanMode.CUSTOM_AREA, [x1, y1, x2, y2]) for area commands

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

import hashlib
import os
import time
from dataclasses import dataclass

from .geometry import Point
from .model import Command, CommandKind, RobotState, Sample


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


_STATE_MAP = {}


class DeebotAdapter:
    def __init__(self, cfg: AdapterConfig):
        self.cfg = cfg
        self._device = None
        self._pos_raw: tuple[float, float, float] | None = None
        self._state = RobotState.UNKNOWN
        self._error = 0
        self._last_pos_t = 0.0

    # ------------------------------------------------------------ convert
    def to_metres(self, raw_x: float, raw_y: float) -> Point:
        x = (raw_x - self.cfg.origin_raw_x) / self.cfg.raw_per_meter
        y = (raw_y - self.cfg.origin_raw_y) / self.cfg.raw_per_meter
        if self.cfg.invert_x:
            x = -x
        if self.cfg.invert_y:
            y = -y
        return Point(x, y)

    def to_raw_rect(self, centre: Point, half_m: float) -> list[int]:
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
        return [int(xs[0]), int(ys[0]), int(xs[1]), int(ys[1])]

    # -------------------------------------------------------------- setup
    async def connect(self, log) -> None:
        import aiohttp
        from deebot_client.api_client import ApiClient
        from deebot_client.authentication import Authenticator, create_rest_config
        from deebot_client.device import Device
        from deebot_client.events import ErrorEvent, PositionsEvent, StateEvent
        from deebot_client.models import State
        from deebot_client.mqtt_client import MqttClient, create_mqtt_config
        from deebot_client.rs.map import PositionType

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

        async def on_pos(event) -> None:
            for p in event.positions:
                if p.type == PositionType.DEEBOT:
                    self._pos_raw = (p.x, p.y, p.a)
                    self._last_pos_t = time.monotonic()

        async def on_state(event) -> None:
            self._state = _STATE_MAP.get(event.state, RobotState.UNKNOWN)

        async def on_error(event) -> None:
            self._error = int(event.code or 0)

        self._device.events.subscribe(PositionsEvent, on_pos)
        self._device.events.subscribe(StateEvent, on_state)
        self._device.events.subscribe(ErrorEvent, on_error)
        self._PositionsEvent = PositionsEvent
        self._StateEvent = StateEvent

    def request_refresh(self) -> None:
        if self._device is not None:
            self._device.events.request_refresh(self._PositionsEvent)
            self._device.events.request_refresh(self._StateEvent)

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
        from deebot_client.commands.json.clean import Clean, CleanArea, CleanAreaV2
        from deebot_client.models import CleanAction, CleanMode

        assert self._device is not None
        if cmd.kind == CommandKind.PAUSE:
            await self._device.execute_command(Clean(CleanAction.PAUSE))
        elif cmd.kind == CommandKind.STOP:
            await self._device.execute_command(Clean(CleanAction.STOP))
        elif cmd.kind == CommandKind.RESUME_AUTO:
            await self._device.execute_command(Clean(CleanAction.START))
        elif cmd.kind == CommandKind.CLEAN_AREA and cmd.point is not None:
            rect = self.to_raw_rect(cmd.point, cmd.rect_half_m)
            cls = CleanAreaV2 if self.cfg.clean_command_v2 else CleanArea
            await self._device.execute_command(
                cls(CleanMode.CUSTOM_AREA, rect, 1)
            )
            log(f"sent CUSTOM_AREA rect={rect}")
        elif cmd.kind == CommandKind.NOTIFY:
            log(f"NOTIFY: {cmd.message}")
