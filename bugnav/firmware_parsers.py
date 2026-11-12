"""Tolerant response parsers for the T80S firmware (fw 1.61.0).

The pinned deebot-client (11.1.0b2) predates this firmware's payload
shapes, and its strict parsers die on them - losing the data AND
spamming a traceback on every refresh (observed live 2026-11-11):

* getWaterInfo: b2 unconditionally reads data["amount"], but this
  firmware reports the water level as a CUSTOM amount and has no
  "amount" key at all:
      {"customAmount": 30, "enable": 1, "mopCount": 2,
       "sideMop": 0, "sweepType": 1, "type": 1}
  (Upstream's newer parser treats customAmount as its own reading
  too - a device reporting customAmount is in custom-amount mode.)
  bugnav's gauge needs the 1..4 band, so the custom value (assumed
  0..100) is mapped onto bands by quartiles until the scale can be
  calibrated against the app.
* getLifeSpan: b2 builds LifeSpan(component["type"]) for every
  component and raises on the first type newer than the library
  (cleaningSolution, sewageBox, ...), discarding the whole list.
  The tolerant handler skips unknown components and keeps the rest.
* resetLifeSpan over MQTT p2p: commands other clients send are
  mirrored to every client and rebuilt by
  CommandMqttP2P.create_from_mqtt, whose argument conversion raises
  DeebotError on the same unknown LifeSpan values - observed live
  2026-11-11 with a mirrored resetLifeSpan for 'cleaningSolution',
  dropped with a traceback.  The converter gets a LifeSpan-only
  fallback that hands back a value-carrying stand-in.
* getMapSet: this firmware permanently answers 20003 ("rcp not
  support") - map sets are simply unreadable here (verified during
  the 2026-11-11 map probes).  The library retries it on every
  map-related dispatch route and logs "Could not parse getMapSet"
  with a full traceback each time (bursts observed live 2026-11-12
  at connect).  A logging filter drops exactly that record; every
  other library warning still shows.

The patch replaces the two handler classmethods on the library's own
command classes, so every dispatch route (message registry, legacy
fallback, request/response pairing - they all reference the same
class objects) picks it up.  Idempotent.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

from deebot_client.message import HandlingResult

if TYPE_CHECKING:
    from deebot_client.event_bus import EventBus


def _water_amount(data: dict[str, Any]):
    """WaterAmount for this payload, or None if it carries none."""
    from deebot_client.events import WaterAmount

    if data.get("amount") is not None:
        return WaterAmount(int(data["amount"]))
    custom = data.get("customAmount")
    if custom is not None:
        v = max(0, min(100, int(custom)))
        return WaterAmount(max(1, math.ceil(v / 25)))
    return None


def _tolerant_water_info(cls, event_bus: EventBus,
                         data: dict[str, Any]) -> HandlingResult:
    from deebot_client.events import SweepType, WaterInfoEvent

    try:
        amount = _water_amount(data)
    except (TypeError, ValueError):
        return HandlingResult.analyse()
    if amount is None:
        return HandlingResult.analyse()
    mop_attached = data.get("enable")
    if mop_attached is not None:
        mop_attached = bool(mop_attached)
    sweep_type = None
    if data.get("sweepType"):
        try:
            sweep_type = SweepType(int(data["sweepType"]))
        except (TypeError, ValueError):
            sweep_type = None
    event_bus.notify(
        WaterInfoEvent(amount, sweep_type, mop_attached=mop_attached))
    return HandlingResult.success()


def _tolerant_life_span(cls, event_bus: EventBus,
                        data: list[dict[str, Any]]) -> HandlingResult:
    from deebot_client.events import LifeSpan, LifeSpanEvent

    for component in data:
        try:
            component_type = LifeSpan(component["type"])
            left = int(component["left"])
            total = int(component["total"])
            if total <= 0:
                continue
        except (KeyError, TypeError, ValueError):
            continue  # newer than the pinned library: skip, keep rest
        percent = round((left / total) * 100, 2)
        event_bus.notify(LifeSpanEvent(component_type, percent, left))
    return HandlingResult.success()


class _UnknownLifeSpan:
    """Stand-in for a LifeSpan member the pinned library predates.

    Only ``.value`` is read on the path that needs it
    (ResetLifeSpan.__init__ builds {"type": life_span.value}), and a
    mirrored command is never executed by this client - at most it
    triggers a LifeSpanEvent refresh, whose response the tolerant
    body handler above already filters."""

    def __init__(self, value) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"<unknown LifeSpan {self.value!r}>"


def _install_tolerant_p2p_lifespan() -> None:
    """Let p2p command reconstruction survive new LifeSpan values.

    Wraps deebot_client.command._pop_or_raise: a LifeSpan conversion
    failure falls back to _UnknownLifeSpan; every other type keeps
    the library's strict behaviour.  Idempotent."""
    import deebot_client.command as command_mod
    from deebot_client.events import LifeSpan
    from deebot_client.exceptions import DeebotError

    orig = command_mod._pop_or_raise
    if getattr(orig, "_bugnav_tolerant", False):
        return

    def tolerant(name, type_, data):
        if type_ is not LifeSpan:
            return orig(name, type_, data)
        value = data.get(name)  # orig pops before converting; peek
        try:
            return orig(name, type_, data)
        except DeebotError:
            return _UnknownLifeSpan(value)

    tolerant._bugnav_tolerant = True  # type: ignore[attr-defined]
    command_mod._pop_or_raise = tolerant


class _UnsupportedMapSetFilter(logging.Filter):
    """Drop the library's getMapSet parse warnings - and only those.

    The firmware answers getMapSet with 20003 on every dispatch
    route, so each answer is one "Could not parse getMapSet" warning
    (the message-dispatch ones carry a full traceback).  Map sets
    being unreadable is a settled fact about this firmware, not a
    malfunction worth a log line; all other records pass."""

    _bugnav_filter = True

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not message.startswith("Could not parse getMapSet")


def _install_log_filters() -> None:
    """Attach the getMapSet noise filter to the two library loggers
    that emit it (message dispatch and command responses).
    Idempotent."""
    for name in ("deebot_client.message", "deebot_client.command"):
        logger = logging.getLogger(name)
        if any(getattr(f, "_bugnav_filter", False) for f in logger.filters):
            continue
        logger.addFilter(_UnsupportedMapSetFilter())


def install_tolerant_parsers() -> None:
    """Replace the two strict handlers with the tolerant ones."""
    from deebot_client.commands.json.life_span import GetLifeSpan
    from deebot_client.commands.json.water_info import GetWaterInfo

    GetWaterInfo._handle_body_data_dict = classmethod(_tolerant_water_info)
    GetLifeSpan._handle_body_data_list = classmethod(_tolerant_life_span)
    _install_tolerant_p2p_lifespan()
    _install_log_filters()
