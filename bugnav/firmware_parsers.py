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

The patch replaces the two handler classmethods on the library's own
command classes, so every dispatch route (message registry, legacy
fallback, request/response pairing - they all reference the same
class objects) picks it up.  Idempotent.
"""

from __future__ import annotations

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


def install_tolerant_parsers() -> None:
    """Replace the two strict handlers with the tolerant ones."""
    from deebot_client.commands.json.life_span import GetLifeSpan
    from deebot_client.commands.json.water_info import GetWaterInfo

    GetWaterInfo._handle_body_data_dict = classmethod(_tolerant_water_info)
    GetLifeSpan._handle_body_data_list = classmethod(_tolerant_life_span)
