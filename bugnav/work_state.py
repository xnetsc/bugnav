"""Vendored GetWorkState command for deebot-client 11.1.0b2.

Upstream deebot-client only gained GetWorkState in 12.x, which also
moved to Python 3.13; bugnav pins 11.1.0b2 on Python 3.12 (the last
release line with a compatible API for the adapter). Newer firmwares
like the T80S (fw 1.61.0) dropped getCleanInfo entirely - both V1 and
V2 are answered `20003 rcp not support` - and report the work state
only via getWorkState / onWorkState, so the command is vendored here
in the exact shape 11.1.0b2 uses for its own commands (cf.
GetChargeState): the class is both the get command and, registered in
the JSON MESSAGES dict under both names, the handler for its
responses ("getWorkState") and for pushes ("onWorkState" - the
dispatcher's on->get rewrite only feeds its legacy fallback, not the
MESSAGES lookup, so both names must be registered).

The robot-state mapping mirrors upstream's OnWorkState table
(robotState x stationState -> State); the station event itself is not
forwarded - bugnav has no StationEvent subscriber.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deebot_client.commands.json.common import JsonCommandWithMessageHandling
from deebot_client.events import StateEvent
from deebot_client.message import HandlingResult, MessageBodyDataDict
from deebot_client.models import State

if TYPE_CHECKING:
    from deebot_client.event_bus import EventBus

# robotState.state -> stationState.state -> robot State (None = no event)
_WORK_STATE_2_ROBOT_STATE: dict[str, dict[str, State | None]] = {
    "idle": {
        "idle": None,
        "goCharging": State.RETURNING,
        "goEmptying": State.RETURNING,
        "emptying": State.DOCKED,
        "washing": State.DOCKED,
        "drying": State.DOCKED,
    },
    "cleaning": {
        "idle": State.CLEANING,
        "goCharging": State.RETURNING,
        "goEmptying": State.RETURNING,
        "emptying": State.DOCKED,
        "washing": State.DOCKED,
        "drying": State.DOCKED,
    },
}


class GetWorkState(JsonCommandWithMessageHandling, MessageBodyDataDict):
    """Get work state command (vendored; missing in deebot-client 11.1.0b2)."""

    NAME = "getWorkState"

    @classmethod
    def _handle_body_data_dict(
        cls, event_bus: EventBus, data: dict[str, Any]
    ) -> HandlingResult:
        """Handle message->body->data and notify StateEvent subscribers.

        :return: A message response
        """
        robot_state = data.get("robotState", {}).get("state")
        station_state = data.get("stationState", {}).get("state")

        row = _WORK_STATE_2_ROBOT_STATE.get(robot_state, {})
        if station_state not in row:
            return HandlingResult.analyse()

        status = row[station_state]
        if data.get("paused") == 1:
            status = State.PAUSED

        if status is not None:
            event_bus.notify(StateEvent(status))
        return HandlingResult.success()


def register_work_state_message() -> None:
    """Make the JSON message dispatcher route work-state messages to
    GetWorkState (idempotent).

    deebot-client 11.1.0b2 dispatches incoming messages by exact name
    via deebot_client.messages.json.MESSAGES (its on->get rewrite only
    feeds a legacy fallback list that has no work-state entry), so
    register both the response name and the push name.
    """
    from deebot_client.messages.json import MESSAGES

    MESSAGES.setdefault(GetWorkState.NAME, GetWorkState)
    MESSAGES.setdefault("onWorkState", GetWorkState)
