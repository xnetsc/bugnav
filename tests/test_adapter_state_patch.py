"""Tests for the hardware-alias state patch.

The T80S firmware (1.61.0) rejects getCleanInfo V1 and V2 with
`20003 rcp not support`, so a state capability that reads work state
from GetCleanInfoV2 never updates (state
stayed "docked" through an entire cleaning run). Aliased classes get
their state capability replaced with GetChargeState + GetWorkState,
mirroring deebot-client's newest hardware classes.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bugnav.deebot_adapter import AdapterConfig, DeebotAdapter


class TestWorkStatePatch(unittest.TestCase):
    def _register(self, **cfg_kwargs):
        from deebot_client.hardware.deebot import DEVICES

        DEVICES.pop("nazy54", None)
        cfg = AdapterConfig(account_id="EXXXXXXXX", **cfg_kwargs)
        adapter = DeebotAdapter(cfg)
        logs = []
        adapter._register_hardware_aliases(logs.append)
        return DEVICES, logs

    def test_state_capability_uses_work_state(self):
        from deebot_client.commands.json.charge_state import GetChargeState
        from deebot_client.commands.json.clean import (
            GetCleanInfo,
            GetCleanInfoV2,
        )

        from bugnav.work_state import GetWorkState

        devices, _ = self._register()
        self.assertIn("nazy54", devices)
        get_cmds = devices["nazy54"].capabilities.state.get
        self.assertEqual(
            [type(c) for c in get_cmds], [GetChargeState, GetWorkState]
        )
        self.assertFalse(
            any(isinstance(c, (GetCleanInfo, GetCleanInfoV2)) for c in get_cmds)
        )

    def test_template_registration_untouched(self):
        from deebot_client.commands.json.clean import GetCleanInfoV2

        devices, _ = self._register()
        template_cmds = devices["9kpees"].capabilities.state.get
        self.assertTrue(
            any(isinstance(c, GetCleanInfoV2) for c in template_cmds)
        )

    def test_rest_of_capabilities_unchanged(self):
        from deebot_client.commands.json.clean import CleanAreaV2, CleanV2

        devices, _ = self._register()
        caps = devices["nazy54"].capabilities
        template = devices["9kpees"].capabilities
        # Clean action must stay the V2 pair the T80S accepts.
        self.assertIs(caps.clean.action.command, CleanV2)
        self.assertIs(caps.clean.action.area, CleanAreaV2)
        # Other capabilities are the very same objects as the template's.
        self.assertIs(caps.map, template.map)
        self.assertIs(caps.error, template.error)
        self.assertIs(caps.battery, template.battery)

    def test_patch_can_be_disabled(self):
        from deebot_client.commands.json.clean import GetCleanInfoV2

        devices, _ = self._register(work_state_patch=False)
        get_cmds = devices["nazy54"].capabilities.state.get
        self.assertTrue(any(isinstance(c, GetCleanInfoV2) for c in get_cmds))

    def test_message_dispatcher_registration(self):
        self._register()
        from deebot_client.messages.json import MESSAGES

        from bugnav.work_state import GetWorkState

        self.assertIs(MESSAGES.get("getWorkState"), GetWorkState)
        self.assertIs(MESSAGES.get("onWorkState"), GetWorkState)


class TestWorkStateHandler(unittest.TestCase):
    def _states_for(self, data):
        from bugnav.work_state import GetWorkState

        events = []

        class Bus:
            def notify(self, event):
                events.append(event)

        GetWorkState._handle_body_data_dict(Bus(), data)
        return [e.state for e in events]

    def test_cleaning(self):
        from deebot_client.models import State

        states = self._states_for(
            {"robotState": {"state": "cleaning"},
             "stationState": {"state": "idle"}}
        )
        self.assertEqual(states, [State.CLEANING])

    def test_returning_and_docked_at_station(self):
        from deebot_client.models import State

        self.assertEqual(
            self._states_for(
                {"robotState": {"state": "cleaning"},
                 "stationState": {"state": "goCharging"}}
            ),
            [State.RETURNING],
        )
        self.assertEqual(
            self._states_for(
                {"robotState": {"state": "idle"},
                 "stationState": {"state": "washing"}}
            ),
            [State.DOCKED],
        )

    def test_paused_overrides(self):
        from deebot_client.models import State

        states = self._states_for(
            {"robotState": {"state": "cleaning"},
             "stationState": {"state": "idle"}, "paused": 1}
        )
        self.assertEqual(states, [State.PAUSED])

    def test_idle_at_station_is_silent(self):
        states = self._states_for(
            {"robotState": {"state": "idle"},
             "stationState": {"state": "idle"}}
        )
        self.assertEqual(states, [])


if __name__ == "__main__":
    unittest.main()
