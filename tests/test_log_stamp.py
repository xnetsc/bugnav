"""Log-line timestamps come from the COMPUTED time (the robot's
clock in the robot's zone); only before the first device header
lands may the caller fall back - to the container's local time
(user 2026-11-11: 改为计算后的时间做时间戳打印；没有取到扫地机
时间时用容器本地时间)."""

import unittest

from bugnav.runner import log_stamp


class TestLogStamp(unittest.TestCase):
    def test_robot_epoch_rendered_in_robot_zone(self):
        # 1794375294 = 2026-11-11 05:34:54 UTC (the live capture).
        self.assertEqual(log_stamp(1794375294.0, 8.0), "13:34:54")
        self.assertEqual(log_stamp(1794375294.0, 0.0), "05:34:54")

    def test_no_basis_returns_none(self):
        self.assertIsNone(log_stamp(None, 8.0))


if __name__ == "__main__":
    unittest.main()


class TestPosixTz(unittest.TestCase):
    def test_built_from_offset_not_hardcoded(self):
        from bugnav.runner import _posix_tz
        self.assertEqual(_posix_tz(8.0), "LOC-8")    # UTC+8
        self.assertEqual(_posix_tz(-5.0), "LOC+5")   # UTC-5
        self.assertEqual(_posix_tz(0.0), "LOC0")
        self.assertEqual(_posix_tz(5.5), "LOC-5:30")  # fractional


class TestCommandExecutable(unittest.TestCase):
    def test_mode_matrix(self):
        from bugnav.model import CommandKind
        from bugnav.runner import command_executable
        for kind in CommandKind:
            self.assertFalse(command_executable(kind, "observe"))
            self.assertTrue(command_executable(kind, "active"))
        # supervised: the safety chain is real, navigation is not.
        self.assertTrue(command_executable(
            CommandKind.PAUSE, "supervised"))
        self.assertTrue(command_executable(
            CommandKind.STOP, "supervised"))
        self.assertTrue(command_executable(
            CommandKind.RETURN_TO_DOCK, "supervised"))
        self.assertTrue(command_executable(
            CommandKind.RESUME_AUTO, "supervised"))
        self.assertTrue(command_executable(
            CommandKind.PLAY_SOUND, "supervised"))
        self.assertFalse(command_executable(
            CommandKind.CLEAN_AREA, "supervised"))
