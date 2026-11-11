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
