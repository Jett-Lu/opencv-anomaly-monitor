from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.main import save_alert_snapshot, save_event_video


class AlertMediaTest(unittest.TestCase):
    def test_save_alert_snapshot_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            path = save_alert_snapshot(Path(directory), 7, frame)

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)

    def test_save_event_video_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(3)]
            path = save_event_video(Path(directory), 7, frames, fps=3.0)

            self.assertIsNotNone(path)
            assert path is not None
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
