from __future__ import annotations

import unittest
from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.config import MonitorConfig, clip_roi_to_frame, roi_area


class RoiHelpersTest(unittest.TestCase):
    def test_clip_roi_inside_frame(self) -> None:
        self.assertEqual(clip_roi_to_frame((10, 20, 30, 40), (100, 200, 3)), (10, 20, 30, 40))

    def test_clip_roi_at_frame_edges(self) -> None:
        self.assertEqual(clip_roi_to_frame((80, 70, 50, 50), (100, 120, 3)), (80, 70, 40, 30))

    def test_clip_roi_outside_frame_returns_none(self) -> None:
        self.assertIsNone(clip_roi_to_frame((150, 10, 20, 20), (100, 120, 3)))

    def test_roi_area_uses_clipped_dimensions(self) -> None:
        self.assertEqual(roi_area((80, 70, 50, 50), (100, 120, 3)), 1200)


class MonitorConfigValidationTest(unittest.TestCase):
    def test_validate_rejects_invalid_background_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "history"):
            MonitorConfig(source="0", output_dir=Path("data/alerts"), history=0).validate()

    def test_validate_rejects_invalid_learning_rate(self) -> None:
        for learning_rate in (-0.5, 1.5):
            with self.subTest(learning_rate=learning_rate):
                with self.assertRaisesRegex(ValueError, "learning_rate"):
                    MonitorConfig(
                        source="0",
                        output_dir=Path("data/alerts"),
                        learning_rate=learning_rate,
                    ).validate()

    def test_validate_rejects_invalid_tracking_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "loitering_seconds"):
            MonitorConfig(
                source="0",
                output_dir=Path("data/alerts"),
                loitering_seconds=0,
            ).validate()

    def test_validate_rejects_empty_alert_clip_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "alert clip seconds"):
            MonitorConfig(
                source="0",
                output_dir=Path("data/alerts"),
                pre_alert_seconds=0,
                post_alert_seconds=0,
            ).validate()

    def test_validate_rejects_invalid_face_engine(self) -> None:
        with self.assertRaisesRegex(ValueError, "face_engine"):
            MonitorConfig(
                source="0",
                output_dir=Path("data/alerts"),
                face_engine="other",
            ).validate()

    def test_validate_rejects_invalid_arcface_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "arcface_similarity_threshold"):
            MonitorConfig(
                source="0",
                output_dir=Path("data/alerts"),
                arcface_similarity_threshold=0,
            ).validate()


if __name__ == "__main__":
    unittest.main()
