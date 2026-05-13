from __future__ import annotations

import sys
import unittest
import os
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
matplotlib_cache = Path(__file__).resolve().parents[1] / ".tmp" / "matplotlib"
matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

import mediapipe as mp

from anomaly_monitor.pose import PoseBehaviorAnalyzer


def pose_index(name: str) -> int:
    return getattr(mp.tasks.vision.PoseLandmark, name).value


class TPoseDetectionTest(unittest.TestCase):
    def test_t_pose_scores_as_alert(self) -> None:
        analyzer = PoseBehaviorAnalyzer.__new__(PoseBehaviorAnalyzer)
        analyzer.t_pose_only = True

        landmarks = {
            pose_index("LEFT_SHOULDER"): (0.42, 0.35),
            pose_index("RIGHT_SHOULDER"): (0.58, 0.35),
            pose_index("LEFT_ELBOW"): (0.28, 0.36),
            pose_index("RIGHT_ELBOW"): (0.72, 0.36),
            pose_index("LEFT_WRIST"): (0.18, 0.36),
            pose_index("RIGHT_WRIST"): (0.82, 0.36),
        }

        score, labels = analyzer._score_behavior(
            landmarks=landmarks,
            frame_shape=(480, 640, 3),
            previous_landmarks=None,
            now=0.0,
        )

        self.assertEqual(score, 1.0)
        self.assertEqual(labels, ["t_pose"])

    def test_normal_pose_scores_as_clear_in_test_mode(self) -> None:
        analyzer = PoseBehaviorAnalyzer.__new__(PoseBehaviorAnalyzer)
        analyzer.t_pose_only = True

        landmarks = {
            pose_index("LEFT_SHOULDER"): (0.42, 0.35),
            pose_index("RIGHT_SHOULDER"): (0.58, 0.35),
            pose_index("LEFT_ELBOW"): (0.42, 0.50),
            pose_index("RIGHT_ELBOW"): (0.58, 0.50),
            pose_index("LEFT_WRIST"): (0.42, 0.65),
            pose_index("RIGHT_WRIST"): (0.58, 0.65),
        }

        score, labels = analyzer._score_behavior(
            landmarks=landmarks,
            frame_shape=(480, 640, 3),
            previous_landmarks=None,
            now=0.0,
        )

        self.assertEqual(score, 0.0)
        self.assertEqual(labels, [])

    def test_bounding_box_wraps_visible_landmarks(self) -> None:
        analyzer = PoseBehaviorAnalyzer.__new__(PoseBehaviorAnalyzer)
        box = analyzer._bounding_box(
            landmarks={
                pose_index("LEFT_SHOULDER"): (0.4, 0.3),
                pose_index("RIGHT_WRIST"): (0.8, 0.7),
            },
            frame_shape=(100, 200, 3),
        )

        self.assertEqual(box, (72, 24, 96, 52))


if __name__ == "__main__":
    unittest.main()
