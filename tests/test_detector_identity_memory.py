from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.config import MonitorConfig
from anomaly_monitor.detector import MotionAnomalyDetector
from anomaly_monitor.pose import PosePersonBehavior


def person(score: float, labels: list[str]) -> PosePersonBehavior:
    return PosePersonBehavior(
        index=0,
        score=score,
        labels=labels,
        anchor=(100, 100),
        center=(0.5, 0.5),
        box=(80, 80, 120, 220),
    )


class DetectorIdentityMemoryTest(unittest.TestCase):
    def test_flagged_unknown_identity_is_remembered_when_it_reappears(self) -> None:
        detector = MotionAnomalyDetector.__new__(MotionAnomalyDetector)
        detector.config = MonitorConfig(
            source="0",
            output_dir=Path("data/alerts"),
            identity_alert_hold_seconds=300,
        )
        detector.person_alerts = {}
        detector.identity_alerts = {}

        active = detector._update_person_alerts(
            people=[person(1.0, ["t_pose"])],
            tracking_behaviors={},
            matched_names={0: "Unknown A"},
        )
        self.assertEqual(active[0], ["t_pose"])
        self.assertIn("Unknown A", detector.identity_alerts)

        active = detector._update_person_alerts(
            people=[person(0.0, [])],
            tracking_behaviors={},
            matched_names={0: "Unknown A"},
        )
        self.assertEqual(active[0], ["t_pose"])

    def test_generic_unidentified_person_label_is_not_remembered(self) -> None:
        detector = MotionAnomalyDetector.__new__(MotionAnomalyDetector)
        detector.config = MonitorConfig(source="0", output_dir=Path("data/alerts"))
        detector.person_alerts = {}
        detector.identity_alerts = {}

        detector._update_person_alerts(
            people=[person(1.0, ["t_pose"])],
            tracking_behaviors={},
            matched_names={0: "Person 1"},
        )

        self.assertEqual(detector.identity_alerts, {})


if __name__ == "__main__":
    unittest.main()
