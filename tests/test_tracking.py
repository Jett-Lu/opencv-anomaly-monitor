from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.tracking import PersonMotionTracker, TrackObservation


def tracker() -> PersonMotionTracker:
    return PersonMotionTracker(
        match_distance=0.2,
        lost_seconds=2.0,
        history_seconds=10.0,
        loitering_seconds=3.0,
        loitering_radius=0.08,
        roi_dwell_seconds=2.0,
        repeated_motion_distance=0.3,
        repeated_motion_radius=0.1,
        rapid_body_speed_threshold=0.5,
    )


class PersonMotionTrackerTest(unittest.TestCase):
    def test_keeps_track_id_for_nearby_observations(self) -> None:
        motion_tracker = tracker()
        first = motion_tracker.update([TrackObservation(index=0, center=(0.5, 0.5))], now=0.0)
        second = motion_tracker.update([TrackObservation(index=0, center=(0.55, 0.5))], now=1.0)

        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_loitering_flags_when_person_stays_near_start(self) -> None:
        motion_tracker = tracker()
        motion_tracker.update([TrackObservation(index=0, center=(0.5, 0.5))], now=0.0)
        motion_tracker.update([TrackObservation(index=0, center=(0.51, 0.5))], now=1.0)
        motion_tracker.update([TrackObservation(index=0, center=(0.52, 0.5))], now=2.0)
        result = motion_tracker.update([TrackObservation(index=0, center=(0.52, 0.5))], now=3.1)

        self.assertIn("loitering", result[0].labels)
        self.assertGreaterEqual(result[0].score, 0.75)

    def test_roi_dwell_flags_after_threshold(self) -> None:
        motion_tracker = tracker()
        motion_tracker.update(
            [TrackObservation(index=0, center=(0.5, 0.5), inside_roi=True)],
            now=0.0,
        )
        motion_tracker.update(
            [TrackObservation(index=0, center=(0.51, 0.5), inside_roi=True)],
            now=1.0,
        )
        result = motion_tracker.update(
            [TrackObservation(index=0, center=(0.51, 0.5), inside_roi=True)],
            now=2.1,
        )

        self.assertIn("restricted_zone_dwell", result[0].labels)
        self.assertGreaterEqual(result[0].score, 0.8)

    def test_repeated_motion_flags_back_and_forth_path(self) -> None:
        motion_tracker = tracker()
        positions = [(0.5, 0.5), (0.65, 0.5), (0.5, 0.5), (0.65, 0.5), (0.5, 0.5)]
        result = {}
        for second, position in enumerate(positions):
            result = motion_tracker.update(
                [TrackObservation(index=0, center=position)],
                now=second,
            )

        self.assertIn("repeated_motion", result[0].labels)
        self.assertGreaterEqual(result[0].path_length, 0.3)


if __name__ == "__main__":
    unittest.main()
