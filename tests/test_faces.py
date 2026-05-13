from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.faces import UnknownFaceMemory


class UnknownFaceMemoryTest(unittest.TestCase):
    def test_reuses_unknown_label_for_similar_face(self) -> None:
        memory = UnknownFaceMemory(match_threshold=10.0)
        face = np.full((160, 160), 80, dtype=np.uint8)

        first_name, first_distance = memory.identify(face)
        second_name, second_distance = memory.identify(face.copy())

        self.assertEqual(first_name, "Unknown A")
        self.assertIsNone(first_distance)
        self.assertEqual(second_name, "Unknown A")
        self.assertEqual(second_distance, 0.0)

    def test_creates_new_unknown_label_for_different_face(self) -> None:
        memory = UnknownFaceMemory(match_threshold=10.0)

        first_name, _ = memory.identify(np.full((160, 160), 20, dtype=np.uint8))
        second_name, _ = memory.identify(np.full((160, 160), 220, dtype=np.uint8))

        self.assertEqual(first_name, "Unknown A")
        self.assertEqual(second_name, "Unknown B")


if __name__ == "__main__":
    unittest.main()
