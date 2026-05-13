from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.faces import UnknownEmbeddingMemory, UnknownFaceMemory, normalize_embedding


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


class UnknownEmbeddingMemoryTest(unittest.TestCase):
    def test_reuses_unknown_label_for_similar_embedding(self) -> None:
        memory = UnknownEmbeddingMemory(similarity_threshold=0.8)
        embedding = normalize_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))

        first_name, first_confidence = memory.identify(embedding)
        second_name, second_confidence = memory.identify(embedding.copy())

        self.assertEqual(first_name, "Unknown A")
        self.assertIsNone(first_confidence)
        self.assertEqual(second_name, "Unknown A")
        self.assertEqual(second_confidence, 100.0)

    def test_creates_new_unknown_label_for_different_embedding(self) -> None:
        memory = UnknownEmbeddingMemory(similarity_threshold=0.8)

        first_name, _ = memory.identify(
            normalize_embedding(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        )
        second_name, _ = memory.identify(
            normalize_embedding(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        )

        self.assertEqual(first_name, "Unknown A")
        self.assertEqual(second_name, "Unknown B")


if __name__ == "__main__":
    unittest.main()
