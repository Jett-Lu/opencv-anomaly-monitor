from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anomaly_monitor.names import normalize_person_name


class NormalizePersonNameTest(unittest.TestCase):
    def test_accepts_simple_demo_names(self) -> None:
        self.assertEqual(normalize_person_name(" Person A "), "Person A")

    def test_rejects_empty_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            normalize_person_name("   ")

    def test_rejects_path_segments(self) -> None:
        for name in ("../outside", "nested/person", "nested\\person"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "single folder"):
                    normalize_person_name(name)

    def test_rejects_windows_awkward_endings(self) -> None:
        with self.assertRaisesRegex(ValueError, "dot or space"):
            normalize_person_name("Person A.")


if __name__ == "__main__":
    unittest.main()
