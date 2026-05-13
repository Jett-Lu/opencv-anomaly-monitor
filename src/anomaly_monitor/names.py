from __future__ import annotations


INVALID_PERSON_NAME_CHARS = set('<>:"/\\|?*')


def normalize_person_name(name: str) -> str:
    normalized = name.strip()
    if not normalized:
        raise ValueError("person name must not be empty")

    if normalized in {".", ".."} or any(char in INVALID_PERSON_NAME_CHARS for char in normalized):
        raise ValueError("person name must be a single folder name")

    if normalized.endswith((".", " ")):
        raise ValueError("person name must not end with a dot or space")

    return normalized
