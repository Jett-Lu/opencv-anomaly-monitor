from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AlertEvent:
    timestamp: str
    frame_number: int
    score: float
    motion_score: float
    pose_score: float
    moving_area: int
    region_count: int
    labels: list[str]
    person_detected: bool
    identity: str
    identities: list[str]
    people: list[str]
    snapshot_path: str
    video_path: str | None

    @classmethod
    def create(
        cls,
        frame_number: int,
        score: float,
        motion_score: float,
        pose_score: float,
        moving_area: int,
        region_count: int,
        labels: list[str],
        person_detected: bool,
        identity: str,
        identities: list[str],
        people: list[str],
        snapshot_path: Path,
        video_path: Path | None,
    ) -> "AlertEvent":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            frame_number=frame_number,
            score=round(score, 5),
            motion_score=round(motion_score, 5),
            pose_score=round(pose_score, 5),
            moving_area=moving_area,
            region_count=region_count,
            labels=labels,
            person_detected=person_detected,
            identity=identity,
            identities=identities,
            people=people,
            snapshot_path=str(snapshot_path),
            video_path=str(video_path) if video_path is not None else None,
        )


class EventLogger:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "events.jsonl"

    def write(self, event: AlertEvent) -> None:
        payload: dict[str, Any] = asdict(event)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload) + "\n")
