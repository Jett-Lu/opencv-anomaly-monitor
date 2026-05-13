from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import hypot


Point = tuple[float, float]


@dataclass(frozen=True)
class TrackObservation:
    index: int
    center: Point
    inside_roi: bool = False


@dataclass(frozen=True)
class TrackBehavior:
    index: int
    track_id: int
    score: float
    labels: list[str]
    age_seconds: float
    roi_dwell_seconds: float
    recent_speed: float
    path_length: float
    displacement: float


@dataclass
class _Track:
    track_id: int
    first_seen: float
    last_seen: float
    positions: deque[tuple[float, Point]] = field(default_factory=deque)
    roi_entered_at: float | None = None

    @property
    def center(self) -> Point:
        return self.positions[-1][1]


class PersonMotionTracker:
    """Tiny centroid tracker with short motion history for pose detections."""

    def __init__(
        self,
        match_distance: float,
        lost_seconds: float,
        history_seconds: float,
        loitering_seconds: float,
        loitering_radius: float,
        roi_dwell_seconds: float,
        repeated_motion_distance: float,
        repeated_motion_radius: float,
        rapid_body_speed_threshold: float,
    ) -> None:
        self.match_distance = match_distance
        self.lost_seconds = lost_seconds
        self.history_seconds = history_seconds
        self.loitering_seconds = loitering_seconds
        self.loitering_radius = loitering_radius
        self.roi_dwell_seconds = roi_dwell_seconds
        self.repeated_motion_distance = repeated_motion_distance
        self.repeated_motion_radius = repeated_motion_radius
        self.rapid_body_speed_threshold = rapid_body_speed_threshold
        self.next_track_id = 1
        self.tracks: dict[int, _Track] = {}

    def update(
        self,
        observations: list[TrackObservation],
        now: float,
    ) -> dict[int, TrackBehavior]:
        self._drop_lost_tracks(now)
        matches = self._match_observations(observations)
        behaviors: dict[int, TrackBehavior] = {}

        for observation in observations:
            track_id = matches.get(observation.index)
            if track_id is None:
                track_id = self._create_track(now)

            track = self.tracks[track_id]
            self._update_track(track, observation, now)
            behaviors[observation.index] = self._score_track(track, observation, now)

        return behaviors

    def _drop_lost_tracks(self, now: float) -> None:
        self.tracks = {
            track_id: track
            for track_id, track in self.tracks.items()
            if now - track.last_seen <= self.lost_seconds
        }

    def _match_observations(self, observations: list[TrackObservation]) -> dict[int, int]:
        pairs: list[tuple[float, int, int]] = []
        for observation in observations:
            for track_id, track in self.tracks.items():
                distance = point_distance(observation.center, track.center)
                if distance <= self.match_distance:
                    pairs.append((distance, observation.index, track_id))

        matches: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, observation_index, track_id in sorted(pairs):
            if observation_index in matches or track_id in used_tracks:
                continue
            matches[observation_index] = track_id
            used_tracks.add(track_id)

        return matches

    def _create_track(self, now: float) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = _Track(
            track_id=track_id,
            first_seen=now,
            last_seen=now,
        )
        return track_id

    def _update_track(
        self,
        track: _Track,
        observation: TrackObservation,
        now: float,
    ) -> None:
        track.last_seen = now
        track.positions.append((now, observation.center))
        while track.positions and now - track.positions[0][0] > self.history_seconds:
            track.positions.popleft()

        if observation.inside_roi:
            if track.roi_entered_at is None:
                track.roi_entered_at = now
        else:
            track.roi_entered_at = None

    def _score_track(
        self,
        track: _Track,
        observation: TrackObservation,
        now: float,
    ) -> TrackBehavior:
        age_seconds = now - track.first_seen
        roi_dwell_seconds = (
            now - track.roi_entered_at
            if track.roi_entered_at is not None
            else 0.0
        )
        recent_speed = self._recent_speed(track)
        path_length = self._path_length(track)
        displacement = self._displacement(track)
        labels: list[str] = []
        score = 0.0

        if observation.inside_roi:
            labels.append("restricted_zone_presence")
            score += 0.2

        if roi_dwell_seconds >= self.roi_dwell_seconds:
            labels.append("restricted_zone_dwell")
            score += 0.8

        if age_seconds >= self.loitering_seconds and displacement <= self.loitering_radius:
            labels.append("loitering")
            score += 0.75

        if (
            path_length >= self.repeated_motion_distance
            and displacement <= self.repeated_motion_radius
        ):
            labels.append("repeated_motion")
            score += 0.45

        if recent_speed >= self.rapid_body_speed_threshold:
            labels.append("rapid_body_motion")
            score += 0.35

        if (
            observation.inside_roi
            and recent_speed >= self.rapid_body_speed_threshold * 0.6
        ):
            labels.append("fast_motion_near_roi")
            score += 0.45

        if "restricted_zone_dwell" in labels and (
            "repeated_motion" in labels or "rapid_body_motion" in labels
        ):
            labels.append("suspicious_zone_behavior")
            score += 0.2

        return TrackBehavior(
            index=observation.index,
            track_id=track.track_id,
            score=min(score, 1.0),
            labels=dedupe(labels),
            age_seconds=age_seconds,
            roi_dwell_seconds=roi_dwell_seconds,
            recent_speed=recent_speed,
            path_length=path_length,
            displacement=displacement,
        )

    def _recent_speed(self, track: _Track) -> float:
        if len(track.positions) < 2:
            return 0.0

        previous_time, previous_center = track.positions[-2]
        current_time, current_center = track.positions[-1]
        elapsed = max(current_time - previous_time, 1e-6)
        return point_distance(previous_center, current_center) / elapsed

    def _path_length(self, track: _Track) -> float:
        if len(track.positions) < 2:
            return 0.0

        positions = list(track.positions)
        return sum(
            point_distance(previous[1], current[1])
            for previous, current in zip(positions, positions[1:])
        )

    def _displacement(self, track: _Track) -> float:
        if len(track.positions) < 2:
            return 0.0

        return point_distance(track.positions[0][1], track.positions[-1][1])


def dedupe(labels: list[str]) -> list[str]:
    merged: list[str] = []
    for label in labels:
        if label not in merged:
            merged.append(label)
    return merged


def point_distance(first: Point, second: Point) -> float:
    return hypot(first[0] - second[0], first[1] - second[1])
