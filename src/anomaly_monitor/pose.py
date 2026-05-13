from __future__ import annotations

import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from anomaly_monitor.config import Roi, clip_roi_to_frame


POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


@dataclass(frozen=True)
class PosePersonBehavior:
    index: int
    score: float
    labels: list[str]
    anchor: tuple[int, int]
    center: tuple[float, float]


@dataclass(frozen=True)
class PoseBehaviorResult:
    frame: np.ndarray
    score: float
    labels: list[str]
    person_detected: bool
    people: list[PosePersonBehavior]

    @property
    def is_anomaly(self) -> bool:
        return self.score > 0


class PoseBehaviorAnalyzer:
    """Pre-trained multi-person pose detection plus simple behavior scoring."""

    def __init__(
        self,
        pose_threshold: float,
        wrist_speed_threshold: float,
        roi: Roi | None,
        model_path: Path,
        max_poses: int,
    ) -> None:
        self.pose_threshold = pose_threshold
        self.wrist_speed_threshold = wrist_speed_threshold
        self.roi = roi
        self.model_path = self._ensure_model(model_path)
        self.previous_landmarks: list[dict[int, tuple[float, float]]] = []
        self.previous_time: float | None = None
        self.start_time = time.monotonic()
        self.last_timestamp_ms = -1

        self.connections = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=max_poses,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)

    def analyze(self, frame: np.ndarray) -> PoseBehaviorResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        results = self.landmarker.detect_for_video(image, self._next_timestamp_ms())

        if not results.pose_landmarks:
            self.previous_landmarks = []
            self.previous_time = None
            return PoseBehaviorResult(
                frame=frame,
                score=0.0,
                labels=[],
                person_detected=False,
                people=[],
            )

        annotated = frame.copy()
        frame_shape = frame.shape
        now = time.monotonic()
        current_landmarks: list[dict[int, tuple[float, float]]] = []
        people: list[PosePersonBehavior] = []
        all_labels: list[str] = []

        for person_index, pose_landmarks in enumerate(results.pose_landmarks):
            self._draw_landmarks(annotated, pose_landmarks, person_index)
            landmarks = self._visible_landmarks(pose_landmarks)
            previous = self._previous_for_person(person_index)
            score, labels = self._score_behavior(landmarks, frame_shape, previous, now)
            anchor = self._anchor(landmarks, frame_shape, person_index)
            center = self._center(landmarks)

            people.append(
                PosePersonBehavior(
                    index=person_index,
                    score=score,
                    labels=labels,
                    anchor=anchor,
                    center=center,
                )
            )
            current_landmarks.append(landmarks)
            for label in labels:
                if label not in all_labels:
                    all_labels.append(label)

        self.previous_landmarks = current_landmarks
        self.previous_time = now

        return PoseBehaviorResult(
            frame=annotated,
            score=max((person.score for person in people), default=0.0),
            labels=all_labels,
            person_detected=True,
            people=people,
        )

    def close(self) -> None:
        self.landmarker.close()

    def _ensure_model(self, model_path: Path) -> Path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if model_path.exists():
            return model_path

        print(f"Downloading MediaPipe pose model to {model_path}...")
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
        return model_path

    def _next_timestamp_ms(self) -> int:
        timestamp_ms = int((time.monotonic() - self.start_time) * 1000)
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        return timestamp_ms

    def _previous_for_person(self, person_index: int) -> dict[int, tuple[float, float]] | None:
        if person_index >= len(self.previous_landmarks):
            return None
        return self.previous_landmarks[person_index]

    def _visible_landmarks(self, landmarks) -> dict[int, tuple[float, float]]:
        visible: dict[int, tuple[float, float]] = {}
        for index, landmark in enumerate(landmarks):
            if landmark.visibility >= 0.55:
                visible[index] = (landmark.x, landmark.y)
        return visible

    def _draw_landmarks(self, frame: np.ndarray, landmarks, person_index: int) -> None:
        height, width = frame.shape[:2]
        points: dict[int, tuple[int, int]] = {}
        color = self._person_color(person_index)

        for index, landmark in enumerate(landmarks):
            if landmark.visibility < 0.55:
                continue
            points[index] = (int(landmark.x * width), int(landmark.y * height))

        for connection in self.connections:
            if connection.start not in points or connection.end not in points:
                continue
            cv2.line(frame, points[connection.start], points[connection.end], color, 2)

        for point in points.values():
            cv2.circle(frame, point, 4, (80, 220, 255), -1)

    def _person_color(self, person_index: int) -> tuple[int, int, int]:
        palette = [
            (255, 180, 0),
            (80, 220, 80),
            (255, 120, 180),
            (120, 180, 255),
            (220, 220, 80),
            (180, 120, 255),
        ]
        return palette[person_index % len(palette)]

    def _score_behavior(
        self,
        landmarks: dict[int, tuple[float, float]],
        frame_shape: tuple[int, ...],
        previous_landmarks: dict[int, tuple[float, float]] | None,
        now: float,
    ) -> tuple[float, list[str]]:
        labels: list[str] = []
        score = 0.0

        max_wrist_speed = self._max_wrist_speed(landmarks, previous_landmarks, now)
        if max_wrist_speed >= self.wrist_speed_threshold:
            labels.append("rapid_hand_motion")
            score += min(0.55, max_wrist_speed / (self.wrist_speed_threshold * 4))

        if self._hands_inside_roi(landmarks, frame_shape):
            labels.append("hands_near_roi")
            score += 0.3

        if self._arms_extended(landmarks):
            labels.append("arms_extended")
            score += 0.2

        if "rapid_hand_motion" in labels and (
            "hands_near_roi" in labels or "arms_extended" in labels
        ):
            labels.append("tamper_like_motion")
            score += 0.25

        return min(score, 1.0), labels

    def _max_wrist_speed(
        self,
        landmarks: dict[int, tuple[float, float]],
        previous_landmarks: dict[int, tuple[float, float]] | None,
        now: float,
    ) -> float:
        if previous_landmarks is None or self.previous_time is None:
            return 0.0

        elapsed = max(now - self.previous_time, 1e-6)
        wrist_indexes = (
            mp.tasks.vision.PoseLandmark.LEFT_WRIST.value,
            mp.tasks.vision.PoseLandmark.RIGHT_WRIST.value,
        )
        speeds: list[float] = []

        for index in wrist_indexes:
            if index not in landmarks or index not in previous_landmarks:
                continue

            x, y = landmarks[index]
            previous_x, previous_y = previous_landmarks[index]
            distance = float(np.hypot(x - previous_x, y - previous_y))
            speeds.append(distance / elapsed)

        return max(speeds, default=0.0)

    def _hands_inside_roi(
        self,
        landmarks: dict[int, tuple[float, float]],
        frame_shape: tuple[int, ...],
    ) -> bool:
        if self.roi is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        clipped_roi = clip_roi_to_frame(self.roi, frame_shape)
        if clipped_roi is None:
            return False

        roi_x, roi_y, roi_width, roi_height = clipped_roi
        wrist_indexes = (
            mp.tasks.vision.PoseLandmark.LEFT_WRIST.value,
            mp.tasks.vision.PoseLandmark.RIGHT_WRIST.value,
        )

        for index in wrist_indexes:
            if index not in landmarks:
                continue

            x, y = landmarks[index]
            pixel_x = int(x * frame_width)
            pixel_y = int(y * frame_height)
            if roi_x <= pixel_x < roi_x + roi_width and roi_y <= pixel_y < roi_y + roi_height:
                return True

        return False

    def _arms_extended(self, landmarks: dict[int, tuple[float, float]]) -> bool:
        pairs = (
            (
                mp.tasks.vision.PoseLandmark.LEFT_SHOULDER.value,
                mp.tasks.vision.PoseLandmark.LEFT_WRIST.value,
            ),
            (
                mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER.value,
                mp.tasks.vision.PoseLandmark.RIGHT_WRIST.value,
            ),
        )

        for shoulder_index, wrist_index in pairs:
            if shoulder_index not in landmarks or wrist_index not in landmarks:
                continue

            shoulder_x, shoulder_y = landmarks[shoulder_index]
            wrist_x, wrist_y = landmarks[wrist_index]
            horizontal_reach = abs(wrist_x - shoulder_x)
            vertical_reach = abs(wrist_y - shoulder_y)
            if horizontal_reach > 0.18 or vertical_reach > 0.22:
                return True

        return False

    def _anchor(
        self,
        landmarks: dict[int, tuple[float, float]],
        frame_shape: tuple[int, ...],
        person_index: int,
    ) -> tuple[int, int]:
        frame_height, frame_width = frame_shape[:2]
        preferred = (
            mp.tasks.vision.PoseLandmark.NOSE.value,
            mp.tasks.vision.PoseLandmark.LEFT_SHOULDER.value,
            mp.tasks.vision.PoseLandmark.RIGHT_SHOULDER.value,
        )
        points = [landmarks[index] for index in preferred if index in landmarks]
        if not points:
            center_x, center_y = self._center(landmarks)
            points = [(center_x, center_y)]

        x = int(sum(point[0] for point in points) / len(points) * frame_width)
        y = int(sum(point[1] for point in points) / len(points) * frame_height)
        return x, max(24 + person_index * 24, y - 16)

    def _center(self, landmarks: dict[int, tuple[float, float]]) -> tuple[float, float]:
        if not landmarks:
            return 0.5, 0.5

        x = sum(point[0] for point in landmarks.values()) / len(landmarks)
        y = sum(point[1] for point in landmarks.values()) / len(landmarks)
        return x, y
