from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from anomaly_monitor.config import MonitorConfig
from anomaly_monitor.faces import FaceIdentity, FaceRecognizer
from anomaly_monitor.pose import PoseBehaviorAnalyzer, PosePersonBehavior


@dataclass(frozen=True)
class MotionRegion:
    x: int
    y: int
    width: int
    height: int
    area: int


@dataclass(frozen=True)
class DetectionResult:
    frame: np.ndarray
    mask: np.ndarray
    score: float
    motion_score: float
    pose_score: float
    moving_area: int
    regions: list[MotionRegion]
    labels: list[str]
    person_detected: bool
    identity: str
    identities: list[FaceIdentity]
    people: list[str]
    is_anomaly: bool


@dataclass(frozen=True)
class ActivePersonAlert:
    labels: list[str]
    score: float
    expires_at: float


class MotionAnomalyDetector:
    def __init__(self, config: MonitorConfig) -> None:
        self.config = config
        self.person_alerts: dict[int, ActivePersonAlert] = {}
        self.background_model = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=True,
        )
        self.pose_analyzer = (
            PoseBehaviorAnalyzer(
                pose_threshold=config.pose_threshold,
                wrist_speed_threshold=config.wrist_speed_threshold,
                roi=config.roi,
                model_path=config.pose_model_path,
                max_poses=config.max_poses,
            )
            if config.enable_pose
            else None
        )
        self.face_recognizer = (
            FaceRecognizer(
                known_faces_dir=config.known_faces_dir,
                confidence_threshold=config.face_confidence_threshold,
            )
            if config.enable_face_recognition
            else None
        )

    def analyze(self, frame: np.ndarray) -> DetectionResult:
        pose_score = 0.0
        labels: list[str] = []
        person_detected = False
        pose_frame = frame.copy()
        identities: list[FaceIdentity] = []
        pose_people: list[PosePersonBehavior] = []

        if self.pose_analyzer is not None:
            pose_result = self.pose_analyzer.analyze(frame)
            pose_frame = pose_result.frame
            pose_score = pose_result.score
            labels = pose_result.labels
            person_detected = pose_result.person_detected
            pose_people = pose_result.people

        if self.face_recognizer is not None:
            identities = self.face_recognizer.recognize(frame)
            self.face_recognizer.draw(pose_frame, identities)
            person_detected = person_detected or bool(identities)

        processed = self._preprocess(frame)
        mask = self.background_model.apply(
            processed,
            learningRate=self.config.learning_rate,
        )
        mask = self._clean_mask(mask)
        mask = self._apply_roi(mask)
        regions = list(self._find_regions(mask))
        moving_area = sum(region.area for region in regions)
        total_area = self._analysis_area(frame)
        motion_score = moving_area / total_area if total_area else 0.0
        alert_motion_score = motion_score if self.config.enable_motion_alerts else 0.0
        matched_names = self._match_identities_to_people(pose_people, identities)
        active_labels = self._update_person_alerts(pose_people)
        active_pose_score = max(
            (
                alert.score
                for alert in self.person_alerts.values()
                if alert.expires_at > time.monotonic()
            ),
            default=0.0,
        )
        score = max(alert_motion_score, pose_score, active_pose_score)
        person_summaries = self._person_summaries(pose_people, matched_names, active_labels)
        identity = self._primary_identity(pose_people, identities, matched_names, active_labels)
        display_frame = self._draw_regions(
            pose_frame.copy(),
            regions,
            motion_score,
            pose_score,
            labels,
            person_detected,
            identity,
            pose_people,
            matched_names,
            active_labels,
        )

        return DetectionResult(
            frame=display_frame,
            mask=mask,
            score=score,
            motion_score=motion_score,
            pose_score=pose_score,
            moving_area=moving_area,
            regions=regions,
            labels=labels,
            person_detected=person_detected,
            identity=identity,
            identities=identities,
            people=person_summaries,
            is_anomaly=(
                (self.config.enable_motion_alerts and motion_score >= self.config.threshold)
                or pose_score >= self.config.pose_threshold
                or active_pose_score >= self.config.pose_threshold
            ),
        )

    def close(self) -> None:
        if self.pose_analyzer is not None:
            self.pose_analyzer.close()

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            frame,
            (self.config.blur_size, self.config.blur_size),
            0,
        )

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        _, thresholded = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        return cv2.dilate(opened, kernel, iterations=2)

    def _apply_roi(self, mask: np.ndarray) -> np.ndarray:
        if self.config.roi is None:
            return mask

        x, y, width, height = self.config.roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y : y + height, x : x + width] = 255
        return cv2.bitwise_and(mask, roi_mask)

    def _analysis_area(self, frame: np.ndarray) -> int:
        if self.config.roi is None:
            return frame.shape[0] * frame.shape[1]

        _, _, width, height = self.config.roi
        return width * height

    def _find_regions(self, mask: np.ndarray) -> Iterable[MotionRegion]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self.config.min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            yield MotionRegion(
                x=x,
                y=y,
                width=width,
                height=height,
                area=area,
            )

    def _match_identities_to_people(
        self,
        people: list[PosePersonBehavior],
        identities: list[FaceIdentity],
    ) -> dict[int, str]:
        matched_names: dict[int, str] = {}
        available = identities.copy()

        for person in people:
            if not available:
                matched_names[person.index] = f"Person {person.index + 1}"
                continue

            best_identity = min(
                available,
                key=lambda identity: self._point_distance(person.anchor, identity.center),
            )
            if self._point_distance(person.anchor, best_identity.center) <= 240:
                matched_names[person.index] = best_identity.name
                available.remove(best_identity)
            else:
                matched_names[person.index] = f"Person {person.index + 1}"

        return matched_names

    def _person_summaries(
        self,
        people: list[PosePersonBehavior],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
    ) -> list[str]:
        summaries: list[str] = []
        for person in people:
            name = matched_names.get(person.index, f"Person {person.index + 1}")
            labels_for_person = active_labels.get(person.index, person.labels)
            if labels_for_person:
                labels = ", ".join(labels_for_person)
                summaries.append(f"{name}: {labels} score={person.score:.2f}")
            else:
                summaries.append(f"{name}: normal score={person.score:.2f}")
        return summaries

    def _primary_identity(
        self,
        people: list[PosePersonBehavior],
        identities: list[FaceIdentity],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
    ) -> str:
        anomalous_people = [
            person
            for person in people
            if person.labels or active_labels.get(person.index)
        ]
        if anomalous_people:
            person = max(
                anomalous_people,
                key=lambda item: self.person_alerts.get(
                    item.index,
                    ActivePersonAlert(labels=item.labels, score=item.score, expires_at=0.0),
                ).score,
            )
            return matched_names.get(person.index, f"Person {person.index + 1}")

        if len(people) > 1:
            return f"{len(people)} people"

        if len(people) == 1:
            return matched_names.get(people[0].index, "Person 1")

        known_identities = [identity for identity in identities if identity.name != "unknown person"]
        if known_identities:
            return min(
                known_identities,
                key=lambda identity: identity.confidence
                if identity.confidence is not None
                else float("inf"),
            ).name
        if identities:
            return "unknown person"
        return "no face"

    def _update_person_alerts(
        self,
        people: list[PosePersonBehavior],
    ) -> dict[int, list[str]]:
        now = time.monotonic()
        self.person_alerts = {
            index: alert
            for index, alert in self.person_alerts.items()
            if alert.expires_at > now
        }

        for person in people:
            if person.score >= self.config.pose_threshold and person.labels:
                self.person_alerts[person.index] = ActivePersonAlert(
                    labels=person.labels,
                    score=person.score,
                    expires_at=now + self.config.alert_hold_seconds,
                )

        return {
            index: alert.labels
            for index, alert in self.person_alerts.items()
            if alert.expires_at > now
        }

    def _point_distance(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> float:
        return float(np.hypot(first[0] - second[0], first[1] - second[1]))

    def _draw_regions(
        self,
        frame: np.ndarray,
        regions: list[MotionRegion],
        motion_score: float,
        pose_score: float,
        labels: list[str],
        person_detected: bool,
        identity: str,
        pose_people: list[PosePersonBehavior],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
    ) -> np.ndarray:
        if self.config.roi is not None:
            x, y, width, height = self.config.roi
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 180, 0), 2)
            cv2.putText(
                frame,
                "ROI",
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 180, 0),
                2,
                cv2.LINE_AA,
            )

        if self.config.show_motion_boxes:
            for region in regions:
                cv2.rectangle(
                    frame,
                    (region.x, region.y),
                    (region.x + region.width, region.y + region.height),
                    (0, 200, 255),
                    2,
                )

        is_anomaly = (
            self.config.enable_motion_alerts and motion_score >= self.config.threshold
        ) or pose_score >= self.config.pose_threshold or bool(active_labels)
        status = "ANOMALY" if is_anomaly else "normal"
        color = (0, 0, 255) if is_anomaly else (80, 220, 80)
        cv2.putText(
            frame,
            f"Status: {status} | Motion: {motion_score:.3f} | Pose: {pose_score:.3f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            color,
            2,
            cv2.LINE_AA,
        )
        pose_status = "person detected" if person_detected else "no person"
        cv2.putText(
            frame,
            f"Person: {identity} | Pose: {pose_status}",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        if labels:
            cv2.putText(
                frame,
                f"Behavior: {', '.join(labels[:3])}",
                (16, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

        self._draw_person_labels(frame, pose_people, matched_names, active_labels)
        return frame

    def _draw_person_labels(
        self,
        frame: np.ndarray,
        people: list[PosePersonBehavior],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
    ) -> None:
        for person in people:
            name = matched_names.get(person.index, f"Person {person.index + 1}")
            labels_for_person = active_labels.get(person.index, person.labels)
            if labels_for_person:
                label = f"ALERT {name}: {', '.join(labels_for_person[:2])}"
            else:
                label = name

            color = (0, 0, 255) if labels_for_person else (230, 230, 230)
            cv2.putText(
                frame,
                label,
                person.anchor,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
