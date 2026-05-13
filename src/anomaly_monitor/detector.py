from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from anomaly_monitor.config import MonitorConfig, clip_roi_to_frame, roi_area
from anomaly_monitor.faces import FaceIdentity, FaceRecognizer
from anomaly_monitor.pose import PoseBehaviorAnalyzer, PosePersonBehavior
from anomaly_monitor.tracking import PersonMotionTracker, TrackBehavior, TrackObservation


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
    tracking_score: float
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
        self.identity_alerts: dict[str, ActivePersonAlert] = {}
        self.background_model = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=True,
        )
        self.pose_analyzer = (
            PoseBehaviorAnalyzer(
                pose_threshold=config.pose_threshold,
                wrist_speed_threshold=config.wrist_speed_threshold,
                t_pose_only=config.t_pose_only,
                roi=config.roi,
                model_path=config.pose_model_path,
                max_poses=config.max_poses,
            )
            if config.enable_pose
            else None
        )
        self.motion_tracker = (
            PersonMotionTracker(
                match_distance=config.track_match_distance,
                lost_seconds=config.track_lost_seconds,
                history_seconds=config.motion_history_seconds,
                loitering_seconds=config.loitering_seconds,
                loitering_radius=config.loitering_radius,
                roi_dwell_seconds=config.roi_dwell_seconds,
                repeated_motion_distance=config.repeated_motion_distance,
                repeated_motion_radius=config.repeated_motion_radius,
                rapid_body_speed_threshold=config.rapid_body_speed_threshold,
            )
            if config.enable_pose and config.enable_tracking
            else None
        )
        self.face_recognizer = (
            FaceRecognizer(
                known_faces_dir=config.known_faces_dir,
                confidence_threshold=config.face_confidence_threshold,
                unknown_match_threshold=config.unknown_face_match_threshold,
                engine=config.face_engine,
                arcface_model_path=config.arcface_model_path,
                arcface_similarity_threshold=config.arcface_similarity_threshold,
                arcface_similarity_margin=config.arcface_similarity_margin,
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
        tracking_behaviors = self._analyze_motion_history(pose_people, frame.shape)
        tracking_score = max(
            (behavior.score for behavior in tracking_behaviors.values()),
            default=0.0,
        )
        tracking_labels = self._tracking_labels(tracking_behaviors)
        active_labels = self._update_person_alerts(
            pose_people,
            tracking_behaviors,
            matched_names,
        )
        display_labels = self._merge_labels(labels + tracking_labels, active_labels)
        active_pose_score = self._active_alert_score(
            pose_people,
            tracking_behaviors,
            matched_names,
        )
        score = max(alert_motion_score, pose_score, tracking_score, active_pose_score)
        person_summaries = self._person_summaries(
            pose_people,
            matched_names,
            active_labels,
            tracking_behaviors,
        )
        identity = self._primary_identity(
            pose_people,
            identities,
            matched_names,
            active_labels,
            tracking_behaviors,
        )
        display_frame = self._draw_regions(
            pose_frame.copy(),
            regions,
            motion_score,
            pose_score,
            tracking_score,
            display_labels,
            person_detected,
            identity,
            pose_people,
            matched_names,
            active_labels,
            tracking_behaviors,
        )

        return DetectionResult(
            frame=display_frame,
            mask=mask,
            score=score,
            motion_score=motion_score,
            pose_score=pose_score,
            tracking_score=tracking_score,
            moving_area=moving_area,
            regions=regions,
            labels=display_labels,
            person_detected=person_detected,
            identity=identity,
            identities=identities,
            people=person_summaries,
            is_anomaly=(
                (self.config.enable_motion_alerts and motion_score >= self.config.threshold)
                or pose_score >= self.config.pose_threshold
                or tracking_score >= self.config.pose_threshold
                or active_pose_score >= self.config.pose_threshold
                or bool(active_labels)
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

        clipped_roi = clip_roi_to_frame(self.config.roi, mask.shape)
        if clipped_roi is None:
            return np.zeros_like(mask)

        x, y, width, height = clipped_roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y : y + height, x : x + width] = 255
        return cv2.bitwise_and(mask, roi_mask)

    def _analysis_area(self, frame: np.ndarray) -> int:
        if self.config.roi is None:
            return frame.shape[0] * frame.shape[1]

        return roi_area(self.config.roi, frame.shape)

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

    def _analyze_motion_history(
        self,
        people: list[PosePersonBehavior],
        frame_shape: tuple[int, ...],
    ) -> dict[int, TrackBehavior]:
        if self.motion_tracker is None or not people:
            return {}

        observations = [
            TrackObservation(
                index=person.index,
                center=person.center,
                inside_roi=self._center_inside_roi(person.center, frame_shape),
            )
            for person in people
        ]
        return self.motion_tracker.update(observations, time.monotonic())

    def _center_inside_roi(
        self,
        center: tuple[float, float],
        frame_shape: tuple[int, ...],
    ) -> bool:
        if self.config.roi is None:
            return False

        clipped_roi = clip_roi_to_frame(self.config.roi, frame_shape)
        if clipped_roi is None:
            return False

        frame_height, frame_width = frame_shape[:2]
        x, y, width, height = clipped_roi
        center_x = int(center[0] * frame_width)
        center_y = int(center[1] * frame_height)
        return x <= center_x < x + width and y <= center_y < y + height

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
        tracking_behaviors: dict[int, TrackBehavior],
    ) -> list[str]:
        summaries: list[str] = []
        for person in people:
            name = matched_names.get(person.index, f"Person {person.index + 1}")
            tracking = tracking_behaviors.get(person.index)
            tracking_labels = tracking.labels if tracking is not None else []
            labels_for_person = active_labels.get(
                person.index,
                self._dedupe(person.labels + tracking_labels),
            )
            score = max(person.score, tracking.score if tracking is not None else 0.0)
            track_text = f" track={tracking.track_id}" if tracking is not None else ""
            if labels_for_person:
                labels = ", ".join(labels_for_person)
                summaries.append(f"{name}{track_text}: {labels} score={score:.2f}")
            else:
                summaries.append(f"{name}{track_text}: normal score={score:.2f}")
        return summaries

    def _primary_identity(
        self,
        people: list[PosePersonBehavior],
        identities: list[FaceIdentity],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
        tracking_behaviors: dict[int, TrackBehavior],
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
                    self._alert_key(item, tracking_behaviors.get(item.index)),
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
        tracking_behaviors: dict[int, TrackBehavior],
        matched_names: dict[int, str],
    ) -> dict[int, list[str]]:
        now = time.monotonic()
        self.person_alerts = {
            index: alert
            for index, alert in self.person_alerts.items()
            if alert.expires_at > now
        }
        self.identity_alerts = {
            name: alert
            for name, alert in self.identity_alerts.items()
            if alert.expires_at > now
        }

        for person in people:
            tracking = tracking_behaviors.get(person.index)
            name = matched_names.get(person.index)
            labels = self._dedupe(
                person.labels + (tracking.labels if tracking is not None else [])
            )
            score = max(person.score, tracking.score if tracking is not None else 0.0)
            if score >= self.config.pose_threshold and labels:
                self.person_alerts[self._alert_key(person, tracking)] = ActivePersonAlert(
                    labels=labels,
                    score=score,
                    expires_at=now + self.config.alert_hold_seconds,
                )
                if name is not None and self._can_remember_identity(name):
                    self.identity_alerts[name] = ActivePersonAlert(
                        labels=labels,
                        score=score,
                        expires_at=now + self.config.identity_alert_hold_seconds,
                    )

        active_labels: dict[int, list[str]] = {}
        for person in people:
            tracking = tracking_behaviors.get(person.index)
            alerts = []
            alert = self.person_alerts.get(self._alert_key(person, tracking))
            if alert is not None and alert.expires_at > now:
                alerts.append(alert)

            name = matched_names.get(person.index)
            identity_alert = self.identity_alerts.get(name) if name is not None else None
            if identity_alert is not None and identity_alert.expires_at > now:
                alerts.append(identity_alert)

            if alerts:
                labels: list[str] = []
                for item in alerts:
                    labels.extend(item.labels)
                active_labels[person.index] = self._dedupe(labels)

        return active_labels

    def _can_remember_identity(self, name: str) -> bool:
        if not name or name == "no face":
            return False
        return not (name.startswith("Person ") and name.removeprefix("Person ").isdigit())

    def _active_alert_score(
        self,
        people: list[PosePersonBehavior],
        tracking_behaviors: dict[int, TrackBehavior],
        matched_names: dict[int, str],
    ) -> float:
        now = time.monotonic()
        scores: list[float] = []
        for person in people:
            tracking = tracking_behaviors.get(person.index)
            alert = self.person_alerts.get(self._alert_key(person, tracking))
            if alert is not None and alert.expires_at > now:
                scores.append(alert.score)

            name = matched_names.get(person.index)
            identity_alert = self.identity_alerts.get(name) if name is not None else None
            if identity_alert is not None and identity_alert.expires_at > now:
                scores.append(identity_alert.score)

        return max(scores, default=0.0)

    def _alert_key(
        self,
        person: PosePersonBehavior,
        tracking: TrackBehavior | None,
    ) -> int:
        if tracking is not None:
            return tracking.track_id
        return person.index

    def _merge_labels(
        self,
        labels: list[str],
        active_labels: dict[int, list[str]],
    ) -> list[str]:
        merged = self._dedupe(labels)
        for person_labels in active_labels.values():
            for label in person_labels:
                if label not in merged:
                    merged.append(label)

        return merged

    def _tracking_labels(
        self,
        tracking_behaviors: dict[int, TrackBehavior],
    ) -> list[str]:
        labels: list[str] = []
        for behavior in tracking_behaviors.values():
            labels.extend(behavior.labels)
        return self._dedupe(labels)

    def _dedupe(self, labels: list[str]) -> list[str]:
        merged: list[str] = []
        for label in labels:
            if label not in merged:
                merged.append(label)
        return merged

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
        tracking_score: float,
        labels: list[str],
        person_detected: bool,
        identity: str,
        pose_people: list[PosePersonBehavior],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
        tracking_behaviors: dict[int, TrackBehavior],
    ) -> np.ndarray:
        if self.config.roi is not None:
            clipped_roi = clip_roi_to_frame(self.config.roi, frame.shape)
            if clipped_roi is not None:
                x, y, width, height = clipped_roi
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
        ) or (
            max(pose_score, tracking_score) >= self.config.pose_threshold
        ) or bool(active_labels)
        status = "ANOMALY" if is_anomaly else "normal"
        color = (0, 0, 255) if is_anomaly else (80, 220, 80)
        cv2.putText(
            frame,
            (
                f"Status: {status} | Motion: {motion_score:.3f} "
                f"| Pose: {pose_score:.3f} | Track: {tracking_score:.3f}"
            ),
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
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

        self._draw_person_labels(
            frame,
            pose_people,
            matched_names,
            active_labels,
            tracking_behaviors,
        )
        return frame

    def _draw_person_labels(
        self,
        frame: np.ndarray,
        people: list[PosePersonBehavior],
        matched_names: dict[int, str],
        active_labels: dict[int, list[str]],
        tracking_behaviors: dict[int, TrackBehavior],
    ) -> None:
        for person in people:
            name = matched_names.get(person.index, f"Person {person.index + 1}")
            tracking = tracking_behaviors.get(person.index)
            tracking_labels = tracking.labels if tracking is not None else []
            labels_for_person = active_labels.get(
                person.index,
                self._dedupe(person.labels + tracking_labels),
            )
            track_name = f"T{tracking.track_id} {name}" if tracking is not None else name
            if labels_for_person:
                label = f"ALERT {track_name}: {', '.join(labels_for_person[:2])}"
                self._draw_alert_box(frame, person)
            else:
                label = track_name

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

    def _draw_alert_box(
        self,
        frame: np.ndarray,
        person: PosePersonBehavior,
    ) -> None:
        x, y, width, height = person.box
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            (0, 0, 255),
            3,
        )
