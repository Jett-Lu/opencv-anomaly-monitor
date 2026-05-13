from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from anomaly_monitor.config import (
    DEFAULT_ARCFACE_MODEL_PATH,
    DEFAULT_KNOWN_FACES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_MODEL_PATH,
    MonitorConfig,
)
from anomaly_monitor.detector import MotionAnomalyDetector
from anomaly_monitor.events import AlertEvent, EventLogger


@dataclass
class PendingAlertClip:
    frame_number: int
    score: float
    motion_score: float
    pose_score: float
    tracking_score: float
    moving_area: int
    region_count: int
    labels: list[str]
    person_detected: bool
    identity: str
    identities: list[str]
    people: list[str]
    snapshot_path: Path
    frames: list
    post_frames_remaining: int


def parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = value.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must use x,y,width,height format")

    try:
        x, y, width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers") from exc

    if x < 0 or y < 0 or width < 1 or height < 1:
        raise argparse.ArgumentTypeError("ROI must be non-negative with positive size")

    return x, y, width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCV camera anomaly detection proof of concept.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video path, or RTSP/HTTP camera URL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder where alert snapshots and events.jsonl are saved.",
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=DEFAULT_KNOWN_FACES_DIR,
        help="Folder of known faces, organized as known_faces/person_name/images.",
    )
    parser.add_argument(
        "--face-engine",
        choices=("arcface", "lbph"),
        default="arcface",
        help="Face recognizer to use. ArcFace uses pretrained ONNX embeddings.",
    )
    parser.add_argument(
        "--face-confidence-threshold",
        type=float,
        default=75.0,
        help="LBPH face recognition threshold. Lower is stricter.",
    )
    parser.add_argument(
        "--unknown-face-match-threshold",
        type=float,
        default=42.0,
        help="Session unknown-face match threshold. Lower is stricter.",
    )
    parser.add_argument(
        "--arcface-model",
        type=Path,
        default=DEFAULT_ARCFACE_MODEL_PATH,
        help="Path where the ArcFace ONNX model is stored or downloaded.",
    )
    parser.add_argument(
        "--arcface-similarity-threshold",
        type=float,
        default=0.34,
        help="ArcFace cosine similarity needed to label a known person. Higher is stricter.",
    )
    parser.add_argument(
        "--arcface-similarity-margin",
        type=float,
        default=0.03,
        help="Minimum gap between best and second-best ArcFace matches.",
    )
    parser.add_argument(
        "--identity-alert-hold-seconds",
        type=float,
        default=300.0,
        help="Seconds to remember that a flagged identity should stay flagged.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
        help="Motion anomaly score threshold from 0 to 1.",
    )
    parser.add_argument(
        "--pose-threshold",
        type=float,
        default=0.9,
        help="Pose behavior score threshold from 0 to 1.",
    )
    parser.add_argument(
        "--wrist-speed-threshold",
        type=float,
        default=3.0,
        help="Normalized wrist speed needed to flag rapid hand movement.",
    )
    parser.add_argument(
        "--loitering-seconds",
        type=float,
        default=30.0,
        help="Seconds a tracked person can stay in place before loitering is flagged.",
    )
    parser.add_argument(
        "--roi-dwell-seconds",
        type=float,
        default=8.0,
        help="Seconds a tracked person can remain inside the ROI before dwell is flagged.",
    )
    parser.add_argument(
        "--motion-history-seconds",
        type=float,
        default=20.0,
        help="Seconds of per-person movement history to retain.",
    )
    parser.add_argument(
        "--rapid-body-speed-threshold",
        type=float,
        default=1.5,
        help="Normalized full-body speed needed to flag rapid body movement.",
    )
    parser.add_argument(
        "--repeated-motion-distance",
        type=float,
        default=0.6,
        help="Recent normalized path length needed to flag repeated motion.",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=4,
        help="Maximum number of people/skeletons to detect at once.",
    )
    parser.add_argument(
        "--pose-model",
        type=Path,
        default=DEFAULT_POSE_MODEL_PATH,
        help="Path to the MediaPipe pose landmarker model file.",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Seconds between saved alert events.",
    )
    parser.add_argument(
        "--alert-hold-seconds",
        type=float,
        default=5.0,
        help="Seconds to keep a person marked red after an anomaly.",
    )
    parser.add_argument(
        "--event-video-seconds",
        type=float,
        default=None,
        help="Deprecated alias for --post-alert-seconds.",
    )
    parser.add_argument(
        "--pre-alert-seconds",
        type=float,
        default=2.0,
        help="Seconds before an alert to include in saved MP4 clips.",
    )
    parser.add_argument(
        "--post-alert-seconds",
        type=float,
        default=3.0,
        help="Seconds after an alert to include in saved MP4 clips.",
    )
    parser.add_argument(
        "--event-video-fps",
        type=float,
        default=12.0,
        help="FPS used when writing alert video clips.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to skip before saving alerts while the scene baseline warms up.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=750,
        help="Minimum contour area in pixels.",
    )
    parser.add_argument(
        "--roi",
        type=parse_roi,
        default=None,
        help="Optional restricted zone as x,y,width,height.",
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show the foreground motion mask window.",
    )
    parser.add_argument(
        "--show-motion-boxes",
        action="store_true",
        help="Draw motion boxes around moving regions.",
    )
    parser.add_argument(
        "--motion-alerts",
        action="store_true",
        help="Allow motion-only alerts. By default alerts focus on person pose behavior.",
    )
    parser.add_argument(
        "--full-behavior",
        action="store_true",
        help="Enable the rapid-hand, ROI, and extended-arm behavior rules.",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable human skeleton and behavior analysis.",
    )
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="Enable person tracking and motion history analysis.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable person tracking and motion history analysis.",
    )
    parser.add_argument(
        "--no-face-recognition",
        action="store_true",
        help="Disable known-face recognition.",
    )
    return parser.parse_args()


def normalize_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def save_alert_snapshot(output_dir: Path, frame_number: int, frame) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / f"alert_frame_{frame_number:06d}.jpg"
    if not cv2.imwrite(str(snapshot_path), frame):
        raise RuntimeError(f"Could not save alert snapshot: {snapshot_path}")
    return snapshot_path


def save_event_video(
    output_dir: Path,
    frame_number: int,
    frames,
    fps: float,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    buffered_frames = list(frames)
    if not buffered_frames:
        return None

    video_path = output_dir / f"alert_clip_{frame_number:06d}.mp4"
    height, width = buffered_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        return None

    for frame in buffered_frames:
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    if not video_path.exists() or video_path.stat().st_size == 0:
        return None

    return video_path


def finalize_pending_clip(
    output_dir: Path,
    logger: EventLogger,
    pending: PendingAlertClip,
    fps: float,
) -> None:
    video_path = save_event_video(
        output_dir,
        pending.frame_number,
        pending.frames,
        fps,
    )
    event = AlertEvent.create(
        frame_number=pending.frame_number,
        score=pending.score,
        motion_score=pending.motion_score,
        pose_score=pending.pose_score,
        tracking_score=pending.tracking_score,
        moving_area=pending.moving_area,
        region_count=pending.region_count,
        labels=pending.labels,
        person_detected=pending.person_detected,
        identity=pending.identity,
        identities=pending.identities,
        people=pending.people,
        snapshot_path=pending.snapshot_path,
        video_path=video_path,
    )
    logger.write(event)
    print(
        f"Alert saved | frame={event.frame_number} "
        f"person={event.identity} score={event.score} labels={event.labels} "
        f"snapshot={event.snapshot_path} video={event.video_path}"
    )


def run_monitor(config: MonitorConfig) -> None:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(normalize_source(config.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {config.source}")

    detector: MotionAnomalyDetector | None = None
    try:
        detector = MotionAnomalyDetector(config)
        logger = EventLogger(config.output_dir)
        last_alert_time = 0.0
        frame_number = 0
        video_buffer_size = max(1, int(config.pre_alert_seconds * config.event_video_fps))
        post_frame_count = max(0, int(config.post_alert_seconds * config.event_video_fps))
        recent_frames = deque(maxlen=video_buffer_size)
        pending_clips: list[PendingAlertClip] = []

        print("Camera anomaly monitor is running.")
        print("Press q to quit.")

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_number += 1
            result = detector.analyze(frame)
            recent_frames.append(result.frame.copy())
            now = time.monotonic()

            for pending in pending_clips[:]:
                pending.frames.append(result.frame.copy())
                pending.post_frames_remaining -= 1
                if pending.post_frames_remaining <= 0:
                    finalize_pending_clip(
                        config.output_dir,
                        logger,
                        pending,
                        config.event_video_fps,
                    )
                    pending_clips.remove(pending)

            warmed_up = frame_number > config.warmup_frames
            if (
                warmed_up
                and result.is_anomaly
                and now - last_alert_time >= config.cooldown_seconds
                and not pending_clips
            ):
                snapshot_path = save_alert_snapshot(config.output_dir, frame_number, result.frame)
                pending = PendingAlertClip(
                    frame_number,
                    score=result.score,
                    motion_score=result.motion_score,
                    pose_score=result.pose_score,
                    tracking_score=result.tracking_score,
                    moving_area=result.moving_area,
                    region_count=len(result.regions),
                    labels=result.labels,
                    person_detected=result.person_detected,
                    identity=result.identity,
                    identities=[identity.name for identity in result.identities],
                    people=result.people,
                    snapshot_path=snapshot_path,
                    frames=list(recent_frames),
                    post_frames_remaining=post_frame_count,
                )
                last_alert_time = now
                if pending.post_frames_remaining <= 0:
                    finalize_pending_clip(
                        config.output_dir,
                        logger,
                        pending,
                        config.event_video_fps,
                    )
                else:
                    pending_clips.append(pending)
                    print(
                        f"Alert triggered | frame={frame_number} "
                        f"capturing {config.post_alert_seconds:.1f}s post-roll..."
                    )

            cv2.imshow("Camera Anomaly Monitor", result.frame)
            if config.show_mask:
                cv2.imshow("Motion Mask", result.mask)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        for pending in pending_clips:
            finalize_pending_clip(
                config.output_dir,
                logger,
                pending,
                config.event_video_fps,
            )
    finally:
        capture.release()
        if detector is not None:
            detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    post_alert_seconds = (
        args.event_video_seconds
        if args.event_video_seconds is not None
        else args.post_alert_seconds
    )
    config = MonitorConfig(
        source=args.source,
        output_dir=args.output_dir,
        known_faces_dir=args.known_faces_dir,
        face_engine=args.face_engine,
        face_confidence_threshold=args.face_confidence_threshold,
        unknown_face_match_threshold=args.unknown_face_match_threshold,
        arcface_model_path=args.arcface_model,
        arcface_similarity_threshold=args.arcface_similarity_threshold,
        arcface_similarity_margin=args.arcface_similarity_margin,
        identity_alert_hold_seconds=args.identity_alert_hold_seconds,
        threshold=args.threshold,
        pose_threshold=args.pose_threshold,
        wrist_speed_threshold=args.wrist_speed_threshold,
        loitering_seconds=args.loitering_seconds,
        roi_dwell_seconds=args.roi_dwell_seconds,
        motion_history_seconds=args.motion_history_seconds,
        rapid_body_speed_threshold=args.rapid_body_speed_threshold,
        repeated_motion_distance=args.repeated_motion_distance,
        max_poses=args.max_poses,
        pose_model_path=args.pose_model,
        cooldown_seconds=args.cooldown,
        alert_hold_seconds=args.alert_hold_seconds,
        event_video_seconds=args.pre_alert_seconds + post_alert_seconds,
        pre_alert_seconds=args.pre_alert_seconds,
        post_alert_seconds=post_alert_seconds,
        event_video_fps=args.event_video_fps,
        warmup_frames=args.warmup_frames,
        min_area=args.min_area,
        roi=args.roi,
        t_pose_only=not args.full_behavior,
        enable_pose=not args.no_pose,
        enable_tracking=args.tracking and not args.no_tracking,
        enable_face_recognition=not args.no_face_recognition,
        enable_motion_alerts=args.motion_alerts,
        show_motion_boxes=args.show_motion_boxes,
        show_mask=args.show_mask,
    )
    run_monitor(config)


if __name__ == "__main__":
    main()
