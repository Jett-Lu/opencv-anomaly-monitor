from dataclasses import dataclass
from pathlib import Path


Roi = tuple[int, int, int, int]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "alerts"
DEFAULT_POSE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "pose_landmarker_lite.task"
DEFAULT_ARCFACE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "arcface.onnx"
DEFAULT_KNOWN_FACES_DIR = PROJECT_ROOT / "data" / "known_faces"


def clip_roi_to_frame(roi: Roi, frame_shape: tuple[int, ...]) -> Roi | None:
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = roi
    left = min(max(x, 0), frame_width)
    top = min(max(y, 0), frame_height)
    right = min(max(x + width, 0), frame_width)
    bottom = min(max(y + height, 0), frame_height)

    clipped_width = right - left
    clipped_height = bottom - top
    if clipped_width <= 0 or clipped_height <= 0:
        return None

    return left, top, clipped_width, clipped_height


def roi_area(roi: Roi, frame_shape: tuple[int, ...]) -> int:
    clipped = clip_roi_to_frame(roi, frame_shape)
    if clipped is None:
        return 0

    _, _, width, height = clipped
    return width * height


@dataclass(frozen=True)
class MonitorConfig:
    source: str
    output_dir: Path
    roi: Roi | None = None
    threshold: float = 0.08
    pose_threshold: float = 0.9
    wrist_speed_threshold: float = 3.0
    max_poses: int = 4
    pose_model_path: Path = DEFAULT_POSE_MODEL_PATH
    known_faces_dir: Path = DEFAULT_KNOWN_FACES_DIR
    face_engine: str = "arcface"
    face_confidence_threshold: float = 75.0
    unknown_face_match_threshold: float = 42.0
    arcface_model_path: Path = DEFAULT_ARCFACE_MODEL_PATH
    arcface_similarity_threshold: float = 0.45
    arcface_similarity_margin: float = 0.03
    identity_alert_hold_seconds: float = 300.0
    cooldown_seconds: float = 5.0
    alert_hold_seconds: float = 5.0
    event_video_seconds: float = 5.0
    pre_alert_seconds: float = 2.0
    post_alert_seconds: float = 3.0
    event_video_fps: float = 12.0
    warmup_frames: int = 30
    min_area: int = 750
    blur_size: int = 7
    history: int = 500
    var_threshold: float = 32.0
    learning_rate: float = -1.0
    t_pose_only: bool = True
    enable_tracking: bool = False
    track_match_distance: float = 0.16
    track_lost_seconds: float = 2.0
    motion_history_seconds: float = 20.0
    loitering_seconds: float = 30.0
    loitering_radius: float = 0.08
    roi_dwell_seconds: float = 8.0
    repeated_motion_distance: float = 0.6
    repeated_motion_radius: float = 0.12
    rapid_body_speed_threshold: float = 1.5
    enable_pose: bool = True
    enable_face_recognition: bool = True
    enable_motion_alerts: bool = False
    show_motion_boxes: bool = False
    show_mask: bool = False

    def validate(self) -> None:
        if self.threshold <= 0 or self.threshold >= 1:
            raise ValueError("threshold must be between 0 and 1")
        if self.pose_threshold <= 0 or self.pose_threshold > 1:
            raise ValueError("pose_threshold must be between 0 and 1")
        if self.wrist_speed_threshold <= 0:
            raise ValueError("wrist_speed_threshold must be greater than zero")
        if self.max_poses < 1:
            raise ValueError("max_poses must be at least 1")
        if self.face_confidence_threshold <= 0:
            raise ValueError("face_confidence_threshold must be greater than zero")
        if self.face_engine not in {"arcface", "lbph"}:
            raise ValueError("face_engine must be arcface or lbph")
        if self.unknown_face_match_threshold <= 0:
            raise ValueError("unknown_face_match_threshold must be greater than zero")
        if not 0 < self.arcface_similarity_threshold <= 1:
            raise ValueError("arcface_similarity_threshold must be between 0 and 1")
        if self.arcface_similarity_margin < 0:
            raise ValueError("arcface_similarity_margin must be zero or greater")
        if self.identity_alert_hold_seconds < 0:
            raise ValueError("identity_alert_hold_seconds must be zero or greater")
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be zero or greater")
        if self.alert_hold_seconds < 0:
            raise ValueError("alert_hold_seconds must be zero or greater")
        if self.event_video_seconds <= 0:
            raise ValueError("event_video_seconds must be greater than zero")
        if self.pre_alert_seconds < 0:
            raise ValueError("pre_alert_seconds must be zero or greater")
        if self.post_alert_seconds < 0:
            raise ValueError("post_alert_seconds must be zero or greater")
        if self.pre_alert_seconds + self.post_alert_seconds <= 0:
            raise ValueError("alert clip seconds must be greater than zero")
        if self.event_video_fps <= 0:
            raise ValueError("event_video_fps must be greater than zero")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be zero or greater")
        if self.min_area < 1:
            raise ValueError("min_area must be greater than zero")
        if self.blur_size < 1 or self.blur_size % 2 == 0:
            raise ValueError("blur_size must be a positive odd number")
        if self.history < 1:
            raise ValueError("history must be at least 1")
        if self.var_threshold <= 0:
            raise ValueError("var_threshold must be greater than zero")
        if self.learning_rate != -1 and not 0 <= self.learning_rate <= 1:
            raise ValueError("learning_rate must be -1 or between 0 and 1")
        if self.track_match_distance <= 0:
            raise ValueError("track_match_distance must be greater than zero")
        if self.track_lost_seconds <= 0:
            raise ValueError("track_lost_seconds must be greater than zero")
        if self.motion_history_seconds <= 0:
            raise ValueError("motion_history_seconds must be greater than zero")
        if self.loitering_seconds <= 0:
            raise ValueError("loitering_seconds must be greater than zero")
        if self.loitering_radius <= 0:
            raise ValueError("loitering_radius must be greater than zero")
        if self.roi_dwell_seconds <= 0:
            raise ValueError("roi_dwell_seconds must be greater than zero")
        if self.repeated_motion_distance <= 0:
            raise ValueError("repeated_motion_distance must be greater than zero")
        if self.repeated_motion_radius <= 0:
            raise ValueError("repeated_motion_radius must be greater than zero")
        if self.rapid_body_speed_threshold <= 0:
            raise ValueError("rapid_body_speed_threshold must be greater than zero")
        if self.roi is not None:
            _, _, width, height = self.roi
            if width < 1 or height < 1:
                raise ValueError("roi width and height must be greater than zero")
