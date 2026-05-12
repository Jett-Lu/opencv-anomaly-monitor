from dataclasses import dataclass
from pathlib import Path


Roi = tuple[int, int, int, int]
DEFAULT_POSE_MODEL_PATH = Path("data/models/pose_landmarker_lite.task")
DEFAULT_KNOWN_FACES_DIR = Path("data/known_faces")


@dataclass(frozen=True)
class MonitorConfig:
    source: str
    output_dir: Path
    roi: Roi | None = None
    threshold: float = 0.08
    pose_threshold: float = 0.75
    wrist_speed_threshold: float = 1.4
    max_poses: int = 4
    pose_model_path: Path = DEFAULT_POSE_MODEL_PATH
    known_faces_dir: Path = DEFAULT_KNOWN_FACES_DIR
    face_confidence_threshold: float = 75.0
    cooldown_seconds: float = 5.0
    alert_hold_seconds: float = 5.0
    event_video_seconds: float = 4.0
    event_video_fps: float = 12.0
    warmup_frames: int = 30
    min_area: int = 750
    blur_size: int = 7
    history: int = 500
    var_threshold: float = 32.0
    learning_rate: float = -1.0
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
        if self.cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be zero or greater")
        if self.alert_hold_seconds < 0:
            raise ValueError("alert_hold_seconds must be zero or greater")
        if self.event_video_seconds <= 0:
            raise ValueError("event_video_seconds must be greater than zero")
        if self.event_video_fps <= 0:
            raise ValueError("event_video_fps must be greater than zero")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be zero or greater")
        if self.min_area < 1:
            raise ValueError("min_area must be greater than zero")
        if self.blur_size < 1 or self.blur_size % 2 == 0:
            raise ValueError("blur_size must be a positive odd number")
        if self.roi is not None:
            _, _, width, height = self.roi
            if width < 1 or height < 1:
                raise ValueError("roi width and height must be greater than zero")
