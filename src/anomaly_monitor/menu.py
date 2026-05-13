from __future__ import annotations

import json
import shutil
from pathlib import Path

from anomaly_monitor.config import (
    DEFAULT_ARCFACE_MODEL_PATH,
    DEFAULT_KNOWN_FACES_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_POSE_MODEL_PATH,
    MonitorConfig,
)
from anomaly_monitor.enroll import run_enrollment
from anomaly_monitor.main import parse_roi, run_monitor
from anomaly_monitor.names import normalize_person_name


def prompt_text(label: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    value = input(f"{label}{suffix}: ").strip()
    if value:
        return value
    if default is not None:
        return default
    return ""


def prompt_int(label: str, default: int) -> int:
    while True:
        value = prompt_text(label, str(default))
        try:
            return int(value)
        except ValueError:
            print("Please enter a whole number.")


def prompt_float(label: str, default: float) -> float:
    while True:
        value = prompt_text(label, str(default))
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def prompt_yes_no(label: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    while True:
        value = prompt_text(f"{label} (y/n)", default_text).lower()
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please enter y or n.")


def prompt_face_engine(default: str = "arcface") -> str:
    while True:
        value = prompt_text("Face engine, arcface or lbph", default).lower()
        if value in {"arcface", "lbph"}:
            return value
        print("Please enter arcface or lbph.")


def pause() -> None:
    input("\nPress Enter to continue...")


def demo_config(source: str = "0") -> MonitorConfig:
    return MonitorConfig(
        source=source,
        output_dir=DEFAULT_OUTPUT_DIR,
        known_faces_dir=DEFAULT_KNOWN_FACES_DIR,
        face_engine="arcface",
        arcface_model_path=DEFAULT_ARCFACE_MODEL_PATH,
        pose_model_path=DEFAULT_POSE_MODEL_PATH,
        pose_threshold=0.9,
        wrist_speed_threshold=3.0,
        max_poses=4,
        alert_hold_seconds=8.0,
        event_video_seconds=5.0,
        pre_alert_seconds=2.0,
        post_alert_seconds=3.0,
    )


def begin_monitoring_default() -> None:
    source = prompt_text("Camera/video source", "0")
    print("\nStarting monitor. Press q in the camera window to stop.")
    run_monitor(demo_config(source))


def begin_monitoring_custom() -> None:
    source = prompt_text("Camera/video source", "0")
    t_pose_only = prompt_yes_no("Only alert on T-pose test behavior", True)
    pose_threshold = prompt_float("Pose alert threshold", 0.9)
    wrist_speed_threshold = prompt_float("Wrist speed threshold", 3.0)
    loitering_seconds = prompt_float("Loitering seconds", 30.0)
    roi_dwell_seconds = prompt_float("Restricted-zone dwell seconds", 8.0)
    motion_history_seconds = prompt_float("Motion history seconds", 20.0)
    rapid_body_speed_threshold = prompt_float("Rapid body speed threshold", 1.5)
    max_poses = prompt_int("Maximum people/skeletons", 4)
    alert_hold_seconds = prompt_float("Keep alert label visible for seconds", 8.0)
    pre_alert_seconds = prompt_float("Seconds before alert in event clip", 2.0)
    post_alert_seconds = prompt_float("Seconds after alert in event clip", 3.0)
    event_video_fps = prompt_float("Event clip FPS", 12.0)
    face_engine = prompt_face_engine("arcface")
    face_confidence_threshold = prompt_float("Face threshold, lower is stricter", 75.0)
    unknown_face_match_threshold = prompt_float("Unknown face match threshold", 42.0)
    arcface_similarity_threshold = prompt_float("ArcFace similarity threshold", 0.34)
    arcface_similarity_margin = prompt_float("ArcFace best-match margin", 0.03)
    identity_alert_hold_seconds = prompt_float("Remember flagged identity seconds", 300.0)
    roi_text = prompt_text("Restricted zone ROI x,y,width,height, or blank", "")
    roi = parse_roi(roi_text) if roi_text else None
    show_motion_boxes = prompt_yes_no("Show motion boxes", False)
    motion_alerts = prompt_yes_no("Enable motion-only alerts", False)
    tracking = prompt_yes_no("Enable person tracking and motion history", False)
    face_recognition = prompt_yes_no("Enable face recognition", True)

    config = MonitorConfig(
        source=source,
        output_dir=DEFAULT_OUTPUT_DIR,
        known_faces_dir=DEFAULT_KNOWN_FACES_DIR,
        face_engine=face_engine,
        face_confidence_threshold=face_confidence_threshold,
        unknown_face_match_threshold=unknown_face_match_threshold,
        arcface_model_path=DEFAULT_ARCFACE_MODEL_PATH,
        arcface_similarity_threshold=arcface_similarity_threshold,
        arcface_similarity_margin=arcface_similarity_margin,
        identity_alert_hold_seconds=identity_alert_hold_seconds,
        pose_threshold=pose_threshold,
        wrist_speed_threshold=wrist_speed_threshold,
        t_pose_only=t_pose_only,
        loitering_seconds=loitering_seconds,
        roi_dwell_seconds=roi_dwell_seconds,
        motion_history_seconds=motion_history_seconds,
        rapid_body_speed_threshold=rapid_body_speed_threshold,
        max_poses=max_poses,
        pose_model_path=DEFAULT_POSE_MODEL_PATH,
        alert_hold_seconds=alert_hold_seconds,
        event_video_seconds=pre_alert_seconds + post_alert_seconds,
        pre_alert_seconds=pre_alert_seconds,
        post_alert_seconds=post_alert_seconds,
        event_video_fps=event_video_fps,
        roi=roi,
        show_motion_boxes=show_motion_boxes,
        enable_motion_alerts=motion_alerts,
        enable_tracking=tracking,
        enable_face_recognition=face_recognition,
    )

    print("\nStarting monitor. Press q in the camera window to stop.")
    run_monitor(config)


def enroll_face() -> None:
    name = prompt_text("Name to remember, for example Person A")
    try:
        name = normalize_person_name(name)
    except ValueError as exc:
        print(f"Invalid name: {exc}")
        return

    source = prompt_text("Camera/video source", "0")
    count = prompt_int("Number of face samples", 8)
    run_enrollment(
        name=name,
        source=source,
        known_faces_dir=DEFAULT_KNOWN_FACES_DIR,
        count=count,
    )


def known_face_dirs() -> list[Path]:
    DEFAULT_KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(path for path in DEFAULT_KNOWN_FACES_DIR.iterdir() if path.is_dir())


def list_faces() -> None:
    people = known_face_dirs()
    if not people:
        print("No known faces yet.")
        return

    print("\nKnown faces:")
    for index, person_dir in enumerate(people, start=1):
        image_count = len(
            [
                path
                for path in person_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            ]
        )
        print(f"{index}. {person_dir.name} ({image_count} image files)")


def choose_face_label() -> Path | None:
    people = known_face_dirs()
    if not people:
        print("No known faces yet.")
        return None

    list_faces()
    choice = prompt_int("Choose person number", 1)
    if choice < 1 or choice > len(people):
        print("Invalid person number.")
        return None
    return people[choice - 1]


def rename_face_label() -> None:
    person_dir = choose_face_label()
    if person_dir is None:
        return

    try:
        new_name = normalize_person_name(prompt_text("New name", person_dir.name))
    except ValueError as exc:
        print(f"Invalid name: {exc}")
        return

    if new_name == person_dir.name:
        print("Name unchanged.")
        return

    target = person_dir.parent / new_name
    if target.exists():
        print(f"A known face folder named {new_name} already exists.")
        return

    person_dir.rename(target)
    print(f"Renamed {person_dir.name} to {new_name}. Restart the monitor to reload names.")


def add_samples_to_face() -> None:
    person_dir = choose_face_label()
    if person_dir is None:
        return

    source = prompt_text("Camera/video source", "0")
    count = prompt_int("Additional face samples", 5)
    run_enrollment(
        name=person_dir.name,
        source=source,
        known_faces_dir=DEFAULT_KNOWN_FACES_DIR,
        count=count,
    )


def delete_face_label() -> None:
    person_dir = choose_face_label()
    if person_dir is None:
        return

    print(f"This deletes all saved face samples for {person_dir.name}.")
    confirm = prompt_text(f"Type {person_dir.name} to confirm delete")
    if confirm != person_dir.name:
        print("Delete cancelled.")
        return

    shutil.rmtree(person_dir)
    print(f"Deleted {person_dir.name}. Restart the monitor to reload names.")


def edit_faces_menu() -> None:
    while True:
        print(
            "\nEdit known faces\n"
            "1. List known faces\n"
            "2. Rename a face label\n"
            "3. Add more samples to a person\n"
            "4. Delete a person\n"
            "0. Back"
        )
        choice = prompt_text("Choose", "0")
        if choice == "1":
            list_faces()
            pause()
        elif choice == "2":
            rename_face_label()
            pause()
        elif choice == "3":
            add_samples_to_face()
            pause()
        elif choice == "4":
            delete_face_label()
            pause()
        elif choice == "0":
            return
        else:
            print("Unknown option.")


def show_recent_alerts(limit: int = 10) -> None:
    log_path = DEFAULT_OUTPUT_DIR / "events.jsonl"
    if not log_path.exists():
        print("No alert log found yet.")
        return

    lines = log_path.read_text(encoding="utf-8").splitlines()[-limit:]
    if not lines:
        print("Alert log is empty.")
        return

    print(f"\nLast {len(lines)} alert(s):")
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(line)
            continue

        labels = ", ".join(event.get("labels", [])) or "none"
        print(
            f"- {event.get('timestamp')} | {event.get('identity')} | "
            f"score={event.get('score')} track={event.get('tracking_score', 0)} | labels={labels}"
        )
        print(f"  snapshot: {event.get('snapshot_path')}")
        if event.get("video_path"):
            print(f"  video:    {event.get('video_path')}")


def show_paths() -> None:
    print("\nProject paths:")
    print(f"- Known faces: {DEFAULT_KNOWN_FACES_DIR.resolve()}")
    print(f"- Alert output: {DEFAULT_OUTPUT_DIR.resolve()}")
    print(f"- Events log: {(DEFAULT_OUTPUT_DIR / 'events.jsonl').resolve()}")
    print(f"- Pose model: {DEFAULT_POSE_MODEL_PATH.resolve()}")
    print(f"- ArcFace model: {DEFAULT_ARCFACE_MODEL_PATH.resolve()}")


def main() -> None:
    while True:
        print(
            "\nCamera Anomaly Monitor\n"
            "1. Begin monitoring and logging\n"
            "2. Begin monitoring with custom settings\n"
            "3. Enroll a new face\n"
            "4. Edit known faces\n"
            "5. View recent alerts\n"
            "6. Show project folders\n"
            "0. Exit"
        )
        choice = prompt_text("Choose", "1")
        try:
            if choice == "1":
                begin_monitoring_default()
            elif choice == "2":
                begin_monitoring_custom()
            elif choice == "3":
                enroll_face()
            elif choice == "4":
                edit_faces_menu()
            elif choice == "5":
                show_recent_alerts()
                pause()
            elif choice == "6":
                show_paths()
                pause()
            elif choice == "0":
                print("Goodbye.")
                return
            else:
                print("Unknown option.")
        except KeyboardInterrupt:
            print("\nCancelled.")
        except Exception as exc:
            print(f"Error: {exc}")
            pause()


if __name__ == "__main__":
    main()
