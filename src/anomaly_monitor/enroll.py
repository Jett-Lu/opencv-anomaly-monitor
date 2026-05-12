from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from anomaly_monitor.main import normalize_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture face samples for local face recognition.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help="Person name to remember, for example 'Person A' or 'Jane'.",
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index, video path, or RTSP/HTTP camera URL.",
    )
    parser.add_argument(
        "--known-faces-dir",
        type=Path,
        default=Path("data/known_faces"),
        help="Folder where known face samples are stored.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=8,
        help="Number of face samples to save.",
    )
    return parser.parse_args()


def largest_face(faces) -> tuple[int, int, int, int] | None:
    if len(faces) == 0:
        return None
    return max(faces, key=lambda face: face[2] * face[3])


def next_sample_path(person_dir: Path) -> Path:
    existing = sorted(person_dir.glob("sample_*.jpg"))
    return person_dir / f"sample_{len(existing) + 1:03d}.jpg"


def run_enrollment(
    name: str,
    source: str,
    known_faces_dir: Path,
    count: int,
) -> None:
    if count < 1:
        raise ValueError("count must be at least 1")

    person_dir = known_faces_dir / name
    person_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(normalize_source(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    saved = 0

    print(f"Enrolling face samples for {name}.")
    print("Look at the camera. Press s to save a face sample, q to quit.")

    while saved < count:
        ok, frame = capture.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        face = largest_face(faces)

        if face is not None:
            x, y, width, height = face
            cv2.rectangle(frame, (x, y), (x + width, y + height), (80, 220, 80), 2)

        cv2.putText(
            frame,
            f"{name}: {saved}/{count} saved | s=save q=quit",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Enroll Face", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and face is not None:
            x, y, width, height = face
            face_crop = frame[y : y + height, x : x + width]
            sample_path = next_sample_path(person_dir)
            cv2.imwrite(str(sample_path), face_crop)
            saved += 1
            print(f"Saved {sample_path}")

    capture.release()
    cv2.destroyAllWindows()
    print(f"Saved {saved} face sample(s) for {name}.")


def main() -> None:
    args = parse_args()
    run_enrollment(
        name=args.name,
        source=args.source,
        known_faces_dir=args.known_faces_dir,
        count=args.count,
    )


if __name__ == "__main__":
    main()
