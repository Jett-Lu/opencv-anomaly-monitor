from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FACE_SIZE = (160, 160)
UNKNOWN_PERSON_PREFIX = "Unknown"


@dataclass(frozen=True)
class FaceIdentity:
    name: str
    confidence: float | None
    box: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.box
        return x + width // 2, y + height // 2

    @property
    def is_unknown(self) -> bool:
        return self.name == "unknown person" or self.name.startswith(UNKNOWN_PERSON_PREFIX)


@dataclass
class UnknownFaceTemplate:
    name: str
    template: np.ndarray


class UnknownFaceMemory:
    def __init__(self, match_threshold: float) -> None:
        self.match_threshold = match_threshold
        self.templates: list[UnknownFaceTemplate] = []

    def identify(self, face: np.ndarray) -> tuple[str, float | None]:
        if not self.templates:
            return self._add(face), None

        distances = [
            (self._distance(face, template.template), template)
            for template in self.templates
        ]
        distance, template = min(distances, key=lambda item: item[0])
        if distance <= self.match_threshold:
            template.template = cv2.addWeighted(template.template, 0.85, face, 0.15, 0)
            return template.name, distance

        return self._add(face), None

    def _add(self, face: np.ndarray) -> str:
        name = f"{UNKNOWN_PERSON_PREFIX} {self._label(len(self.templates))}"
        self.templates.append(UnknownFaceTemplate(name=name, template=face.copy()))
        return name

    def _label(self, index: int) -> str:
        alphabet_size = 26
        if index < alphabet_size:
            return chr(ord("A") + index)
        return str(index + 1)

    def _distance(self, first: np.ndarray, second: np.ndarray) -> float:
        difference = cv2.absdiff(first, second)
        return float(np.mean(difference))


class FaceRecognizer:
    """Small local known-face recognizer for demo use."""

    def __init__(
        self,
        known_faces_dir: Path,
        confidence_threshold: float,
        unknown_match_threshold: float = 42.0,
    ) -> None:
        self.known_faces_dir = known_faces_dir
        self.confidence_threshold = confidence_threshold
        self.unknown_faces = UnknownFaceMemory(unknown_match_threshold)
        self.label_names: dict[int, str] = {}
        self.recognizer = self._create_recognizer()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Could not load OpenCV Haar cascade for face detection.")

        self._train()

    @property
    def is_trained(self) -> bool:
        return bool(self.label_names)

    def recognize(self, frame: np.ndarray) -> list[FaceIdentity]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        identities: list[FaceIdentity] = []

        for x, y, width, height in faces:
            face_crop = self._prepare_face(gray[y : y + height, x : x + width])
            name = "unknown person"
            confidence: float | None = None

            if self.is_trained:
                label_id, raw_confidence = self.recognizer.predict(face_crop)
                confidence = float(raw_confidence)
                if confidence <= self.confidence_threshold:
                    name = self.label_names.get(label_id, name)

            if name == "unknown person":
                name, confidence = self.unknown_faces.identify(face_crop)

            identities.append(
                FaceIdentity(
                    name=name,
                    confidence=confidence,
                    box=(int(x), int(y), int(width), int(height)),
                )
            )

        return identities

    def draw(self, frame: np.ndarray, identities: list[FaceIdentity]) -> None:
        for identity in identities:
            x, y, width, height = identity.box
            known = not identity.is_unknown
            color = (80, 220, 80) if known else (80, 180, 255)
            label = identity.name

            if identity.confidence is not None and known:
                label = f"{identity.name} ({identity.confidence:.0f})"

            cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
            cv2.putText(
                frame,
                label,
                (x, max(24, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )

    def _create_recognizer(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "Face recognition needs opencv-contrib-python. "
                "Install project requirements again."
            )
        return cv2.face.LBPHFaceRecognizer_create()

    def _train(self) -> None:
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        training_faces: list[np.ndarray] = []
        training_labels: list[int] = []

        label_id = 0
        for person_dir in sorted(path for path in self.known_faces_dir.iterdir() if path.is_dir()):
            person_faces = self._load_person_faces(person_dir)
            if not person_faces:
                continue

            self.label_names[label_id] = person_dir.name
            training_faces.extend(person_faces)
            training_labels.extend([label_id] * len(person_faces))
            label_id += 1

        if not training_faces:
            print(
                f"No known faces found in {self.known_faces_dir}. "
                "Face boxes will be labeled as unknown person."
            )
            return

        self.recognizer.train(training_faces, np.array(training_labels, dtype=np.int32))
        known_names = ", ".join(self.label_names.values())
        print(f"Loaded known faces: {known_names}")

    def _load_person_faces(self, person_dir: Path) -> list[np.ndarray]:
        faces: list[np.ndarray] = []
        image_paths = [
            path
            for path in sorted(person_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            detected_faces = self._detect_faces(image)
            if len(detected_faces) == 0:
                print(f"No face found in {image_path}; using full image.")
                faces.append(self._prepare_face(image))
                continue

            x, y, width, height = max(detected_faces, key=lambda box: box[2] * box[3])
            faces.append(self._prepare_face(image[y : y + height, x : x + width]))

        return faces

    def _detect_faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [tuple(face) for face in faces]

    def _prepare_face(self, face: np.ndarray) -> np.ndarray:
        equalized = cv2.equalizeHist(face)
        return cv2.resize(equalized, FACE_SIZE)
