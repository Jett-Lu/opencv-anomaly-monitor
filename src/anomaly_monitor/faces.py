from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LBPH_FACE_SIZE = (160, 160)
ARCFACE_FACE_SIZE = (112, 112)
ARCFACE_REPO_ID = "onnx-community/arcface-onnx"
ARCFACE_FILENAME = "arcface.onnx"
MIN_ARCFACE_MODEL_BYTES = 1_000_000
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


@dataclass(frozen=True)
class KnownFaceEmbedding:
    name: str
    embedding: np.ndarray
    source_path: Path


@dataclass
class UnknownEmbeddingTemplate:
    name: str
    embedding: np.ndarray


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


class UnknownEmbeddingMemory:
    def __init__(self, similarity_threshold: float) -> None:
        self.similarity_threshold = similarity_threshold
        self.templates: list[UnknownEmbeddingTemplate] = []

    def identify(self, embedding: np.ndarray) -> tuple[str, float | None]:
        if not self.templates:
            return self._add(embedding), None

        similarities = [
            (cosine_similarity(embedding, template.embedding), template)
            for template in self.templates
        ]
        similarity, template = max(similarities, key=lambda item: item[0])
        if similarity >= self.similarity_threshold:
            template.embedding = normalize_embedding(
                template.embedding * 0.85 + embedding * 0.15
            )
            return template.name, similarity * 100.0

        return self._add(embedding), None

    def _add(self, embedding: np.ndarray) -> str:
        name = f"{UNKNOWN_PERSON_PREFIX} {self._label(len(self.templates))}"
        self.templates.append(
            UnknownEmbeddingTemplate(name=name, embedding=embedding.copy())
        )
        return name

    def _label(self, index: int) -> str:
        alphabet_size = 26
        if index < alphabet_size:
            return chr(ord("A") + index)
        return str(index + 1)


class ArcFaceEmbeddingModel:
    def __init__(self, model_path: Path) -> None:
        model_path = ensure_arcface_model(model_path)
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ArcFace face recognition needs onnxruntime. "
                "Run: .venv312-run\\Scripts\\python.exe -m pip install -r requirements.txt"
            ) from exc

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.output_name = self.session.get_outputs()[0].name
        self.input_layout = self._input_layout(self.input.shape)
        print(f"Loaded ArcFace model: {model_path}")

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        model_input = self._preprocess(face_bgr)
        result = self.session.run([self.output_name], {self.input_name: model_input})[0]
        embedding = np.asarray(result, dtype=np.float32).reshape(-1)
        return normalize_embedding(embedding)

    def _preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_bgr, ARCFACE_FACE_SIZE)
        model_input = resized.astype(np.float32)

        if self.input_layout == "nchw":
            model_input = np.transpose(model_input, (2, 0, 1))

        return model_input[np.newaxis, ...].astype(np.float32)

    def _input_layout(self, shape: list | tuple) -> str:
        if len(shape) == 4 and shape[1] == 3:
            return "nchw"
        return "nhwc"


class FaceRecognizer:
    """Local known-face recognizer with ArcFace embeddings or OpenCV LBPH."""

    def __init__(
        self,
        known_faces_dir: Path,
        confidence_threshold: float,
        unknown_match_threshold: float = 42.0,
        engine: str = "arcface",
        arcface_model_path: Path | None = None,
        arcface_similarity_threshold: float = 0.34,
        arcface_similarity_margin: float = 0.03,
    ) -> None:
        self.known_faces_dir = known_faces_dir
        self.confidence_threshold = confidence_threshold
        self.unknown_faces = UnknownFaceMemory(unknown_match_threshold)
        self.unknown_embeddings = UnknownEmbeddingMemory(arcface_similarity_threshold)
        self.arcface_similarity_threshold = arcface_similarity_threshold
        self.arcface_similarity_margin = arcface_similarity_margin
        self.label_names: dict[int, str] = {}
        self.recognizer = None
        self.arcface: ArcFaceEmbeddingModel | None = None
        self.known_embeddings: list[KnownFaceEmbedding] = []
        self.engine = engine.lower().strip()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        if self.face_cascade.empty():
            raise RuntimeError("Could not load OpenCV Haar cascade for face detection.")

        if self.engine == "arcface":
            try:
                if arcface_model_path is None:
                    raise RuntimeError("ArcFace model path was not configured.")
                self.arcface = ArcFaceEmbeddingModel(arcface_model_path)
                self._load_arcface_embeddings()
                return
            except Exception as exc:
                print(f"ArcFace unavailable ({exc}). Falling back to OpenCV LBPH.")

        self.engine = "lbph"
        self.recognizer = self._create_lbph_recognizer()
        self._train_lbph()

    @property
    def is_trained(self) -> bool:
        if self.engine == "arcface":
            return bool(self.known_embeddings)
        return bool(self.label_names)

    def recognize(self, frame: np.ndarray) -> list[FaceIdentity]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        identities: list[FaceIdentity] = []

        for x, y, width, height in faces:
            name = "unknown person"
            confidence: float | None = None

            if self.engine == "arcface" and self.arcface is not None:
                face_crop = self._crop_bgr(frame, (x, y, width, height), margin=0.22)
                embedding = self.arcface.embed(face_crop)
                name, confidence = self._match_arcface_embedding(embedding)
                if name == "unknown person":
                    name, confidence = self.unknown_embeddings.identify(embedding)
            else:
                face_crop = self._prepare_lbph_face(gray[y : y + height, x : x + width])
                if self.is_trained and self.recognizer is not None:
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

    def _create_lbph_recognizer(self):
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "Face recognition needs opencv-contrib-python. "
                "Install project requirements again."
            )
        return cv2.face.LBPHFaceRecognizer_create()

    def _load_arcface_embeddings(self) -> None:
        if self.arcface is None:
            return

        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        for person_dir in self._person_dirs():
            image_paths = self._image_paths(person_dir)
            for image_path in image_paths:
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    continue

                face_crop = self._largest_face_bgr(image)
                if face_crop is None:
                    print(f"No face found in {image_path}; using full image.")
                    face_crop = image

                embedding = self.arcface.embed(face_crop)
                self.known_embeddings.append(
                    KnownFaceEmbedding(
                        name=person_dir.name,
                        embedding=embedding,
                        source_path=image_path,
                    )
                )

        if not self.known_embeddings:
            print(
                f"No known faces found in {self.known_faces_dir}. "
                "Face boxes will be labeled as unknown person."
            )
            return

        known_names = sorted({item.name for item in self.known_embeddings})
        print(
            "Loaded ArcFace known faces: "
            f"{', '.join(known_names)} ({len(self.known_embeddings)} samples)"
        )

    def _match_arcface_embedding(self, embedding: np.ndarray) -> tuple[str, float | None]:
        if not self.known_embeddings:
            return "unknown person", None

        scores = sorted(
            (
                (cosine_similarity(embedding, known.embedding), known)
                for known in self.known_embeddings
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, best_match = scores[0]
        second_other_score = max(
            (
                score
                for score, known in scores[1:]
                if known.name != best_match.name
            ),
            default=-1.0,
        )
        margin = best_score - second_other_score

        if (
            best_score >= self.arcface_similarity_threshold
            and margin >= self.arcface_similarity_margin
        ):
            return best_match.name, best_score * 100.0

        return "unknown person", best_score * 100.0

    def _train_lbph(self) -> None:
        self.known_faces_dir.mkdir(parents=True, exist_ok=True)
        training_faces: list[np.ndarray] = []
        training_labels: list[int] = []

        label_id = 0
        for person_dir in self._person_dirs():
            person_faces = self._load_person_lbph_faces(person_dir)
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

        if self.recognizer is None:
            return

        self.recognizer.train(training_faces, np.array(training_labels, dtype=np.int32))
        known_names = ", ".join(self.label_names.values())
        print(f"Loaded LBPH known faces: {known_names}")

    def _load_person_lbph_faces(self, person_dir: Path) -> list[np.ndarray]:
        faces: list[np.ndarray] = []
        for image_path in self._image_paths(person_dir):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            detected_faces = self._detect_faces(image)
            if len(detected_faces) == 0:
                print(f"No face found in {image_path}; using full image.")
                faces.append(self._prepare_lbph_face(image))
                continue

            x, y, width, height = max(detected_faces, key=lambda box: box[2] * box[3])
            faces.append(self._prepare_lbph_face(image[y : y + height, x : x + width]))

        return faces

    def _person_dirs(self) -> list[Path]:
        return sorted(path for path in self.known_faces_dir.iterdir() if path.is_dir())

    def _image_paths(self, person_dir: Path) -> list[Path]:
        return [
            path
            for path in sorted(person_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def _detect_faces(self, gray: np.ndarray) -> list[tuple[int, int, int, int]]:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [tuple(face) for face in faces]

    def _largest_face_bgr(self, image: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self._detect_faces(gray)
        if not faces:
            return None

        box = max(faces, key=lambda item: item[2] * item[3])
        return self._crop_bgr(image, box, margin=0.22)

    def _crop_bgr(
        self,
        image: np.ndarray,
        box: tuple[int, int, int, int],
        margin: float = 0.0,
    ) -> np.ndarray:
        x, y, width, height = box
        expand_x = int(width * margin)
        expand_y = int(height * margin)
        left = max(0, x - expand_x)
        top = max(0, y - expand_y)
        right = min(image.shape[1], x + width + expand_x)
        bottom = min(image.shape[0], y + height + expand_y)
        return image[top:bottom, left:right]

    def _prepare_lbph_face(self, face: np.ndarray) -> np.ndarray:
        equalized = cv2.equalizeHist(face)
        return cv2.resize(equalized, LBPH_FACE_SIZE)


def ensure_arcface_model(model_path: Path) -> Path:
    if model_path.exists() and model_path.stat().st_size >= MIN_ARCFACE_MODEL_BYTES:
        return model_path

    if model_path.exists():
        model_path.unlink()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "ArcFace model download needs huggingface_hub. "
            "Run: .venv312-run\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from exc

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ArcFace model from Hugging Face to {model_path}...")
    downloaded_path = Path(
        hf_hub_download(
            repo_id=ARCFACE_REPO_ID,
            filename=ARCFACE_FILENAME,
            local_dir=model_path.parent,
        )
    )
    if downloaded_path.resolve() != model_path.resolve():
        shutil.copy2(downloaded_path, model_path)

    if model_path.stat().st_size < MIN_ARCFACE_MODEL_BYTES:
        model_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded ArcFace model file was not a valid ONNX model.")

    return model_path


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm == 0.0:
        return embedding
    return embedding / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.dot(first, second))
