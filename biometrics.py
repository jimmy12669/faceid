"""Face detection, quality control and template matching.

What changed from the prototype, and why it matters:

The old code compared a 256-bin grayscale histogram of the face crop. A
histogram knows nothing about a face -- it's a summary of how bright the pixels
are. A printed photo passes. A different person under the same lamp passes. In
testing, a beige wall passes. It is not a biometric comparison, and no amount
of tuning the 0.7 threshold makes it one.

This module uses YuNet for detection and SFace for a 128-dimensional embedding,
both shipped by OpenCV and run locally with no network calls. Same-person pairs
land around 0.9 cosine similarity, different-person pairs around 0.1 -- an
actual decision boundary rather than a coin flip.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

TEMPLATE_MAGIC = b"FTPL"
TEMPLATE_FORMAT = 1

DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNISER_MODEL = "face_recognition_sface_2021dec.onnx"
ENGINE_ID = "yunet2023mar+sface2021dec"


class BiometricError(Exception):
    pass


class EngineUnavailable(BiometricError):
    pass


class NoFaceFound(BiometricError):
    pass


class QualityRejected(BiometricError):
    """The frame had a face, but not one we're willing to enrol or trust."""


@dataclass
class Face:
    """One detected face, in pixel coordinates of the frame it came from."""

    box: tuple[int, int, int, int]  # x, y, w, h
    landmarks: np.ndarray  # 5x2: right eye, left eye, nose, right mouth, left mouth
    score: float
    raw: np.ndarray = field(repr=False)

    @property
    def width(self) -> int:
        return self.box[2]

    @property
    def centre(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    @property
    def eye_distance(self) -> float:
        return float(np.linalg.norm(self.landmarks[1] - self.landmarks[0]))

    @property
    def roll_degrees(self) -> float:
        """Head tilt, from the line between the eyes."""
        dx, dy = self.landmarks[1] - self.landmarks[0]
        return abs(math.degrees(math.atan2(float(dy), float(dx))))

    @property
    def yaw_ratio(self) -> float:
        """Cheap left/right turn estimate: where the nose sits between the eyes.

        0 is facing the camera; positive means the head is turned one way,
        negative the other. Not a calibrated pose angle, but stable enough to
        drive a liveness challenge.
        """
        eye_mid = (self.landmarks[0] + self.landmarks[1]) / 2.0
        span = self.eye_distance or 1.0
        return float((self.landmarks[2][0] - eye_mid[0]) / span)


@dataclass
class Quality:
    sharpness: float
    brightness: float
    face_px: int
    roll_deg: float
    detector_score: float
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons

    def score(self) -> float:
        """A rough 0-1 usefulness number, for ranking samples and logging."""
        sharp = min(self.sharpness / 300.0, 1.0)
        size = min(self.face_px / 220.0, 1.0)
        exposure = 1.0 - abs(self.brightness - 128.0) / 128.0
        return round(max(0.0, 0.4 * sharp + 0.35 * size + 0.25 * exposure), 4)


class FaceEngine:
    """YuNet + SFace, loaded once and reused."""

    def __init__(self, models_dir: Path, *, detector_score: float = 0.85,
                 input_size: tuple[int, int] = (320, 320)):
        self.models_dir = Path(models_dir)
        detector_path = self.models_dir / DETECTOR_MODEL
        recogniser_path = self.models_dir / RECOGNISER_MODEL
        missing = [p.name for p in (detector_path, recogniser_path) if not p.exists()]
        if missing:
            raise EngineUnavailable(
                "missing model file(s): " + ", ".join(missing) + "\n"
                f"Expected under {self.models_dir}. Run: python fetch_models.py"
            )
        self.id = ENGINE_ID
        self._detector = cv2.FaceDetectorYN.create(
            str(detector_path), "", input_size, detector_score, 0.3, 5000
        )
        self._recogniser = cv2.FaceRecognizerSF.create(str(recogniser_path), "")
        self._input_size = input_size

    def detect(self, frame: np.ndarray) -> list[Face]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        self._detector.setInputSize((w, h))
        _, raw = self._detector.detect(frame)
        if raw is None:
            return []
        faces = []
        for row in raw:
            x, y, bw, bh = (int(round(v)) for v in row[:4])
            faces.append(
                Face(
                    box=(x, y, bw, bh),
                    landmarks=np.array(row[4:14], dtype=np.float32).reshape(5, 2),
                    score=float(row[14]),
                    raw=row,
                )
            )
        faces.sort(key=lambda f: f.width * f.box[3], reverse=True)
        return faces

    def detect_single(self, frame: np.ndarray) -> Face:
        """Detection for authentication: exactly one face, or we refuse.

        Two faces in frame is the classic 'hold the phone next to your victim'
        setup, and it's also just ambiguous. Refusing is the safe answer.
        """
        faces = self.detect(frame)
        if not faces:
            raise NoFaceFound("no face in frame")
        if len(faces) > 1:
            raise QualityRejected(f"{len(faces)} faces in frame; step in front alone")
        return faces[0]

    def embed(self, frame: np.ndarray, face: Face) -> np.ndarray:
        """L2-normalised 128-d embedding of an aligned face crop."""
        aligned = self._recogniser.alignCrop(frame, face.raw)
        vector = self._recogniser.feature(aligned).astype(np.float32).ravel()
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise BiometricError("embedding collapsed to zero")
        return vector / norm

    def crop(self, frame: np.ndarray, face: Face) -> np.ndarray:
        return self._recogniser.alignCrop(frame, face.raw)


def assess(frame: np.ndarray, face: Face, settings) -> Quality:
    """Reject inputs too poor to make a trustworthy decision on."""
    x, y, w, h = face.box
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    patch = frame[y0 : min(fh, y + h), x0 : min(fw, x + w)]
    if patch.size == 0:
        return Quality(0.0, 0.0, 0, 0.0, face.score, ["face is outside the frame"])

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    roll = face.roll_degrees
    roll = min(roll, 180.0 - roll)

    reasons: list[str] = []
    if w < settings.min_face_px:
        reasons.append(f"face too small ({w}px, need {settings.min_face_px}px) -- move closer")
    if sharpness < settings.min_sharpness:
        reasons.append("image too blurry -- hold still or clean the lens")
    if brightness < settings.min_brightness:
        reasons.append("too dark -- add light")
    if brightness > settings.max_brightness:
        reasons.append("overexposed -- move away from the light behind you")
    if roll > settings.max_roll_deg:
        reasons.append(f"head tilted {roll:.0f} degrees -- straighten up")
    if face.score < settings.detector_score:
        reasons.append("detector not confident this is a face")
    if x < 0 or y < 0 or x + w > fw or y + h > fh:
        reasons.append("face is cut off at the edge of the frame")

    return Quality(sharpness, brightness, w, roll, face.score, reasons)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Both sides are normalised, so this is just the dot product."""
    return float(np.dot(np.asarray(a).ravel(), np.asarray(b).ravel()))


def build_template(embeddings: list[np.ndarray], *, max_similarity: float = 0.995) -> np.ndarray:
    """Average several samples into one template.

    Averaging normalised embeddings is the standard trick: it cancels
    per-frame noise and widens the margin between the true user and everyone
    else. We insist the samples actually differ, because five copies of one
    frozen frame (a photo held up to the camera) would otherwise look like a
    healthy multi-sample enrolment.
    """
    if len(embeddings) < 2:
        raise BiometricError("need at least two samples to build a template")
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if cosine(embeddings[i], embeddings[j]) > max_similarity:
                raise QualityRejected(
                    "enrolment samples are nearly identical -- these look like the same "
                    "still image rather than a live person"
                )
    stacked = np.stack([np.asarray(e, dtype=np.float32).ravel() for e in embeddings])
    mean = stacked.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        raise BiometricError("samples cancelled out; re-run enrolment")
    return (mean / norm).astype(np.float32)


def cohesion(embeddings: list[np.ndarray]) -> float:
    """Mean pairwise similarity of the enrolment set -- how consistent it is."""
    pairs = [
        cosine(embeddings[i], embeddings[j])
        for i in range(len(embeddings))
        for j in range(i + 1, len(embeddings))
    ]
    return round(sum(pairs) / len(pairs), 4) if pairs else 0.0


def serialise_template(vector: np.ndarray, *, engine: str, samples: int) -> bytes:
    """Small self-describing blob: magic, header length, JSON header, float32 body."""
    body = np.asarray(vector, dtype=np.float32).tobytes()
    header = json.dumps(
        {"format": TEMPLATE_FORMAT, "engine": engine, "dim": int(np.size(vector)),
         "samples": samples, "dtype": "float32"},
        sort_keys=True,
    ).encode("utf-8")
    return TEMPLATE_MAGIC + len(header).to_bytes(2, "big") + header + body


def deserialise_template(blob: bytes) -> tuple[np.ndarray, dict]:
    if not blob.startswith(TEMPLATE_MAGIC):
        raise BiometricError("not a face template")
    header_len = int.from_bytes(blob[4:6], "big")
    header = json.loads(blob[6 : 6 + header_len].decode("utf-8"))
    if header.get("format") != TEMPLATE_FORMAT:
        raise BiometricError(f"unsupported template format {header.get('format')!r}")
    vector = np.frombuffer(blob[6 + header_len :], dtype=np.float32)
    if vector.size != header["dim"]:
        raise BiometricError("template is truncated")
    return vector.copy(), header


@dataclass
class MatchResult:
    uid: str | None
    name: str | None
    similarity: float
    runner_up: float
    accepted: bool
    reason: str


def identify(probe: np.ndarray, candidates: list[tuple[str, str, np.ndarray]], *,
             threshold: float, margin: float) -> MatchResult:
    """1:N search.

    Two conditions, not one. The best match must clear the threshold *and* beat
    the second-best by a margin. Without the margin, a probe that sits halfway
    between two enrolled people gets assigned to whichever is a hair closer --
    which is exactly the case where you want a human to look up.
    """
    if not candidates:
        return MatchResult(None, None, 0.0, 0.0, False, "no active enrolments")

    scored = sorted(
        ((cosine(probe, tpl), uid, name) for uid, name, tpl in candidates), reverse=True
    )
    best, uid, name = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0

    if best < threshold:
        return MatchResult(None, None, best, second, False, "no enrolment matched")
    if len(scored) > 1 and (best - second) < margin:
        return MatchResult(
            None, None, best, second, False,
            "match was ambiguous between two enrolments",
        )
    return MatchResult(uid, name, best, second, True, "matched")


def verify_against(probe: np.ndarray, template: np.ndarray, *, threshold: float) -> MatchResult:
    """1:1 check against a claimed identity."""
    similarity = cosine(probe, template)
    if similarity < threshold:
        return MatchResult(None, None, similarity, 0.0, False, "face did not match the claim")
    return MatchResult(None, None, similarity, 0.0, True, "matched")
