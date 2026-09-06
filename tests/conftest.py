"""Shared fixtures.

The interesting fixture here is FakeCamera. Authentication is a loop over
frames from a webcam, which is exactly the kind of thing that ends up untested
because "you need a camera for that". You don't: you need something that
produces frames and a face that moves plausibly. So the fake camera renders a
static background with a face patch that turns and zooms on a fixed schedule,
and the liveness state machine can't tell the difference.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
import crypto  # noqa: E402
import database  # noqa: E402

FRAME_W, FRAME_H = 640, 480
FACE_BOX = (240, 140, 180, 200)


@pytest.fixture
def settings(tmp_path):
    s = config.Settings(home=tmp_path / "state")
    s.scrypt_n = 1 << 12  # keep the tests quick; production uses 2**15
    s.min_response_ms = 0
    s.enroll_samples = 3
    s.liveness_challenges = 2
    return s


@pytest.fixture
def keyring(settings):
    settings.ensure_home()
    return crypto.KeyStore(settings.keystore_path).create(
        "test passphrase 123", n=settings.scrypt_n, r=settings.scrypt_r, p=settings.scrypt_p
    )


@pytest.fixture
def store(settings, keyring):
    s = database.Store(settings.db_path, keyring, mirror=settings.audit_mirror_path)
    yield s
    s.close()


# -- stand-ins for the camera and the recognition engine -------------------


@dataclass
class FakeFace:
    box: tuple[int, int, int, int]
    yaw_ratio: float
    score: float = 0.99

    @property
    def width(self) -> int:
        return self.box[2]

    @property
    def landmarks(self) -> np.ndarray:
        x, y, w, h = self.box
        eye_y = y + h * 0.4
        left = (x + w * 0.32, eye_y)
        right = (x + w * 0.68, eye_y)
        span = right[0] - left[0]
        nose_x = (left[0] + right[0]) / 2 + self.yaw_ratio * span
        return np.array(
            [left, right, (nose_x, y + h * 0.6), (x + w * 0.38, y + h * 0.78),
             (x + w * 0.62, y + h * 0.78)],
            dtype=np.float32,
        )

    @property
    def roll_degrees(self) -> float:
        return 0.0


class FakeCamera:
    """A cooperative user, simulated.

    Renders a face patch against a fixed background. Give it the prompt the
    liveness session is currently showing (see `obedient`) and it moves the way
    a person who read that prompt would: gradually, from wherever it currently
    is. That responsiveness is the point -- the challenges are deliberately
    relative to the pose you are in when they are asked, so a fake that follows
    a fixed script cannot satisfy them, and neither can a recording.
    """

    #: frames between seeing a new prompt and starting to move. Real people
    #: are not instant, and the session takes its reference pose in that gap.
    REACTION_FRAMES = 9

    def __init__(self, seed: int = 7, frames: int | None = None):
        rng = np.random.default_rng(seed)
        self.background = rng.integers(55, 205, (FRAME_H, FRAME_W, 3), dtype=np.uint8)
        self.texture = rng.integers(60, 200, (FACE_BOX[3], FACE_BOX[2], 3), dtype=np.uint8)
        self.i = 0
        self.limit = frames
        self.yaw = 0.0
        self.scale = 1.0
        self.target_yaw = 0.0
        self.target_scale = 1.0
        self._prompt = ""
        self._react_at: int | None = None
        self.noise = np.random.default_rng(seed + 1)

    def asked(self, prompt: str) -> None:
        prompt = (prompt or "").lower()
        if prompt and prompt != self._prompt:
            self._prompt = prompt
            self._react_at = self.i + self.REACTION_FRAMES

    def _react(self) -> None:
        prompt = self._prompt
        if "left shoulder" in prompt:
            self.target_yaw = self.yaw + 0.35
        elif "right shoulder" in prompt:
            self.target_yaw = self.yaw - 0.35
        elif "closer" in prompt:
            self.target_scale = min(self.scale * 1.5, 1.8)
        elif "lean back" in prompt:
            self.target_scale = max(self.scale * 0.6, 0.55)
        elif "straight at the camera" in prompt:
            self.target_yaw, self.target_scale = 0.0, 1.0

    def face(self) -> FakeFace:
        x, y, w, h = FACE_BOX
        nw, nh = int(w * self.scale), int(h * self.scale)
        jitter = int(3 * np.sin(self.i / 3.0))  # nobody holds perfectly still
        return FakeFace((x + (w - nw) // 2 + jitter, y + (h - nh) // 2, nw, nh), self.yaw)

    def read(self):
        if self.limit is not None and self.i >= self.limit:
            return None
        if self._react_at is not None and self.i >= self._react_at:
            self._react()
            self._react_at = None
        # ease towards whatever was last asked for, with the small involuntary
        # drift any real head has
        self.yaw += (self.target_yaw - self.yaw) * 0.34 + self.noise.normal(0, 0.006)
        self.scale += (self.target_scale - self.scale) * 0.34 + self.noise.normal(0, 0.004)
        frame = self.background.copy()
        x, y, w, h = self.face().box
        frame[y : y + h, x : x + w] = np.resize(self.texture, (h, w, 3)).astype(np.uint8)
        self.i += 1
        return frame


def obedient(camera: "FakeCamera"):
    """Progress hook that relays the on-screen prompt to the fake user."""

    def hook(update):
        if update.progress is not None:
            camera.asked(update.progress.prompt)

    return hook


class FakeEngine:
    """Deterministic embeddings, so matching behaviour is testable."""

    id = "fake-engine-v1"

    def __init__(self, identity: int = 0, noise: float = 0.09, seed: int = 3):
        self.rng = np.random.default_rng(seed)
        self.identity = identity
        self.noise = noise
        self._camera: FakeCamera | None = None

    def bind(self, camera: FakeCamera) -> "FakeEngine":
        self._camera = camera
        return self

    def detect_single(self, frame):
        assert self._camera is not None, "bind the camera first"
        return self._camera.face()

    def embed(self, frame, face) -> np.ndarray:
        base = np.zeros(128, dtype=np.float32)
        base[self.identity * 8 : self.identity * 8 + 8] = 1.0
        vector = base + self.rng.normal(0, self.noise, 128).astype(np.float32)
        return vector / np.linalg.norm(vector)

    def crop(self, frame, face):
        x, y, w, h = face.box
        return frame[y : y + h, x : x + w]


@pytest.fixture
def camera():
    return FakeCamera()


@pytest.fixture
def engine(camera):
    return FakeEngine().bind(camera)
