"""Presentation-attack resistance: is there a live person in front of the lens?

The prototype had no answer to this at all -- hold up a phone showing the
enrolled user's photo and it matched happily. Two layers here:

  1. An active challenge. The system asks for a short, randomly ordered
     sequence of head movements with a deadline. A still photo can't comply,
     and a pre-recorded video can only comply if it happens to contain the
     right moves in the right order inside the window.

  2. Passive checks that run on every frame regardless: is the frame moving at
     all, does the face texture look like skin rather than paper, does the head
     shift independently of the room behind it, does its geometry change the
     way a 3D head does.

     An earlier version of this file also looked for screen moire in the
     frequency domain. It was cut: measured against blurred prints and
     simulated panels it did not separate them from genuine faces, and a
     check that fires at random is worse than no check.

Be clear-eyed about what this is. These are heuristics, not a certified
presentation-attack detection system (ISO/IEC 30107-3). They raise the cost of
a photo or a laptop-screen replay from "zero effort" to "needs a decent video
of the target and good timing". A determined attacker with a 3D mask or an
injected video stream will get through, which is exactly why face is one
factor here and not the whole login.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

# Analysis runs on the raw (unmirrored) camera frame. Turning your head towards
# your own left shoulder swings the nose towards the right-hand side of that
# frame, so a leftward turn reads as a positive yaw delta. Some webcams mirror
# in hardware; pass invert_yaw=True if the challenges only pass backwards.
_YAW_SIGN = {"turn_left": +1.0, "turn_right": -1.0}


class Challenge(str, Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    MOVE_CLOSER = "move_closer"
    MOVE_BACK = "move_back"

    @property
    def prompt(self) -> str:
        return {
            Challenge.TURN_LEFT: "Turn your head towards your left shoulder",
            Challenge.TURN_RIGHT: "Turn your head towards your right shoulder",
            Challenge.MOVE_CLOSER: "Lean a little closer to the camera",
            Challenge.MOVE_BACK: "Lean back from the camera",
        }[self]


class LivenessState(str, Enum):
    CALIBRATING = "calibrating"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class PadReport:
    """Passive presentation-attack signals for the current frame."""

    score: float
    flags: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)

    @property
    def suspicious(self) -> bool:
        return bool(self.flags)


class PassiveDetector:
    """Frame-by-frame spoof signals. Cheap enough to run at video rate."""

    #: below this much movement in the face box there is nothing to judge
    MOTION_FLOOR = 3.0

    def __init__(self, *, static_frame_limit: int = 12, history: int = 30):
        self.static_frame_limit = static_frame_limit
        self._last_hash: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._static_frames = 0
        self._yaw_history: list[float] = []
        self._width_history: list[float] = []
        self._context_history: list[float] = []
        self._history = history

    @staticmethod
    def _average_hash(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (16, 16))
        return (small > small.mean()).astype(np.uint8)

    def _motion_split(self, gray: np.ndarray, box) -> tuple[float, float]:
        """Mean absolute change inside the face box versus everywhere else."""
        if self._last_gray is None or self._last_gray.shape != gray.shape:
            return 0.0, 0.0
        delta = cv2.absdiff(self._last_gray, gray).astype(np.float32)
        x, y, w, h = box
        mask = np.zeros(delta.shape, dtype=bool)
        mask[max(0, y) : y + h, max(0, x) : x + w] = True
        if not mask.any() or mask.all():
            return 0.0, 0.0
        return float(delta[mask].mean()), float(delta[~mask].mean())

    def update(self, frame: np.ndarray, face, crop: np.ndarray | None = None) -> PadReport:
        flags: list[str] = []
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Is the image moving at all? A held-up photo often isn't, and a
        #    frame injected into the video device is frequently pixel-identical.
        digest = self._average_hash(frame)
        if self._last_hash is not None and int(np.abs(digest - self._last_hash).sum()) <= 2:
            self._static_frames += 1
        else:
            self._static_frames = 0
        self._last_hash = digest
        if self._static_frames >= self.static_frame_limit:
            flags.append("video is frozen or the same image is being replayed")
        static_score = 0.0 if self._static_frames >= self.static_frame_limit else 1.0

        # 2. Texture. Prints and low-quality replays lose fine detail; in
        #    testing a blurred print of a face dropped Laplacian variance by
        #    more than an order of magnitude.
        if crop is None:
            x, y, w, h = face.box
            crop = frame[max(0, y) : y + h, max(0, x) : x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if gray.size == 0:
            return PadReport(0.0, ["face crop was empty"], {})
        texture = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        texture_score = min(texture / 120.0, 1.0)
        if texture_score < 0.25:
            flags.append("face texture is unusually flat for skin")

        # 3. Does the face move independently of the room behind it?
        #    A person shifts against a still background. When the *picture*
        #    moves instead of the person -- a monitor filling the frame, a
        #    printout waved at the lens, an injected video stream -- the pixels
        #    outside the face move about as much as the ones inside it.
        #    Note the gap: a phone held up in an otherwise still room leaves a
        #    real static background, and this check will not catch it. The
        #    active challenge is what covers that case.
        face_motion, background_motion = self._motion_split(gray_frame, face.box)
        if face_motion > self.MOTION_FLOOR:
            ratio = background_motion / (face_motion + 1e-3)
            context_score = 1.0 - min(max(ratio - 0.20, 0.0) / 0.60, 1.0)
            self._context_history.append(context_score)
            self._context_history = self._context_history[-self._history :]
        # Judge on the recent average: one hand wave shouldn't condemn anyone.
        context_score = (
            float(np.mean(self._context_history)) if self._context_history else 1.0
        )
        if self._context_history and context_score < 0.4:
            flags.append("the whole scene moves with the face, as if held in front of the lens")
        self._last_gray = gray_frame

        # 4. Micro-motion. A real head never holds perfectly still; the
        #    relationship between nose and eyes shifts even when you try.
        self._yaw_history.append(face.yaw_ratio)
        self._width_history.append(float(face.width))
        self._yaw_history = self._yaw_history[-self._history :]
        self._width_history = self._width_history[-self._history :]
        if len(self._yaw_history) >= 10:
            # Thresholds sit near the landmark noise floor of a typical webcam:
            # high enough that a rigid image scores zero, low enough that
            # someone deliberately holding still is not called a fake.
            yaw_var = float(np.std(self._yaw_history))
            width_var = float(np.std(self._width_history) / max(np.mean(self._width_history), 1.0))
            motion_score = min(yaw_var / 0.012, 1.0) * 0.6 + min(width_var / 0.006, 1.0) * 0.4
            motion_score = min(motion_score, 1.0)
            if motion_score < 0.15:
                flags.append("no natural head micro-movement")
        else:
            motion_score = 1.0  # not enough history yet; don't penalise

        components = {
            "static": round(static_score, 3),
            "texture": round(texture_score, 3),
            "context": round(context_score, 3),
            "motion": round(motion_score, 3),
        }
        score = (
            0.25 * static_score + 0.20 * texture_score + 0.30 * context_score + 0.25 * motion_score
        )
        return PadReport(round(score, 3), flags, components)


@dataclass
class Progress:
    state: LivenessState
    prompt: str
    message: str = ""
    index: int = 0
    total: int = 0
    pad: PadReport | None = None

    @property
    def done(self) -> bool:
        return self.state in (LivenessState.PASSED, LivenessState.FAILED)


class LivenessSession:
    """Runs the challenge sequence for one authentication attempt.

    Feed it every frame. It answers with what to show the user next.
    """

    #: how far the yaw ratio must move from the reference pose
    YAW_DELTA = 0.14
    #: relative change in face width for the proximity challenges
    SIZE_DELTA = 0.16
    #: a face this far off centre is not a usable starting pose
    FRONTAL_LIMIT = 0.12
    CALIBRATION_FRAMES = 8
    #: frames averaged into the reference pose for each challenge
    REFERENCE_WINDOW = 8

    def __init__(self, settings, *, rng=None, invert_yaw: bool = False,
                 challenges: list[Challenge] | None = None):
        self.settings = settings
        self.invert_yaw = invert_yaw
        rng = rng or secrets.SystemRandom()
        pool = list(Challenge)
        count = max(1, min(settings.liveness_challenges, len(pool)))
        # Sampled fresh per attempt: the sequence is the nonce. Replaying a
        # recording of a previous successful attempt only works if the same
        # challenges come up in the same order, inside the same short window.
        self.challenges = challenges if challenges is not None else rng.sample(pool, count)
        self.passive = PassiveDetector(static_frame_limit=settings.static_frame_limit)
        self.state = LivenessState.CALIBRATING
        self.index = 0
        self.failure: str = ""
        self._frontal_frames = 0
        self._recent_yaw: list[float] = []
        self._recent_width: list[float] = []
        self.reference_yaw = 0.0
        self.reference_width = 0.0
        self._deadline = 0.0
        self._calibration_deadline: float | None = None
        self._pending_reference = False
        self.started_at = time.monotonic()
        self.pad_worst = 1.0

    @property
    def current(self) -> Challenge | None:
        if self.index < len(self.challenges):
            return self.challenges[self.index]
        return None

    def submit(self, frame: np.ndarray, face, *, now: float | None = None,
               crop: np.ndarray | None = None) -> Progress:
        now = time.monotonic() if now is None else now
        pad = self.passive.update(frame, face, crop)
        self.pad_worst = min(self.pad_worst, pad.score)
        self._remember(face)

        if pad.suspicious and pad.score < self.settings.min_pad_score:
            return self._fail(pad.flags[0], pad)

        if self.state is LivenessState.CALIBRATING:
            return self._calibrate(face, now, pad)
        if self.state is LivenessState.RUNNING:
            return self._run(face, now, pad)
        return Progress(self.state, "", self.failure, self.index, len(self.challenges), pad)

    # -- internals ---------------------------------------------------------

    def _remember(self, face) -> None:
        self._recent_yaw.append(face.yaw_ratio)
        self._recent_width.append(float(face.width))
        self._recent_yaw = self._recent_yaw[-self.REFERENCE_WINDOW :]
        self._recent_width = self._recent_width[-self.REFERENCE_WINDOW :]

    def _snapshot_reference(self) -> None:
        """Fix the pose each challenge is measured against.

        Taken again before every challenge rather than once at the start. If
        it were only taken at the start, someone who happened to be looking
        sideways during calibration would be asked to turn from a pose they
        were already in -- an instruction they cannot satisfy, and a rejection
        they cannot understand.
        """
        self.reference_yaw = float(np.median(self._recent_yaw)) if self._recent_yaw else 0.0
        self.reference_width = (
            float(np.median(self._recent_width)) if self._recent_width else 1.0
        )

    def _calibrate(self, face, now: float, pad: PadReport) -> Progress:
        if self._calibration_deadline is None:
            self._calibration_deadline = now + self.settings.challenge_timeout_s
        # Only count frames where the person is actually facing the camera.
        if abs(face.yaw_ratio) <= self.FRONTAL_LIMIT:
            self._frontal_frames += 1
        else:
            self._frontal_frames = 0

        if self._frontal_frames < self.CALIBRATION_FRAMES:
            if now > self._calibration_deadline:
                return self._fail("could not get a straight-on view of your face", pad)
            return Progress(
                self.state, "Look straight at the camera", "getting a baseline",
                0, len(self.challenges), pad,
            )

        self._snapshot_reference()
        self.state = LivenessState.RUNNING
        self._deadline = now + self.settings.challenge_timeout_s
        return Progress(
            self.state, self.challenges[0].prompt, "", 1, len(self.challenges), pad
        )

    def _run(self, face, now: float, pad: PadReport) -> Progress:
        challenge = self.current
        if challenge is None:  # pragma: no cover - guarded by state machine
            return self._pass(pad)

        # Between challenges, spend a few frames measuring where the person is
        # now. Otherwise the reference for 'turn right' is contaminated by the
        # frames from the 'turn left' they just did.
        if self._pending_reference:
            if len(self._recent_yaw) < self.REFERENCE_WINDOW:
                return Progress(self.state, challenge.prompt, "hold there for a moment",
                                self.index + 1, len(self.challenges), pad)
            self._snapshot_reference()
            self._pending_reference = False
            self._deadline = now + self.settings.challenge_timeout_s

        if now > self._deadline:
            return self._fail(f"timed out on: {challenge.prompt.lower()}", pad)

        if self._satisfied(challenge, face):
            self.index += 1
            if self.index >= len(self.challenges):
                return self._pass(pad)
            self._recent_yaw.clear()
            self._recent_width.clear()
            self._pending_reference = True
            self._deadline = now + self.settings.challenge_timeout_s
            return Progress(
                self.state, self.challenges[self.index].prompt, "good",
                self.index + 1, len(self.challenges), pad,
            )

        return Progress(
            self.state, challenge.prompt, f"{max(0, int(self._deadline - now))}s left",
            self.index + 1, len(self.challenges), pad,
        )

    def _satisfied(self, challenge: Challenge, face) -> bool:
        if challenge in (Challenge.TURN_LEFT, Challenge.TURN_RIGHT):
            delta = face.yaw_ratio - self.reference_yaw
            sign = _YAW_SIGN[challenge.value] * (-1.0 if self.invert_yaw else 1.0)
            return (delta * sign) > self.YAW_DELTA
        ratio = (float(face.width) - self.reference_width) / max(self.reference_width, 1.0)
        if challenge is Challenge.MOVE_CLOSER:
            return ratio > self.SIZE_DELTA
        return ratio < -self.SIZE_DELTA

    def _pass(self, pad: PadReport) -> Progress:
        self.state = LivenessState.PASSED
        return Progress(
            self.state, "", "liveness confirmed", len(self.challenges),
            len(self.challenges), pad,
        )

    def _fail(self, reason: str, pad: PadReport) -> Progress:
        self.state = LivenessState.FAILED
        self.failure = reason
        return Progress(
            self.state, "", reason, self.index, len(self.challenges), pad
        )

    def summary(self) -> dict:
        """Safe to write to the audit log -- no biometric data in here."""
        return {
            "challenges": [c.value for c in self.challenges],
            "completed": self.index,
            "state": self.state.value,
            "pad_worst": round(self.pad_worst, 3),
            "elapsed_s": round(time.monotonic() - self.started_at, 2),
            "failure": self.failure,
        }
