"""Runtime settings.

Everything here can be overridden with FACEID_* environment variables so the
same code runs on a workstation and on a locked-down kiosk without edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"FACEID_{name}", default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(f"FACEID_{name}")
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"FACEID_{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(f"FACEID_{name}")
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"FACEID_{name} must be a number, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"FACEID_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- where state lives -------------------------------------------------
    home: Path = field(default_factory=lambda: Path(_env_str("HOME", ".faceid")).expanduser())

    # --- recognition -------------------------------------------------------
    # SFace produces L2-normalised 128-d embeddings; OpenCV's published
    # operating point is a cosine similarity of 0.363. We run tighter by
    # default: fewer false accepts, at the cost of the occasional retry.
    match_threshold: float = _env_float("MATCH_THRESHOLD", 0.40)
    # In 1:N mode the winner must also beat the runner-up by this margin,
    # otherwise the claim is too close to call and we reject.
    identify_margin: float = _env_float("IDENTIFY_MARGIN", 0.06)
    detector_score: float = _env_float("DETECTOR_SCORE", 0.85)

    # --- sample quality gates ---------------------------------------------
    enroll_samples: int = _env_int("ENROLL_SAMPLES", 5)
    min_face_px: int = _env_int("MIN_FACE_PX", 96)
    min_sharpness: float = _env_float("MIN_SHARPNESS", 45.0)
    min_brightness: float = _env_float("MIN_BRIGHTNESS", 40.0)
    max_brightness: float = _env_float("MAX_BRIGHTNESS", 215.0)
    max_roll_deg: float = _env_float("MAX_ROLL_DEG", 20.0)
    # Two enrolment samples that are *this* alike came from one frozen frame.
    max_sample_similarity: float = _env_float("MAX_SAMPLE_SIMILARITY", 0.995)

    # --- liveness ----------------------------------------------------------
    liveness_challenges: int = _env_int("LIVENESS_CHALLENGES", 2)
    challenge_timeout_s: float = _env_float("CHALLENGE_TIMEOUT", 12.0)
    min_pad_score: float = _env_float("MIN_PAD_SCORE", 0.55)
    static_frame_limit: int = _env_int("STATIC_FRAME_LIMIT", 12)

    # --- policy ------------------------------------------------------------
    max_failed_attempts: int = _env_int("MAX_FAILED_ATTEMPTS", 5)
    lockout_base_s: int = _env_int("LOCKOUT_BASE", 30)
    lockout_max_s: int = _env_int("LOCKOUT_MAX", 3600)
    session_ttl_s: int = _env_int("SESSION_TTL", 900)
    require_second_factor: bool = _env_bool("REQUIRE_SECOND_FACTOR", True)
    # Floor on how long a failed verification takes, so an attacker can't read
    # the outcome off the clock.
    min_response_ms: int = _env_int("MIN_RESPONSE_MS", 350)

    # --- key derivation ----------------------------------------------------
    scrypt_n: int = _env_int("SCRYPT_N", 1 << 15)
    scrypt_r: int = _env_int("SCRYPT_R", 8)
    scrypt_p: int = _env_int("SCRYPT_P", 1)

    @property
    def db_path(self) -> Path:
        return self.home / "faceid.db"

    @property
    def keystore_path(self) -> Path:
        return self.home / "keystore.json"

    @property
    def models_dir(self) -> Path:
        return Path(_env_str("MODELS_DIR", str(self.home / "models"))).expanduser()

    @property
    def audit_mirror_path(self) -> Path:
        """Append-only copy of the audit log, for shipping off the box."""
        return self.home / "audit.log"

    def ensure_home(self) -> Path:
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        # mkdir's mode is masked by umask, so pin it explicitly.
        os.chmod(self.home, 0o700)
        return self.home


def load() -> Settings:
    return Settings()
