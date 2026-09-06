"""Authentication policy -- the part that decides yes or no.

Recognition is a similarity score. Authentication is a decision, and a decision
needs rules around it: who may enrol, how many tries before we stop answering,
what a match on its own is worth, what gets written down afterwards. That's
what lives here.

The flow for a verification:

    lockout check -> live capture (quality gates + liveness challenge)
                  -> biometric match -> second factor -> session

Any step can say no. Only the last one issues a token.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

import biometrics
import totp as totp_mod
from biometrics import NoFaceFound, QualityRejected
from database import Store, User
from liveness import LivenessSession, LivenessState, Progress

#: probes collected during one attempt, before even spacing is applied
MAX_SAMPLES = 4

# What the user is told when the identity decision itself fails. Deliberately
# the same string whether the face was unknown, the account was disabled, or
# the second factor was wrong -- an attacker shouldn't learn which.
GENERIC_FAILURE = "Authentication failed."


def _mean_unit(vectors: list[np.ndarray]) -> np.ndarray:
    mean = np.mean(np.stack(vectors), axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm else mean


class FrameSource(Protocol):
    def read(self) -> np.ndarray | None: ...


ProgressHook = Callable[["CaptureUpdate"], None]
SecondFactorProvider = Callable[[str], str | None]


@dataclass
class CaptureUpdate:
    """Everything the UI needs to draw one frame of feedback."""

    frame: np.ndarray
    face: object | None
    progress: Progress | None
    hint: str = ""
    samples: int = 0
    needed: int = 0


@dataclass
class Capture:
    embeddings: list[np.ndarray]
    quality: list[biometrics.Quality]
    liveness: dict
    frames_seen: int

    @property
    def mean_quality(self) -> float:
        return round(sum(q.score() for q in self.quality) / len(self.quality), 4) if self.quality else 0.0


@dataclass
class AuthResult:
    ok: bool
    message: str
    reason: str = ""
    user: User | None = None
    session_token: str | None = None
    similarity: float = 0.0
    detail: dict = field(default_factory=dict)


class CaptureFailed(Exception):
    """Couldn't get a usable live sample. Safe to explain to the user."""


class Authenticator:
    def __init__(self, store: Store, engine: biometrics.FaceEngine, settings, *,
                 actor: str = "cli", invert_yaw: bool = False):
        self.store = store
        self.engine = engine
        self.settings = settings
        self.actor = actor
        self.invert_yaw = invert_yaw

    # -- enrolment ---------------------------------------------------------

    def enroll(self, name: str, source: FrameSource, *, on_progress: ProgressHook | None = None,
               force: bool = False, second_factor: bool = True) -> AuthResult:
        """Register a new face.

        The prototype enrolled whoever happened to be in frame first, silently,
        and re-enrolled on every capture -- so the last person to press a key
        owned the account. Enrolment here is an explicit, named, audited act
        that refuses to overwrite an existing record unless asked twice.
        """
        existing = self.store.find_user(name)
        if existing is not None and not force:
            return AuthResult(
                False,
                f"{name} is already enrolled. Re-enrol with --force if that's intended.",
                reason="already_enrolled",
            )

        capture = self._capture(
            source,
            on_progress=on_progress,
            samples_wanted=self.settings.enroll_samples,
            require_liveness=True,
        )
        template = biometrics.build_template(
            capture.embeddings, max_similarity=self.settings.max_sample_similarity
        )
        cohesion = biometrics.cohesion(capture.embeddings)

        # Don't let one person quietly enrol twice under two names -- that
        # breaks accountability, and 1:N matching gets ambiguous fast.
        clash = self._closest_existing(template, skip_uid=existing.uid if existing else None)
        if clash is not None:
            return AuthResult(
                False,
                f"That face is already enrolled as {clash!r}.",
                reason="duplicate_face",
            )

        blob = biometrics.serialise_template(
            template, engine=self.engine.id, samples=len(capture.embeddings)
        )
        if existing is not None:
            self.store.replace_template(
                existing.uid, engine=self.engine.id, template=blob,
                sample_count=len(capture.embeddings), quality=capture.mean_quality,
            )
            user = self.store.get_user(existing.uid)
            self.store.revoke_sessions(user.uid)  # old sessions don't survive re-enrolment
            action = "user.reenrolled"
        else:
            user = self.store.create_user(
                name, engine=self.engine.id, template=blob,
                sample_count=len(capture.embeddings), quality=capture.mean_quality,
            )
            action = "user.enrolled"

        detail: dict = {
            "samples": len(capture.embeddings),
            "quality": capture.mean_quality,
            "cohesion": cohesion,
            "liveness": capture.liveness,
        }
        secret = None
        codes: list[str] = []
        if second_factor:
            secret = totp_mod.generate_secret()
            self.store.set_second_factor(user.uid, secret)
            codes = totp_mod.generate_recovery_codes()
            # Stored normalised, so the dashes people copy across are cosmetic.
            self.store.store_recovery_codes(
                user.uid, [totp_mod.normalise_recovery_code(c) for c in codes]
            )
            detail["second_factor"] = "totp"

        self.store.audit(action, actor=self.actor, subject=user.name, **detail)
        return AuthResult(
            True,
            f"Enrolled {user.name} from {len(capture.embeddings)} samples.",
            reason="enrolled",
            user=user,
            detail={
                "totp_secret": secret,
                "totp_uri": totp_mod.provisioning_uri(secret, user.name) if secret else None,
                "recovery_codes": codes,
                "cohesion": cohesion,
                "quality": capture.mean_quality,
            },
        )

    def _closest_existing(self, template: np.ndarray, *, skip_uid: str | None) -> str | None:
        for user, blob in self.store.iter_templates(active_only=False):
            if user.uid == skip_uid:
                continue
            vector, _ = biometrics.deserialise_template(blob)
            if biometrics.cosine(template, vector) >= self.settings.match_threshold:
                return user.name
        return None

    # -- verification ------------------------------------------------------

    def authenticate(self, source: FrameSource, *, claimed_name: str | None = None,
                     on_progress: ProgressHook | None = None,
                     second_factor: SecondFactorProvider | None = None) -> AuthResult:
        started = time.monotonic()
        subject = (claimed_name or "*anonymous*").strip().lower()

        lock = self.store.get_lockout(subject)
        if lock.locked:
            self.store.record_attempt(subject, success=False, reason="locked_out")
            self.store.audit("auth.blocked", actor=self.actor, subject=subject,
                             seconds_left=lock.seconds_left, failures=lock.failures)
            return self._pace(
                started,
                AuthResult(
                    False,
                    f"Too many failed attempts. Try again in {lock.seconds_left}s.",
                    reason="locked_out",
                ),
            )

        try:
            capture = self._capture(
                source, on_progress=on_progress, samples_wanted=min(3, MAX_SAMPLES),
                require_liveness=True,
            )
        except CaptureFailed as exc:
            # Interaction problems (no face, gave up, liveness) are explained
            # plainly -- they leak nothing about who is enrolled.
            self._fail(subject, reason=f"capture:{exc}")
            return self._pace(started, AuthResult(False, str(exc), reason="capture_failed"))

        # One embedding from the first good frame would leave the match
        # unbound from the liveness challenges: show a photo, then step in
        # front of the camera yourself to do the head turns. Instead the probes
        # are spread across the attempt and every one of them has to match.
        probes = capture.embeddings
        probe = _mean_unit(probes)
        if claimed_name:
            user = self.store.find_user(claimed_name)
            if user is None or not user.active:
                # Run the comparison anyway against a throwaway vector so an
                # unknown name doesn't return faster than a known one.
                biometrics.cosine(probe, np.zeros_like(probe))
                self._fail(subject, reason="unknown_or_disabled_user")
                return self._pace(started, AuthResult(False, GENERIC_FAILURE, reason="no_match"))
            matched_template, _ = biometrics.deserialise_template(
                self.store.load_template(user.uid)
            )
            match = biometrics.verify_against(
                probe, matched_template, threshold=self.settings.match_threshold
            )
        else:
            candidates = []
            templates: dict[str, np.ndarray] = {}
            for candidate, blob in self.store.iter_templates(active_only=True):
                vector, _ = biometrics.deserialise_template(blob)
                templates[candidate.uid] = vector
                candidates.append((candidate.uid, candidate.name, vector))
            match = biometrics.identify(
                probe, candidates,
                threshold=self.settings.match_threshold,
                margin=self.settings.identify_margin,
            )
            user = self.store.get_user(match.uid) if match.uid else None
            matched_template = templates.get(match.uid) if match.uid else None

        if not match.accepted or user is None:
            self._fail(subject, reason=match.reason, similarity=round(match.similarity, 4))
            return self._pace(started, AuthResult(False, GENERIC_FAILURE, reason="no_match",
                                                  similarity=match.similarity))

        # Every probe taken during the attempt has to be the same person as the
        # one who passed the challenges.
        weakest = min(biometrics.cosine(p, matched_template) for p in probes)
        if weakest < self.settings.match_threshold:
            self._fail(subject, reason="inconsistent_probe",
                       weakest=round(weakest, 4), uid=user.uid)
            return self._pace(started, AuthResult(False, GENERIC_FAILURE,
                                                  reason="inconsistent_probe",
                                                  similarity=match.similarity))

        factors = ["face"]
        needs_2fa = self.store.get_second_factor(user.uid) is not None
        if self.settings.require_second_factor and not needs_2fa:
            self._fail(subject, reason="second_factor_missing", uid=user.uid)
            return self._pace(started, AuthResult(
                False,
                f"{user.name} has no second factor enrolled and policy requires one.",
                reason="second_factor_missing",
            ))
        if needs_2fa:
            ok, detail = self._check_second_factor(user, second_factor)
            if not ok:
                self._fail(subject, reason=detail, uid=user.uid)
                return self._pace(started, AuthResult(False, GENERIC_FAILURE, reason=detail))
            factors.append(detail)

        token = self.store.create_session(
            user.uid, ttl_s=self.settings.session_ttl_s, factors=factors
        )
        self.store.clear_failures(subject)
        self.store.clear_failures(user.name.lower())
        self.store.record_attempt(subject, success=True, reason="ok", uid=user.uid)
        self.store.audit(
            "auth.success", actor=self.actor, subject=user.name,
            similarity=round(match.similarity, 4),
            runner_up=round(match.runner_up, 4),
            factors=factors, liveness=capture.liveness,
            mode="verify" if claimed_name else "identify",
        )
        return self._pace(started, AuthResult(
            True, f"Welcome, {user.name}.", reason="ok", user=user,
            session_token=token, similarity=match.similarity,
            detail={"expires_in": self.settings.session_ttl_s, "factors": factors},
        ))

    def _check_second_factor(self, user: User, provider: SecondFactorProvider | None):
        pair = self.store.get_second_factor(user.uid)
        if pair is None:
            return False, "second_factor_missing"
        if provider is None:
            return False, "second_factor_not_supplied"
        secret, last_counter = pair
        code = provider(f"Authenticator code for {user.name}")
        if not code:
            return False, "second_factor_not_supplied"
        code = code.strip()

        if "-" in code or len(totp_mod.normalise_recovery_code(code)) == 16:
            if self.store.spend_recovery_code(user.uid, totp_mod.normalise_recovery_code(code)):
                left = self.store.recovery_codes_left(user.uid)
                self.store.audit("second_factor.recovery_used", actor=self.actor,
                                 subject=user.name, codes_left=left)
                return True, "recovery_code"
            return False, "second_factor_bad_recovery_code"

        counter = totp_mod.verify(secret, code, last_counter=last_counter)
        if counter is None:
            return False, "second_factor_bad_code"
        # Burn the counter so the same code can't be used twice.
        if not self.store.spend_totp_counter(user.uid, counter):
            return False, "second_factor_replayed"
        return True, "totp"

    # -- sessions ----------------------------------------------------------

    def whoami(self, token: str) -> User | None:
        return self.store.validate_session(token)

    def logout(self, token: str) -> bool:
        user = self.store.validate_session(token)
        revoked = self.store.revoke_session(token)
        if revoked:
            self.store.audit("session.revoked", actor=self.actor,
                             subject=user.name if user else "-")
        return revoked

    # -- capture -----------------------------------------------------------

    def _capture(self, source: FrameSource, *, on_progress: ProgressHook | None,
                 samples_wanted: int, require_liveness: bool) -> Capture:
        """Drive the camera until we have good samples and liveness passed."""
        session = LivenessSession(self.settings, invert_yaw=self.invert_yaw)
        budget = self.settings.challenge_timeout_s * (len(session.challenges) + 1) + 10.0
        deadline = time.monotonic() + budget

        embeddings: list[np.ndarray] = []
        qualities: list[biometrics.Quality] = []
        last_sample = 0.0
        frames = 0
        progress: Progress | None = None
        ceiling = max(samples_wanted, MAX_SAMPLES)

        while time.monotonic() < deadline:
            frame = source.read()
            if frame is None:
                break
            frames += 1
            hint = ""
            face = None
            try:
                face = self.engine.detect_single(frame)
            except NoFaceFound:
                hint = "No face in view -- centre yourself in the frame"
            except QualityRejected as exc:
                hint = str(exc)

            if face is not None:
                quality = biometrics.assess(frame, face, self.settings)
                if not quality.ok:
                    hint = quality.reasons[0]
                else:
                    progress = session.submit(frame, face)
                    now = time.monotonic()
                    # Space samples out in time so a single frozen frame can't
                    # fill the set, and keep sampling for the whole session so
                    # the samples span the liveness challenges rather than
                    # clustering in the first second.
                    if len(embeddings) < ceiling and (now - last_sample) > 0.4:
                        embeddings.append(self.engine.embed(frame, face))
                        qualities.append(quality)
                        last_sample = now

            if on_progress is not None:
                on_progress(CaptureUpdate(
                    frame=frame, face=face, progress=progress, hint=hint,
                    samples=len(embeddings), needed=samples_wanted,
                ))

            if progress is not None and progress.state is LivenessState.FAILED:
                raise CaptureFailed(f"Liveness check failed: {progress.message}")
            if (
                len(embeddings) >= samples_wanted
                and (not require_liveness or session.state is LivenessState.PASSED)
            ):
                return self._finish(embeddings, qualities, session, frames, samples_wanted)

        if session.state is LivenessState.PASSED and len(embeddings) >= samples_wanted:
            return self._finish(embeddings, qualities, session, frames, samples_wanted)
        if not embeddings:
            raise CaptureFailed("Timed out without a usable view of your face.")
        raise CaptureFailed("Timed out before the movement checks were completed.")

    @staticmethod
    def _finish(embeddings, qualities, session, frames, wanted) -> Capture:
        """Keep an evenly spaced subset, so the samples cover the whole attempt."""
        if len(embeddings) > wanted:
            step = len(embeddings) / wanted
            picks = [min(int(i * step), len(embeddings) - 1) for i in range(wanted)]
            embeddings = [embeddings[i] for i in picks]
            qualities = [qualities[i] for i in picks]
        return Capture(embeddings, qualities, session.summary(), frames)

    # -- bookkeeping -------------------------------------------------------

    def _fail(self, subject: str, *, reason: str, **detail) -> None:
        self.store.record_attempt(subject, success=False, reason=reason)
        lock = self.store.register_failure(
            subject,
            threshold=self.settings.max_failed_attempts,
            base_s=self.settings.lockout_base_s,
            max_s=self.settings.lockout_max_s,
        )
        self.store.audit("auth.failure", actor=self.actor, subject=subject,
                         reason=reason, failures=lock.failures,
                         locked_for=lock.seconds_left, **detail)

    def _pace(self, started: float, result: AuthResult) -> AuthResult:
        """Give every failure the same minimum duration.

        Without this, 'no such user' returns in milliseconds while a real
        comparison takes longer, and the response time becomes an oracle for
        who is enrolled.
        """
        if not result.ok:
            floor = self.settings.min_response_ms / 1000.0
            remaining = floor - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
        return result
