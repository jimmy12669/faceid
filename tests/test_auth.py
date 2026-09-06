"""End-to-end policy tests, driven by the fake camera.

These are the ones that matter: they exercise the same code path the webcam
does, so the enrolment and verification flows are covered without hardware.
"""

import time

import pytest

import auth
import biometrics
import totp as totp_mod
from conftest import FakeCamera, FakeEngine, obedient


@pytest.fixture
def authenticator(store, settings, camera, engine):
    return auth.Authenticator(store, engine, settings, actor="test")


def code_for(store, user):
    secret, _ = store.get_second_factor(user.uid)
    return lambda prompt: totp_mod.totp(secret)


def enrol(authenticator, name="alice", **kwargs):
    """Enrol through the real pipeline, with the fake user following prompts."""
    camera = authenticator.engine._camera
    kwargs.setdefault("on_progress", obedient(camera))
    result = authenticator.enroll(name, camera, **kwargs)
    assert result.ok, result.message
    return result


def try_enrol(authenticator, name, **kwargs):
    camera = authenticator.engine._camera
    kwargs.setdefault("on_progress", obedient(camera))
    return authenticator.enroll(name, camera, **kwargs)


def verify(authenticator, code="000000", **kwargs):
    camera = authenticator.engine._camera
    kwargs.setdefault("on_progress", obedient(camera))
    kwargs.setdefault("second_factor", code if callable(code) else (lambda prompt: code))
    return authenticator.authenticate(camera, **kwargs)


# -- enrolment -------------------------------------------------------------


def test_enrolment_creates_a_user_with_a_second_factor(authenticator, store):
    result = enrol(authenticator)
    user = store.find_user("alice")
    assert user is not None and user.has_second_factor
    assert user.sample_count == authenticator.settings.enroll_samples
    assert len(result.detail["recovery_codes"]) == 10
    assert result.detail["totp_uri"].startswith("otpauth://totp/")


def test_enrolment_is_audited(authenticator, store):
    enrol(authenticator)
    assert "user.enrolled" in [row["action"] for row in store.read_audit()]
    assert store.verify_audit_chain()[0]


def test_the_stored_template_is_not_the_raw_samples(authenticator, store, settings):
    enrol(authenticator)
    user = store.find_user("alice")
    vector, header = biometrics.deserialise_template(store.load_template(user.uid))
    assert header["engine"] == "fake-engine-v1"
    assert vector.shape == (128,)
    assert b"FTPL" not in settings.db_path.read_bytes()  # even the header is sealed


def test_re_enrolment_needs_force(authenticator, store):
    """The prototype let the last person to press a key take over the account."""
    enrol(authenticator)
    again = try_enrol(authenticator, "alice")
    assert not again.ok and again.reason == "already_enrolled"


def test_forced_re_enrolment_revokes_old_sessions(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    token = store.create_session(user.uid, ttl_s=600, factors=["face"])
    assert try_enrol(authenticator, "alice", force=True).ok
    assert store.validate_session(token) is None


def test_the_same_face_cannot_enrol_twice_under_two_names(authenticator, store):
    enrol(authenticator, "alice")
    result = try_enrol(authenticator, "alice-again")
    assert not result.ok and result.reason == "duplicate_face"


def test_a_frozen_camera_cannot_enrol(store, settings):
    """A single repeated frame is a photo, and photos do not enrol."""

    class FrozenCamera(FakeCamera):
        def read(self):
            frame = super().read()
            self.i = 0  # same frame, same pose, forever
            return frame

    camera = FrozenCamera()
    engine = FakeEngine().bind(camera)
    settings.challenge_timeout_s = 0.4
    authenticator = auth.Authenticator(store, engine, settings, actor="test")
    with pytest.raises(auth.CaptureFailed):
        authenticator.enroll("mallory", camera, on_progress=obedient(camera))


def test_an_uncooperative_subject_does_not_enrol(store, settings, camera, engine):
    """No response to the movement prompts means no enrolment."""
    settings.challenge_timeout_s = 0.3
    authenticator = auth.Authenticator(store, engine, settings, actor="test")
    with pytest.raises(auth.CaptureFailed):
        authenticator.enroll("mallory", camera)  # note: no obedient() hook


# -- verification ----------------------------------------------------------


def test_verify_the_enrolled_user(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    result = verify(authenticator, code=code_for(store, user))
    assert result.ok
    assert result.user.name == "alice"
    assert result.detail["factors"] == ["face", "totp"]
    assert store.validate_session(result.session_token).name == "alice"


def test_identification_without_a_claimed_name(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    result = verify(authenticator, code=code_for(store, user), claimed_name=None)
    assert result.ok and result.similarity > authenticator.settings.match_threshold


def test_a_different_face_is_rejected(authenticator, store, settings):
    enrol(authenticator)
    intruder_camera = FakeCamera(seed=99)
    intruder = FakeEngine(identity=5).bind(intruder_camera)
    attacker = auth.Authenticator(store, intruder, settings, actor="test")
    result = attacker.authenticate(
        intruder_camera, on_progress=obedient(intruder_camera),
        second_factor=lambda p: "000000",
    )
    assert not result.ok
    assert result.message == auth.GENERIC_FAILURE


def test_a_wrong_second_factor_blocks_a_correct_face(authenticator, store):
    """Face alone is not enough, which is the entire point of the second factor."""
    enrol(authenticator)
    result = verify(authenticator, code="123456")
    assert not result.ok and result.reason == "second_factor_bad_code"


def test_a_totp_code_cannot_be_used_twice(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    secret, _ = store.get_second_factor(user.uid)
    code = totp_mod.totp(secret)
    assert verify(authenticator, code=code).ok
    assert not verify(authenticator, code=code).ok


def test_recovery_code_works_once(authenticator, store):
    result = enrol(authenticator)
    recovery = result.detail["recovery_codes"][0]
    first = verify(authenticator, code=recovery)
    assert first.ok and first.detail["factors"] == ["face", "recovery_code"]
    assert not verify(authenticator, code=recovery).ok


def test_a_disabled_user_cannot_authenticate(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    store.set_status(user.uid, "disabled")
    result = verify(authenticator, code=code_for(store, user), claimed_name="alice")
    assert not result.ok and result.message == auth.GENERIC_FAILURE


def test_unknown_and_known_claims_fail_identically(authenticator, store):
    """The error text must not tell an attacker who is enrolled."""
    enrol(authenticator)
    unknown = verify(authenticator, claimed_name="nobody")
    known = verify(authenticator, claimed_name="alice")
    assert unknown.message == known.message == auth.GENERIC_FAILURE


# -- throttling ------------------------------------------------------------


def test_repeated_failures_trigger_a_lockout(authenticator, store, settings):
    enrol(authenticator)
    for _ in range(settings.max_failed_attempts):
        verify(authenticator, claimed_name="alice")
    blocked = verify(authenticator, claimed_name="alice")
    assert blocked.reason == "locked_out"
    assert "Try again in" in blocked.message


def test_lockout_is_recorded_and_auditable(authenticator, store, settings):
    enrol(authenticator)
    for _ in range(settings.max_failed_attempts + 1):
        verify(authenticator, claimed_name="alice")
    actions = [row["action"] for row in store.read_audit(50)]
    assert "auth.failure" in actions and "auth.blocked" in actions
    assert store.verify_audit_chain()[0]


def test_success_clears_the_failure_count(authenticator, store, settings):
    enrol(authenticator)
    user = store.find_user("alice")
    for _ in range(settings.max_failed_attempts - 1):
        verify(authenticator, claimed_name="alice")
    assert verify(authenticator, code=code_for(store, user), claimed_name="alice").ok
    assert store.get_lockout("alice").failures == 0


def test_failures_are_paced(store, settings, camera, engine):
    """Fast failures leak whether an account exists."""
    settings.min_response_ms = 200
    authenticator = auth.Authenticator(store, engine, settings, actor="test")
    for _ in range(settings.max_failed_attempts + 1):
        store.register_failure(
            "ghost", threshold=settings.max_failed_attempts,
            base_s=settings.lockout_base_s, max_s=settings.lockout_max_s,
        )
    started = time.monotonic()
    verify(authenticator, claimed_name="ghost")
    assert time.monotonic() - started >= 0.2


# -- sessions --------------------------------------------------------------


def test_session_lifecycle(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    token = verify(authenticator, code=code_for(store, user)).session_token
    assert authenticator.whoami(token).name == "alice"
    assert authenticator.logout(token)
    assert authenticator.whoami(token) is None
    assert not authenticator.logout(token)


def test_audit_log_never_contains_a_session_token(authenticator, store):
    enrol(authenticator)
    user = store.find_user("alice")
    result = verify(authenticator, code=code_for(store, user))
    dumped = "".join(row["detail"] for row in store.read_audit(100))
    assert result.session_token not in dumped


def test_a_face_swapped_mid_attempt_is_rejected(store, settings, camera):
    """Photo first, then step in and do the head turns yourself.

    The probes are spread across the attempt and every one of them has to be
    the enrolled person, so switching faces partway through does not work.
    """

    class SwitchingEngine(FakeEngine):
        def __init__(self, switch_after: int):
            super().__init__()
            self.switch_after = switch_after
            self.calls = 0

        def embed(self, frame, face):
            self.calls += 1
            self.identity = 0 if self.calls <= self.switch_after else 6
            return super().embed(frame, face)

    honest = FakeEngine().bind(camera)
    authenticator = auth.Authenticator(store, honest, settings, actor="test")
    enrol(authenticator)
    user = store.find_user("alice")

    impostor = SwitchingEngine(switch_after=1).bind(camera)
    attacker = auth.Authenticator(store, impostor, settings, actor="test")
    result = attacker.authenticate(
        camera, on_progress=obedient(camera), second_factor=code_for(store, user)
    )
    assert not result.ok
    assert result.reason in ("inconsistent_probe", "no_match")
    assert result.message == auth.GENERIC_FAILURE
