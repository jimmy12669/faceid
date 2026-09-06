import numpy as np
import pytest

from conftest import FACE_BOX, FakeCamera, FakeFace
from liveness import Challenge, LivenessSession, LivenessState, PassiveDetector


def centred(scale=1.0, yaw=0.0):
    x, y, w, h = FACE_BOX
    nw, nh = int(w * scale), int(h * scale)
    return FakeFace((x + (w - nw) // 2, y + (h - nh) // 2, nw, nh), yaw)


def calibrate(session, frames, face=None):
    face = face or centred()
    for frame in frames[: LivenessSession.CALIBRATION_FRAMES]:
        progress = session.submit(frame, face, now=0.0)
    return progress


@pytest.fixture
def frames():
    camera = FakeCamera()
    return [camera.read() for _ in range(60)]


def test_calibration_needs_a_frontal_view(frames):
    from config import Settings

    settings = Settings()
    session = LivenessSession(settings, challenges=[Challenge.TURN_LEFT])
    # Someone looking sideways is not a baseline; keep asking.
    for i in range(20):
        progress = session.submit(frames[i], centred(yaw=0.4), now=1.0)
    assert session.state is LivenessState.CALIBRATING
    assert progress.prompt == "Look straight at the camera"
    progress = session.submit(frames[21], centred(yaw=0.4),
                              now=settings.challenge_timeout_s + 2)
    assert session.state is LivenessState.FAILED
    assert "straight-on" in progress.message


def test_reference_pose_is_retaken_before_each_challenge(frames):
    """Turning from wherever you are, not from where you were 10 seconds ago."""
    from config import Settings

    session = LivenessSession(
        Settings(), challenges=[Challenge.TURN_LEFT, Challenge.TURN_RIGHT]
    )
    calibrate(session, frames)
    for i in range(12):
        session.submit(frames[10 + i], centred(yaw=0.3), now=1.0)
    assert session.index == 1
    # Now settled at +0.3. The right turn is measured from here, not from
    # centre, so merely returning towards the middle satisfies it.
    session.submit(frames[30], centred(yaw=0.1), now=2.0)
    assert session.state is LivenessState.PASSED


def test_calibration_then_challenge(frames):
    from config import Settings

    session = LivenessSession(Settings(), challenges=[Challenge.TURN_LEFT])
    progress = calibrate(session, frames)
    assert session.state is LivenessState.RUNNING
    assert progress.prompt == Challenge.TURN_LEFT.prompt

    progress = session.submit(frames[10], centred(yaw=0.3), now=1.0)
    assert session.state is LivenessState.PASSED
    assert progress.done


def test_wrong_direction_does_not_satisfy_the_challenge(frames):
    from config import Settings

    session = LivenessSession(Settings(), challenges=[Challenge.TURN_LEFT])
    calibrate(session, frames)
    session.submit(frames[10], centred(yaw=-0.3), now=1.0)
    assert session.state is LivenessState.RUNNING


def test_invert_yaw_flips_the_expected_direction(frames):
    from config import Settings

    session = LivenessSession(Settings(), challenges=[Challenge.TURN_LEFT], invert_yaw=True)
    calibrate(session, frames)
    session.submit(frames[10], centred(yaw=-0.3), now=1.0)
    assert session.state is LivenessState.PASSED


def test_proximity_challenges(frames):
    from config import Settings

    session = LivenessSession(Settings(), challenges=[Challenge.MOVE_CLOSER])
    calibrate(session, frames)
    session.submit(frames[10], centred(scale=0.9), now=1.0)
    assert session.state is LivenessState.RUNNING
    session.submit(frames[11], centred(scale=1.4), now=1.5)
    assert session.state is LivenessState.PASSED


def test_challenge_times_out(frames):
    from config import Settings

    settings = Settings()
    session = LivenessSession(settings, challenges=[Challenge.TURN_RIGHT])
    calibrate(session, frames)
    progress = session.submit(frames[9], centred(), now=settings.challenge_timeout_s + 1)
    assert session.state is LivenessState.FAILED
    assert "timed out" in progress.message


def test_a_still_image_is_flagged():
    camera = FakeCamera()
    frame = camera.read()
    detector = PassiveDetector(static_frame_limit=5)
    face = centred()
    for _ in range(10):
        report = detector.update(frame, face)
    assert report.score < 0.6
    assert any("frozen" in flag or "replay" in flag for flag in report.flags)


def test_a_moving_face_against_a_still_room_looks_live(frames):
    detector = PassiveDetector()
    camera = FakeCamera()
    report = None
    for i in range(20):
        frame = camera.read()
        report = detector.update(frame, camera.face())
    assert report.score > 0.7
    assert report.flags == []


def test_scene_moving_with_the_face_is_flagged():
    """A photo or phone held up drags the background along with it."""
    import cv2

    camera = FakeCamera()
    base = camera.read()
    detector = PassiveDetector()
    report = None
    for i in range(20):
        shift = np.float32([[1, 0, 5 * (i % 3)], [0, 1, 2 * (i % 3)]])
        frame = cv2.warpAffine(base, shift, (base.shape[1], base.shape[0]))
        report = detector.update(frame, centred(yaw=0.01 * i))
    assert report.components["context"] < 0.5


def test_challenges_are_chosen_freshly_each_attempt():
    from config import Settings

    settings = Settings()
    seen = {tuple(LivenessSession(settings).challenges) for _ in range(40)}
    # If the sequence were fixed, a recording of one successful attempt would
    # replay forever.
    assert len(seen) > 1


def test_summary_contains_no_biometric_data(frames):
    from config import Settings

    session = LivenessSession(Settings(), challenges=[Challenge.TURN_LEFT])
    calibrate(session, frames)
    session.submit(frames[10], centred(yaw=0.3), now=1.0)
    summary = session.summary()
    assert set(summary) == {"challenges", "completed", "state", "pad_worst", "elapsed_s", "failure"}
    assert summary["state"] == "passed"
