import os
from pathlib import Path

import numpy as np
import pytest

import biometrics
import config
from conftest import FACE_BOX, FakeCamera, FakeFace


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def jitter(base, scale, rng):
    return unit(np.asarray(base) + rng.normal(0, scale, len(base)).astype(np.float32))


@pytest.fixture
def rng():
    return np.random.default_rng(11)


@pytest.fixture
def person(rng):
    base = np.zeros(128, dtype=np.float32)
    base[:8] = 1.0
    return unit(base)


def test_cosine_of_a_vector_with_itself(person):
    assert biometrics.cosine(person, person) == pytest.approx(1.0, abs=1e-5)


def test_template_averaging_beats_a_single_sample(person, rng):
    samples = [jitter(person, 0.35, rng) for _ in range(6)]
    template = biometrics.build_template(samples)
    probe = jitter(person, 0.35, rng)
    # Averaging cancels per-frame noise, so the probe sits closer to the
    # template than to any one enrolment frame.
    assert biometrics.cosine(probe, template) > np.mean(
        [biometrics.cosine(probe, s) for s in samples]
    )


def test_identical_samples_are_refused(person):
    with pytest.raises(biometrics.QualityRejected, match="same still image"):
        biometrics.build_template([person, person.copy(), person.copy()])


def test_one_sample_is_not_a_template(person):
    with pytest.raises(biometrics.BiometricError):
        biometrics.build_template([person])


def test_template_serialisation_roundtrip(person):
    blob = biometrics.serialise_template(person, engine="sface", samples=5)
    vector, header = biometrics.deserialise_template(blob)
    assert np.allclose(vector, person)
    assert header["engine"] == "sface" and header["samples"] == 5 and header["dim"] == 128


def test_truncated_template_is_rejected(person):
    blob = biometrics.serialise_template(person, engine="sface", samples=5)
    with pytest.raises(biometrics.BiometricError, match="truncated"):
        biometrics.deserialise_template(blob[:-8])


def test_garbage_is_not_a_template():
    with pytest.raises(biometrics.BiometricError):
        biometrics.deserialise_template(b"just some bytes")


def test_identification_picks_the_right_person(rng):
    people = {}
    for i, name in enumerate(["alice", "bob", "carol"]):
        base = np.zeros(128, dtype=np.float32)
        base[i * 8 : i * 8 + 8] = 1.0
        people[name] = unit(base)
    candidates = [(name, name, vector) for name, vector in people.items()]
    probe = jitter(people["bob"], 0.15, rng)
    result = biometrics.identify(probe, candidates, threshold=0.4, margin=0.06)
    assert result.accepted and result.name == "bob"


def test_a_stranger_is_rejected(rng):
    enrolled = [("alice", "alice", unit(np.eye(128)[0]))]
    stranger = unit(np.eye(128)[64])
    result = biometrics.identify(stranger, enrolled, threshold=0.4, margin=0.06)
    assert not result.accepted and result.reason == "no enrolment matched"


def test_ambiguous_match_is_refused():
    """Two enrolments equally close is exactly when not to guess."""
    a = unit(np.eye(128)[0])
    b = unit(np.eye(128)[1])
    probe = unit(a + b)  # dead centre between them
    result = biometrics.identify(
        probe, [("a", "a", a), ("b", "b", b)], threshold=0.4, margin=0.06
    )
    assert not result.accepted and "ambiguous" in result.reason


def test_empty_gallery_matches_nobody():
    result = biometrics.identify(unit(np.eye(128)[0]), [], threshold=0.4, margin=0.06)
    assert not result.accepted


def test_verify_against_a_claim(person, rng):
    assert biometrics.verify_against(
        jitter(person, 0.2, rng), person, threshold=0.4
    ).accepted
    assert not biometrics.verify_against(
        unit(np.eye(128)[100]), person, threshold=0.4
    ).accepted


# -- quality gates ---------------------------------------------------------


def frame_and_face(*, blur=0, brightness=1.0, size=1.0):
    import cv2

    camera = FakeCamera()
    frame = camera.read()
    if blur:
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)
    if brightness != 1.0:
        frame = np.clip(frame.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    x, y, w, h = FACE_BOX
    nw, nh = int(w * size), int(h * size)
    return frame, FakeFace((x, y, nw, nh), 0.0)


def test_a_good_frame_passes():
    settings = config.Settings()
    frame, face = frame_and_face()
    quality = biometrics.assess(frame, face, settings)
    assert quality.ok, quality.reasons
    assert 0.0 < quality.score() <= 1.0


def test_a_blurry_frame_is_rejected():
    settings = config.Settings()
    frame, face = frame_and_face(blur=31)
    reasons = biometrics.assess(frame, face, settings).reasons
    assert any("blurry" in r for r in reasons)


def test_a_dark_frame_is_rejected():
    settings = config.Settings()
    frame, face = frame_and_face(brightness=0.15)
    assert any("dark" in r for r in biometrics.assess(frame, face, settings).reasons)


def test_a_tiny_face_is_rejected():
    settings = config.Settings()
    frame, face = frame_and_face(size=0.3)
    assert any("too small" in r for r in biometrics.assess(frame, face, settings).reasons)


def test_quality_messages_tell_you_what_to_do():
    settings = config.Settings()
    frame, face = frame_and_face(size=0.3)
    # A rejection the user cannot act on is a support ticket.
    assert "move closer" in biometrics.assess(frame, face, settings).reasons[0]


# -- the real models, when they are present --------------------------------

models = config.Settings().models_dir
real_models = pytest.mark.skipif(
    not (models / biometrics.RECOGNISER_MODEL).exists(),
    reason="models not downloaded; run python fetch_models.py",
)


@real_models
def test_real_engine_loads_and_finds_no_face_in_noise():
    """Cheap end-to-end check that the pinned models load and behave."""
    settings = config.Settings()
    engine = biometrics.FaceEngine(settings.models_dir)
    noise = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    assert engine.detect(noise) == []
    with pytest.raises(biometrics.NoFaceFound):
        engine.detect_single(noise)


@real_models
@pytest.mark.skipif(
    not os.environ.get("FACEID_TEST_FACES"),
    reason="set FACEID_TEST_FACES to a directory of face photos to run this",
)
def test_real_engine_separates_people():
    """The claim the whole rewrite rests on, measured on real photographs.

    No face images are committed to this repository -- publishing other
    people's biometrics to make a test suite green is not a trade we make.
    Point FACEID_TEST_FACES at a folder of portraits named <person>_<n>.jpg
    and this checks that same-person pairs clear the threshold and
    different-person pairs do not.
    """
    import re
    from collections import defaultdict

    import cv2

    settings = config.Settings()
    engine = biometrics.FaceEngine(settings.models_dir)
    groups: dict[str, list] = defaultdict(list)
    for path in sorted(Path(os.environ["FACEID_TEST_FACES"]).glob("*")):
        image = cv2.imread(str(path))
        if image is None:
            continue
        faces = engine.detect(image)
        if not faces:
            continue
        person = re.split(r"[_\-.]", path.name)[0]
        groups[person].append(engine.embed(image, faces[0]))

    same = [
        biometrics.cosine(a, b)
        for vectors in groups.values()
        for i, a in enumerate(vectors)
        for b in vectors[i + 1 :]
    ]
    names = list(groups)
    different = [
        biometrics.cosine(a, b)
        for i, first in enumerate(names)
        for second in names[i + 1 :]
        for a in groups[first]
        for b in groups[second]
    ]
    if same:
        assert min(same) > settings.match_threshold, "a genuine pair fell below threshold"
    if different:
        assert max(different) < settings.match_threshold, "an impostor pair cleared threshold"
