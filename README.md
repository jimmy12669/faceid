# faceid

Face authentication for a laptop or a kiosk, built around the assumption that
someone will try to beat it.

This started as a ~100-line OpenCV script: grab a webcam frame, find a face with
a Haar cascade, compare a grayscale histogram against a reference JPEG sitting
in a folder, print "Match found!" if the correlation cleared 0.7. It was a
working demo of face *detection*. It was not authentication, and the gap between
those two things is most of this repository.

## What was wrong with the original, and what replaced it

| The prototype | The problem | Now |
| --- | --- | --- |
| Histogram correlation of a grayscale crop | A histogram describes brightness, not a face. A printed photo passes. A stranger under the same lamp passes. | 128-d [SFace](https://github.com/opencv/opencv_zoo) embeddings from an aligned crop; same-person pairs land near 0.9 cosine, different-person pairs near 0.1 |
| First face seen becomes the reference | Whoever walks past the camera first owns the account | Enrolment is a named, audited command that needs the keystore passphrase |
| Every capture overwrote the reference | Last person to press a key takes over the account | Re-enrolment requires `--force`, and it revokes existing sessions |
| Reference stored as `reference_image.jpg` | The credential is a file anyone can copy or swap | Only encrypted embeddings are stored (AES-256-GCM), bound to the user row so they cannot be moved between accounts |
| No liveness check | A phone showing a photo authenticates | Randomised movement challenges plus passive checks on every frame |
| Unlimited attempts | Free rein to keep trying | Per-subject failure counting with exponential backoff, held in the database |
| Face was the only factor | You cannot rotate your face after a breach | TOTP second factor with single-use recovery codes |
| No record of anything | No way to answer "who got in, and when" | HMAC-chained audit log that detects edits, deletions and truncation |

## Getting started

```sh
pip install -r requirements.txt
python fetch_models.py        # ~39 MB, SHA-256 pinned
python cli.py init            # choose a keystore passphrase
python cli.py enroll alice    # follow the on-screen prompts
python cli.py verify          # or: python cli.py verify alice
```

`python image.py` still opens the old-style camera window, and still answers to
`c` to enrol, `v` to verify and `q` to quit.

Enrolment prints a TOTP secret and ten recovery codes exactly once. Put the
secret in an authenticator app before closing the terminal.

## Commands

```
python cli.py init                  create the keystore and database
python cli.py enroll NAME           register a face (--force to replace one)
python cli.py verify [NAME]         authenticate; omit NAME to search all enrolments
python cli.py whoami                who the saved session belongs to
python cli.py logout                end the saved session
python cli.py users                 list enrolments
python cli.py disable NAME          suspend a user and kill their sessions
python cli.py delete NAME           remove an enrolment (the audit record stays)
python cli.py sessions --purge      list, purge or revoke live sessions
python cli.py audit --verify        recompute the audit chain
python cli.py passphrase            change the keystore passphrase
python cli.py doctor --deep         check the install for weak spots
```

`--headless` swaps the preview window for terminal prompts. `--replay DIR`
reads frames from a folder of images instead of a camera, which is a convenient
way to watch the still-image path get rejected.

## How a verification actually runs

1. **Lockout check.** Too many recent failures for this subject and nothing else
   happens.
2. **Capture.** Frames are gated on quality: face size, sharpness, exposure,
   head tilt, detector confidence, and exactly one face in view. Rejections come
   with something the person can act on ("move closer", "add light").
3. **Liveness.** Two challenges drawn at random per attempt — turn left, turn
   right, lean in, lean back — each measured against the pose you are in when it
   is asked, with a deadline. Meanwhile every frame is checked for a frozen
   image, flat texture, whole-scene movement and missing head micro-motion.
4. **Match.** Cosine similarity against the enrolled template. Probes are taken
   at intervals across the whole attempt, not from one frame, and every one of
   them has to match — otherwise a photo held up at the start and a real face
   for the head turns would sail through. In 1:N mode the winner must also beat
   the runner-up by a margin, or the result is "ambiguous" rather than a guess.
5. **Second factor.** A TOTP code, or a recovery code. Used counters are burned,
   so a code works exactly once.
6. **Session.** A random token, stored keyed-hashed with a TTL, revocable. It
   is written to `$FACEID_HOME/session.token` with mode 600 rather than printed,
   so it does not sit in the terminal scrollback; `--show-token` overrides that.

Failures after step 3 all return the same sentence and take the same minimum
time, so neither the message nor the clock says who is enrolled.

## Configuration

Everything is an environment variable, so a kiosk build differs from a laptop
build by config rather than by patch:

| Variable | Default | Notes |
| --- | --- | --- |
| `FACEID_HOME` | `.faceid` | keystore, database, audit mirror |
| `FACEID_MATCH_THRESHOLD` | `0.40` | OpenCV's published SFace operating point is 0.363; higher is stricter |
| `FACEID_IDENTIFY_MARGIN` | `0.06` | how far the winner must beat the runner-up in 1:N |
| `FACEID_ENROLL_SAMPLES` | `5` | frames averaged into a template |
| `FACEID_LIVENESS_CHALLENGES` | `2` | movement challenges per attempt |
| `FACEID_MIN_PAD_SCORE` | `0.55` | passive spoof-signal floor |
| `FACEID_MAX_FAILED_ATTEMPTS` | `5` | before backoff starts |
| `FACEID_SESSION_TTL` | `900` | seconds |
| `FACEID_REQUIRE_SECOND_FACTOR` | `1` | set to `0` only if you know why |

`python cli.py doctor` flags the settings that weaken the system, along with
file permissions and model digests.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest tests -q
```

98 tests, no camera required. The fake camera in `tests/conftest.py` simulates a
cooperative user: it renders a face against a fixed background and responds to
whatever prompt the liveness session displays, after a human-like reaction
delay. That means enrolment, verification, lockout, replay and session handling
all run through the same code path the webcam does.

No face images are committed here — publishing someone's biometrics to keep a
test suite green is not a trade worth making. Point `FACEID_TEST_FACES` at a
folder of portraits named `person_1.jpg` to run the recognition-accuracy test
against real photographs.

## Layout

```
cli.py           commands, and the only place that prompts for the passphrase
image.py         camera, on-screen prompts, overlay drawing
auth.py          policy: lockout, capture, match, second factor, sessions
biometrics.py    detection, quality gates, templates, matching
liveness.py      challenge state machine and passive spoof signals
database.py      encrypted SQLite store and the audit chain
crypto.py        keystore, AES-GCM, HKDF subkeys, HMAC
totp.py          RFC 6238 and recovery codes
fetch_models.py  pinned model download
```

## Limitations

Read [SECURITY.md](SECURITY.md) before deploying this anywhere that matters. The
short version: the liveness checks are heuristics, not certified presentation-
attack detection, and they will not stop a determined attacker with a good video
of you and a way to inject it into the video device. Face is treated as one
factor here for exactly that reason.
