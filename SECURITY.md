# Threat model and known limits

Written so that anyone deciding whether to use this can see what it actually
defends against, and where it stops. If a claim below is not backed by a test in
`tests/`, it is marked as an expectation rather than a guarantee.

## What is being protected

1. **The enrolment templates.** A face embedding is not a password: it cannot be
   changed, and it links to the person everywhere else their face appears.
2. **The decision.** Whether the system says yes to the person in front of it.
3. **The record.** The ability to answer "who authenticated, when, and how" —
   after an incident, when it is being questioned.

## Adversaries considered

| # | Adversary | Capability |
| --- | --- | --- |
| A1 | Passer-by | Physical access to the camera for a few seconds |
| A2 | Acquaintance | Photos of the target from social media, a phone, a printer |
| A3 | Thief with the disk | The database file, offline, no passphrase |
| A4 | Local attacker with write access | Can modify files as the user running the process |
| A5 | Root on the machine | Full control of the host |

## What is defended, and how

**A1 — walk-up attempts.** Every attempt needs a live face that passes the
quality gate, completes randomly chosen movement challenges within a deadline,
matches an enrolled template, and produces a valid TOTP code. Failures are
counted per subject in the database with exponential backoff, so the attempt
rate collapses after five tries and restarting the process does not reset it
(`test_lockout_survives_a_restart`).

**A2 — photos and replays.** A still image cannot complete the challenges: the
sequence is drawn per attempt, so a recording of a previous successful attempt
only helps if the same challenges come up in the same order inside the same
window. Passive checks run on every frame — a frozen or repeated image, a face
crop too flat to be skin, a scene where the background moves with the face, a
face with no involuntary micro-motion. Enrolment additionally refuses sample
sets that are nearly identical to each other, which is what a held-up photo
produces (`test_identical_samples_are_refused`, `test_a_frozen_camera_cannot_enrol`).
The biometric match is also bound to the liveness sequence: probes are sampled
across the whole attempt and each must match, so presenting a photo for the
comparison and a real face for the movements fails
(`test_a_face_swapped_mid_attempt_is_rejected`).

**A3 — stolen disk.** Templates, second-factor secrets and recovery codes are
sealed with AES-256-GCM under keys derived from the keystore master key, which
is itself wrapped under scrypt(N=2^15) over the operator passphrase. The
database contains no plaintext biometric data, no usable session token, and no
recovery code (`test_template_is_encrypted_on_disk`,
`test_session_tokens_are_not_stored_verbatim`, `test_recovery_codes_are_hashed`).
Raw frames are never written to disk at all.

**A4 — local tampering.** Every sealed record is bound with additional
authenticated data to the row it belongs to (user id, name, engine), so pasting
one user's ciphertext into another user's row fails decryption rather than
granting access (`test_template_cannot_be_moved_between_users`). Renaming a row
invalidates its template. The audit log is an HMAC chain with the head committed
separately, which detects edited records, deleted records, and records removed
from the end (`test_audit_chain_detects_*`).

**Cross-cutting.** Failure messages after the identity decision are identical
whether the account is unknown, disabled, or the second factor was wrong, and
failures are padded to a minimum duration so response time is not an oracle
(`test_unknown_and_known_claims_fail_identically`, `test_failures_are_paced`).

## What is not defended

**A5 — root, or the running process.** Root can read the derived keys out of
process memory, hook the camera, or replace the models. Nothing in a userspace
program changes that. If you need to survive it, the matching and the key
material have to live in a TEE or a secure element, not in Python.

**Injected video.** The liveness checks look at the frames the operating system
hands over. An attacker who can write to the video device (a virtual camera,
a compromised driver) can present a rendered video that responds to the
challenges. The randomised challenge order raises the bar to real-time
rendering, but does not close this.

**A good 3D artefact.** A well-made mask or a high-resolution 3D print will
defeat texture and motion heuristics. This is not certified presentation-attack
detection (ISO/IEC 30107-3) and should not be described as such.

**A phone held up in a still room.** The face-versus-background motion check
catches a *picture* that fills the frame or an injected stream, where everything
moves together. A phone held in a static room leaves a genuinely static
background, and only the active challenge stands in the way.

**The passphrase.** If it is weak, guessable, or sitting in
`FACEID_PASSPHRASE` on a shared machine, the encryption at rest is decoration.
`cli.py doctor` warns when the passphrase is in the environment.

**Denial of service.** The lockout that stops brute force also lets someone lock
a named account out by failing against it repeatedly. That trade is deliberate,
but if availability matters more than it does here, rate-limit by source rather
than by subject.

**Accuracy claims.** The thresholds ship at OpenCV's published SFace operating
point, tightened slightly. No false-accept/false-reject rate is claimed for your
population, your cameras or your lighting. Measure it on your own data before
relying on a number — `FACEID_TEST_FACES` exists for that.

## If you deploy this

- Run `python cli.py doctor --deep` after install and after config changes.
- Keep `FACEID_REQUIRE_SECOND_FACTOR=1`. Face alone is an identity claim, not
  proof of one.
- Ship the audit mirror (`$FACEID_HOME/audit.log`) off the machine. An attacker
  who owns the box can rewrite both the log and its head; the chain only proves
  integrity against someone who cannot rewrite everything atomically.
- Back up the keystore. Losing the passphrase means re-enrolling everyone, which
  is the intended failure mode.
- Decide your retention policy before enrolling anyone. Biometric data is
  regulated in most places this will run (GDPR Art. 9, BIPA, and equivalents),
  and "we kept it because deleting it was work" is not a defence. `cli.py
  delete` removes the enrolment and keeps only the audit record of the deletion.
- Tell people what is stored. A system that quietly enrols faces is the thing
  this rewrite exists to stop.

## Reporting

Found a hole? Open an issue with enough detail to reproduce. If it is a live
deployment issue, contact the maintainer privately first.
