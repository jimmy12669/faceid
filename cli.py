#!/usr/bin/env python3
"""Command line front end.

    python cli.py init                 set up the keystore and database
    python cli.py enroll alice         register a face (and a second factor)
    python cli.py verify               authenticate whoever is at the camera
    python cli.py verify alice         authenticate a specific claim
    python cli.py users                who is enrolled
    python cli.py audit --verify       check the log has not been edited
    python cli.py doctor               look for weak spots in the install

Every command that touches enrolments needs the keystore passphrase. That's the
point: without it the templates on disk are ciphertext and nothing else.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

import config
import crypto
import database


def _passphrase(prompt: str = "Keystore passphrase: ", *, confirm: bool = False) -> str:
    from_env = os.environ.get("FACEID_PASSPHRASE")
    if from_env:
        print(
            "warning: using FACEID_PASSPHRASE from the environment. Fine for a "
            "kiosk with a locked-down profile, bad on a shared machine "
            "(it shows up in /proc and in shell history).",
            file=sys.stderr,
        )
        return from_env
    value = getpass.getpass(prompt)
    if confirm and value != getpass.getpass("Confirm passphrase: "):
        raise SystemExit("passphrases did not match")
    return value


def _open_store(settings, *, passphrase: str | None = None) -> database.Store:
    keystore = crypto.KeyStore(settings.keystore_path)
    if not keystore.exists():
        raise SystemExit(f"no keystore at {settings.keystore_path} -- run `python cli.py init`")
    keyring = keystore.unlock(passphrase or _passphrase())
    return database.Store(settings.db_path, keyring, mirror=settings.audit_mirror_path)


def _engine(settings):
    import biometrics

    return biometrics.FaceEngine(settings.models_dir, detector_score=settings.detector_score)


def _source(args):
    import image

    if getattr(args, "replay", None):
        import cv2

        frames = [cv2.imread(str(p)) for p in sorted(Path(args.replay).glob("*"))]
        frames = [f for f in frames if f is not None]
        if not frames:
            raise SystemExit(f"no readable images in {args.replay}")
        return image.ReplaySource(frames), (lambda: None)
    camera = image.Camera(args.camera).open()
    return camera, camera.close


def _ui(args, title: str):
    import image

    return image.select_ui(headless=args.headless, title=title)


def _write_session(settings, token: str):
    """Park the session token in a private file rather than in the scrollback."""
    path = settings.home / "session.token"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(token)
    return path


def _read_session(settings) -> str | None:
    path = settings.home / "session.token"
    return path.read_text("utf-8").strip() if path.exists() else None


def _relative(ts: int) -> str:
    delta = max(0, int(time.time()) - ts)
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if delta >= size:
            return f"{delta // size}{unit} ago"
    return "just now"


# -- commands --------------------------------------------------------------


def cmd_init(args, settings) -> int:
    settings.ensure_home()
    keystore = crypto.KeyStore(settings.keystore_path)
    if keystore.exists():
        print(f"Keystore already exists at {settings.keystore_path}")
        return 1
    print(
        "Choose a keystore passphrase. It protects every enrolment on this\n"
        "machine, it is not recoverable, and there is no reset.\n"
    )
    passphrase = _passphrase("New keystore passphrase: ", confirm=True)
    keyring = keystore.create(
        passphrase, n=settings.scrypt_n, r=settings.scrypt_r, p=settings.scrypt_p
    )
    with database.Store(settings.db_path, keyring, mirror=settings.audit_mirror_path) as store:
        store.audit("store.initialised", actor=os.environ.get("USER", "unknown"))
    print(f"\nReady. State lives in {settings.home.resolve()}")
    if not (settings.models_dir / "face_recognition_sface_2021dec.onnx").exists():
        print("Next: python fetch_models.py")
    return 0


def cmd_enroll(args, settings) -> int:
    import auth

    store = _open_store(settings)
    try:
        engine = _engine(settings)
        source, close = _source(args)
        ui = _ui(args, f"Enrolling {args.name}")
        authenticator = auth.Authenticator(
            store, engine, settings, actor=os.environ.get("USER", "operator"),
            invert_yaw=args.invert_yaw,
        )
        print(f"Enrolling {args.name}. Follow the prompts; this takes a few seconds.")
        try:
            result = authenticator.enroll(
                args.name, source, on_progress=ui, force=args.force,
                second_factor=not args.no_second_factor,
            )
        finally:
            close()
            ui.close()

        print()
        print(result.message)
        if not result.ok:
            return 1
        secret = result.detail.get("totp_secret")
        if secret:
            print("\nSecond factor -- add this to an authenticator app now:")
            print(f"  secret : {secret}")
            print(f"  uri    : {result.detail['totp_uri']}")
            print("\nRecovery codes (each works once, store them somewhere safe):")
            for code in result.detail["recovery_codes"]:
                print(f"  {code}")
            print("\nThis is the only time any of that is shown.")
        return 0
    finally:
        store.close()


def cmd_verify(args, settings) -> int:
    import auth

    store = _open_store(settings)
    try:
        engine = _engine(settings)
        source, close = _source(args)
        ui = _ui(args, "Verifying")
        authenticator = auth.Authenticator(
            store, engine, settings, actor="cli", invert_yaw=args.invert_yaw
        )

        def second_factor(prompt: str) -> str | None:
            try:
                return input(f"{prompt}: ").strip()
            except (EOFError, KeyboardInterrupt):
                return None

        try:
            result = authenticator.authenticate(
                source, claimed_name=args.name, on_progress=ui,
                second_factor=second_factor,
            )
        finally:
            close()
            ui.close()

        print()
        print(result.message)
        if result.ok:
            print(f"  similarity : {result.similarity:.3f}")
            print(f"  factors    : {', '.join(result.detail['factors'])}")
            print(f"  expires in : {result.detail['expires_in']}s")
            path = _write_session(settings, result.session_token)
            if args.show_token:
                print(f"  session    : {result.session_token}")
            else:
                print(f"  session    : written to {path}")
            return 0
        return 1
    finally:
        store.close()


def cmd_whoami(args, settings) -> int:
    token = args.token or _read_session(settings)
    if not token:
        print("No session. Run `python cli.py verify` first.")
        return 1
    store = _open_store(settings)
    try:
        user = store.validate_session(token)
        if user is None:
            print("Session is not valid (expired, revoked, or the user was disabled).")
            return 1
        print(f"{user.name} (enrolled {_relative(user.created_at)})")
        return 0
    finally:
        store.close()


def cmd_logout(args, settings) -> int:
    token = args.token or _read_session(settings)
    if not token:
        print("No session to end.")
        return 1
    store = _open_store(settings)
    try:
        user = store.validate_session(token)
        if store.revoke_session(token):
            store.audit("session.revoked", actor="cli",
                        subject=user.name if user else "-")
            print("Signed out.")
        else:
            print("That session was already over.")
        (settings.home / "session.token").unlink(missing_ok=True)
        return 0
    finally:
        store.close()


def cmd_users(args, settings) -> int:
    store = _open_store(settings)
    try:
        users = store.list_users()
        if not users:
            print("Nobody is enrolled yet.")
            return 0
        print(f"{'NAME':<20}{'STATUS':<10}{'2FA':<6}{'SAMPLES':<9}{'QUALITY':<9}ENROLLED")
        for user in users:
            print(
                f"{user.name:<20}{user.status:<10}"
                f"{'yes' if user.has_second_factor else 'no':<6}"
                f"{user.sample_count:<9}{user.quality:<9.2f}{_relative(user.created_at)}"
            )
        return 0
    finally:
        store.close()


def cmd_status(args, settings) -> int:
    """Set a user active or disabled."""
    store = _open_store(settings)
    try:
        user = store.find_user(args.name)
        if user is None:
            raise SystemExit(f"no user named {args.name!r}")
        status = "disabled" if args.command == "disable" else "active"
        store.set_status(user.uid, status)
        store.audit(f"user.{status}", actor=os.environ.get("USER", "operator"), subject=user.name)
        print(f"{user.name} is now {status}.")
        return 0
    finally:
        store.close()


def cmd_delete(args, settings) -> int:
    store = _open_store(settings)
    try:
        user = store.find_user(args.name)
        if user is None:
            raise SystemExit(f"no user named {args.name!r}")
        if not args.yes:
            answer = input(f"Delete the enrolment for {user.name}? [y/N] ").strip().lower()
            if answer != "y":
                print("Left alone.")
                return 1
        store.delete_user(user.uid)
        store.audit("user.deleted", actor=os.environ.get("USER", "operator"), subject=user.name)
        print(f"Deleted {user.name}. The audit record of the deletion stays.")
        return 0
    finally:
        store.close()


def cmd_sessions(args, settings) -> int:
    store = _open_store(settings)
    try:
        if args.purge:
            print(f"Removed {store.purge_expired_sessions()} expired sessions.")
        if args.revoke:
            user = store.find_user(args.revoke)
            if user is None:
                raise SystemExit(f"no user named {args.revoke!r}")
            count = store.revoke_sessions(user.uid)
            store.audit("session.revoked_all", actor=os.environ.get("USER", "operator"),
                        subject=user.name, count=count)
            print(f"Revoked {count} session(s) for {user.name}.")
        rows = store.conn.execute(
            "SELECT s.*, u.name FROM sessions s JOIN users u ON u.uid = s.uid"
            " WHERE s.revoked_at IS NULL AND s.expires_at > ? ORDER BY s.issued_at DESC",
            (int(time.time()),),
        ).fetchall()
        if not rows:
            print("No live sessions.")
            return 0
        print(f"{'USER':<20}{'FACTORS':<18}{'ISSUED':<14}EXPIRES IN")
        for row in rows:
            print(
                f"{row['name']:<20}{row['factors']:<18}{_relative(row['issued_at']):<14}"
                f"{max(0, row['expires_at'] - int(time.time()))}s"
            )
        return 0
    finally:
        store.close()


def cmd_audit(args, settings) -> int:
    store = _open_store(settings)
    try:
        if args.verify:
            ok, detail = store.verify_audit_chain()
            print(("chain intact: " if ok else "CHAIN BROKEN: ") + detail)
            return 0 if ok else 2
        rows = store.read_audit(args.number)
        if not rows:
            print("Audit log is empty.")
            return 0
        for row in reversed(rows):
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"]))
            detail = json.loads(row["detail"])
            extra = " ".join(f"{k}={v}" for k, v in sorted(detail.items()) if k != "liveness")
            print(f"{row['seq']:>5}  {stamp}  {row['action']:<22}{row['subject']:<16}{extra}")
        return 0
    finally:
        store.close()


def cmd_passphrase(args, settings) -> int:
    keystore = crypto.KeyStore(settings.keystore_path)
    old = _passphrase("Current passphrase: ")
    new = _passphrase("New passphrase: ", confirm=True)
    keystore.rewrap(old, new, n=settings.scrypt_n, r=settings.scrypt_r, p=settings.scrypt_p)
    print("Passphrase changed. Enrolments were not re-encrypted -- only the key wrapping.")
    return 0


def cmd_doctor(args, settings) -> int:
    """Look for the boring mistakes that undo all of the above."""
    problems: list[str] = []
    notes: list[str] = []

    def check(label: str, ok: bool, detail: str = "", *, warn_only: bool = False) -> None:
        mark = "ok  " if ok else ("warn" if warn_only else "FAIL")
        print(f"  [{mark}] {label}" + (f" -- {detail}" if detail else ""))
        if not ok:
            (notes if warn_only else problems).append(label)

    print(f"faceid doctor -- {settings.home.resolve()}\n")

    home_exists = settings.home.exists()
    check("state directory exists", home_exists, str(settings.home))
    if home_exists:
        mode = settings.home.stat().st_mode & 0o777
        check("state directory is private", not (mode & 0o077), f"mode {mode:o}")

    for label, path in (("keystore", settings.keystore_path), ("database", settings.db_path)):
        if path.exists():
            mode = path.stat().st_mode & 0o777
            check(f"{label} permissions", not (mode & 0o077), f"mode {mode:o}")
        else:
            check(f"{label} present", False, f"missing at {path}")

    import fetch_models

    for model in fetch_models.MODELS:
        path = settings.models_dir / model["name"]
        if not path.exists():
            check(f"model {model['name']}", False, "not downloaded")
            continue
        check(f"model {model['name']}", fetch_models.digest(path) == model["sha256"],
              "sha256 pinned")

    check("second factor required by policy", settings.require_second_factor,
          "FACEID_REQUIRE_SECOND_FACTOR=0" if not settings.require_second_factor else "",
          warn_only=True)
    check("match threshold is not loose", settings.match_threshold >= 0.363,
          f"threshold {settings.match_threshold}")
    check("passphrase not held in the environment", "FACEID_PASSPHRASE" not in os.environ,
          warn_only=True)
    check("lockout enabled", settings.max_failed_attempts > 0,
          f"{settings.max_failed_attempts} attempts then backoff")

    if settings.keystore_path.exists() and args.deep:
        store = _open_store(settings)
        try:
            ok, detail = store.verify_audit_chain()
            check("audit chain", ok, detail)
            weak = [u.name for u in store.list_users() if not u.has_second_factor]
            check("all users have a second factor", not weak,
                  ", ".join(weak) if weak else "", warn_only=True)
        finally:
            store.close()
    elif not args.deep:
        print("\n  (run with --deep to unlock the keystore and check the audit chain)")

    print()
    if problems:
        print(f"{len(problems)} problem(s) need attention.")
        return 2
    if notes:
        print(f"No failures. {len(notes)} thing(s) worth a second look.")
        return 0
    print("Nothing to report.")
    return 0


# -- wiring ----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="faceid",
        description="Face authentication with liveness checks, encrypted templates and an audit log.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def camera_args(p):
        p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
        p.add_argument("--headless", action="store_true",
                       help="terminal prompts instead of a preview window")
        p.add_argument("--replay", type=Path,
                       help="read frames from a directory of images instead of a camera")
        p.add_argument("--invert-yaw", action="store_true",
                       help="use if your webcam mirrors and the turn challenges read backwards")

    sub.add_parser("init", help="create the keystore and database").set_defaults(func=cmd_init)

    p = sub.add_parser("enroll", help="register a face")
    p.add_argument("name")
    p.add_argument("--force", action="store_true", help="replace an existing enrolment")
    p.add_argument("--no-second-factor", action="store_true",
                   help="skip TOTP setup (not recommended)")
    camera_args(p)
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("verify", help="authenticate")
    p.add_argument("name", nargs="?", help="claimed identity; omit to search all enrolments")
    p.add_argument("--show-token", action="store_true",
                   help="print the session token instead of only writing it to a file")
    camera_args(p)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("whoami", help="who the current session belongs to")
    p.add_argument("token", nargs="?", help="defaults to the saved session")
    p.set_defaults(func=cmd_whoami)

    p = sub.add_parser("logout", help="end a session")
    p.add_argument("token", nargs="?", help="defaults to the saved session")
    p.set_defaults(func=cmd_logout)

    sub.add_parser("users", help="list enrolments").set_defaults(func=cmd_users)

    for name, helptext in (("disable", "suspend a user"), ("enable", "restore a user")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("name")
        p.set_defaults(func=cmd_status)

    p = sub.add_parser("delete", help="remove an enrolment")
    p.add_argument("name")
    p.add_argument("--yes", action="store_true", help="skip the confirmation")
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("sessions", help="list, purge or revoke sessions")
    p.add_argument("--purge", action="store_true", help="delete expired sessions")
    p.add_argument("--revoke", metavar="NAME", help="revoke every session for a user")
    p.set_defaults(func=cmd_sessions)

    p = sub.add_parser("audit", help="read or verify the audit log")
    p.add_argument("-n", "--number", type=int, default=25)
    p.add_argument("--verify", action="store_true", help="recompute the HMAC chain")
    p.set_defaults(func=cmd_audit)

    sub.add_parser("passphrase", help="change the keystore passphrase").set_defaults(
        func=cmd_passphrase
    )

    p = sub.add_parser("doctor", help="check the install for weak spots")
    p.add_argument("--deep", action="store_true", help="also unlock the keystore and audit")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = config.load()
    try:
        return args.func(args, settings)
    except crypto.BadPassphrase:
        print("Passphrase rejected.", file=sys.stderr)
        return 1
    except (crypto.CryptoError, database.StoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
