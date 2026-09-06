"""Encrypted store for enrolments, sessions and the audit trail.

The original prototype kept a JPEG of your face sitting in a folder. That file
*is* the credential: copy it, and you are the user. Here nothing biometric is
written in the clear -- only AES-GCM-sealed embeddings, bound to the user row
they belong to, so swapping ciphertext between users fails authentication
instead of granting it.

Three things are worth knowing about the schema:

  * Templates and second-factor secrets are sealed with subkeys from the
    keystore. Losing the passphrase means losing the enrolments -- by design.
  * The audit table is an HMAC chain: each row commits to the one before it,
    so a row cannot be edited or removed without breaking every MAC after it.
  * Failed attempts are counted per subject with exponential backoff, and the
    counter lives in the database rather than in memory, so restarting the
    process is not a way to clear a lockout.
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import crypto

SCHEMA_VERSION = 1
TEMPLATE_AAD_PREFIX = b"faceid/template/v1"
SECRET_AAD_PREFIX = b"faceid/second-factor/v1"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    uid            TEXT PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    status         TEXT NOT NULL DEFAULT 'active',
    engine         TEXT NOT NULL,
    template       BLOB NOT NULL,
    sample_count   INTEGER NOT NULL,
    quality        REAL NOT NULL,
    totp_secret    BLOB,
    totp_counter   INTEGER NOT NULL DEFAULT 0,
    created_at     INTEGER NOT NULL,
    updated_at     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS recovery_codes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    uid       TEXT NOT NULL REFERENCES users(uid) ON DELETE CASCADE,
    salt      BLOB NOT NULL,
    code_hash BLOB NOT NULL,
    used_at   INTEGER
);

CREATE TABLE IF NOT EXISTS attempts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    subject  TEXT NOT NULL,
    uid      TEXT,
    ts       INTEGER NOT NULL,
    success  INTEGER NOT NULL,
    reason   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_subject ON attempts(subject, ts);

CREATE TABLE IF NOT EXISTS lockouts (
    subject      TEXT PRIMARY KEY,
    failures     INTEGER NOT NULL DEFAULT 0,
    locked_until INTEGER NOT NULL DEFAULT 0,
    updated_at   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    uid        TEXT NOT NULL REFERENCES users(uid) ON DELETE CASCADE,
    issued_at  INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    revoked_at INTEGER,
    factors    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit (
    seq      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       INTEGER NOT NULL,
    actor    TEXT NOT NULL,
    action   TEXT NOT NULL,
    subject  TEXT NOT NULL,
    detail   TEXT NOT NULL,
    prev_mac TEXT NOT NULL,
    mac      TEXT NOT NULL
);
"""

GENESIS_MAC = "0" * 44  # base64 of 32 zero bytes is 44 chars; any fixed string works


class StoreError(Exception):
    pass


class UnknownUser(StoreError):
    pass


@dataclass
class User:
    uid: str
    name: str
    status: str
    engine: str
    sample_count: int
    quality: float
    has_second_factor: bool
    created_at: int
    updated_at: int

    @property
    def active(self) -> bool:
        return self.status == "active"


@dataclass
class Lockout:
    subject: str
    failures: int
    locked_until: int

    @property
    def locked(self) -> bool:
        return self.locked_until > time.time()

    @property
    def seconds_left(self) -> int:
        return max(0, int(self.locked_until - time.time()))


class Store:
    """Everything that touches disk goes through here."""

    def __init__(self, path: Path, keyring: crypto.Keyring, *, mirror: Path | None = None):
        self.path = Path(path)
        self.keys = keyring
        self.mirror = Path(mirror) if mirror else None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        first_run = not self.path.exists()
        if first_run:
            # Create the file ourselves so it is never briefly world-readable.
            os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
        self._check_permissions()
        self.conn = sqlite3.connect(str(self.path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(_SCHEMA)
        self._init_meta()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _check_permissions(self) -> None:
        mode = self.path.stat().st_mode & 0o777
        if mode & 0o077:
            raise StoreError(
                f"{self.path} is readable by other users ({mode:o}). "
                f"Fix with: chmod 600 {self.path}"
            )

    def _init_meta(self) -> None:
        found = self._meta_get("schema_version")
        if found is None:
            self._meta_set("schema_version", str(SCHEMA_VERSION))
            self._meta_set("audit_head", GENESIS_MAC)
        elif int(found) != SCHEMA_VERSION:
            raise StoreError(
                f"database schema v{found} does not match this build (v{SCHEMA_VERSION})"
            )

    def _meta_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    # -- users -------------------------------------------------------------

    @staticmethod
    def _template_aad(uid: str, name: str, engine: str) -> bytes:
        # Binding the ciphertext to the row means an attacker who can write to
        # the DB cannot paste their own template into someone else's account.
        return b"|".join([TEMPLATE_AAD_PREFIX, uid.encode(), name.lower().encode(), engine.encode()])

    @staticmethod
    def _secret_aad(uid: str) -> bytes:
        return b"|".join([SECRET_AAD_PREFIX, uid.encode()])

    def create_user(self, name: str, *, engine: str, template: bytes, sample_count: int,
                    quality: float) -> User:
        name = name.strip()
        if not name:
            raise StoreError("user name must not be empty")
        if self.find_user(name) is not None:
            raise StoreError(f"user {name!r} is already enrolled")
        uid = uuid.uuid4().hex
        now = int(time.time())
        sealed = crypto.seal(self.keys.template, template, self._template_aad(uid, name, engine))
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO users(uid, name, status, engine, template, sample_count, quality,"
                " created_at, updated_at) VALUES(?,?,'active',?,?,?,?,?,?)",
                (uid, name, engine, sealed, sample_count, quality, now, now),
            )
        return self.get_user(uid)

    def replace_template(self, uid: str, *, engine: str, template: bytes, sample_count: int,
                         quality: float) -> None:
        user = self.get_user(uid)
        sealed = crypto.seal(
            self.keys.template, template, self._template_aad(uid, user.name, engine)
        )
        with self.transaction() as conn:
            conn.execute(
                "UPDATE users SET engine=?, template=?, sample_count=?, quality=?, updated_at=?"
                " WHERE uid=?",
                (engine, sealed, sample_count, quality, int(time.time()), uid),
            )

    def _row_to_user(self, row: sqlite3.Row) -> User:
        return User(
            uid=row["uid"],
            name=row["name"],
            status=row["status"],
            engine=row["engine"],
            sample_count=row["sample_count"],
            quality=row["quality"],
            has_second_factor=row["totp_secret"] is not None,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_user(self, uid: str) -> User:
        row = self.conn.execute("SELECT * FROM users WHERE uid = ?", (uid,)).fetchone()
        if row is None:
            raise UnknownUser(f"no user with id {uid}")
        return self._row_to_user(row)

    def find_user(self, name: str) -> User | None:
        row = self.conn.execute(
            "SELECT * FROM users WHERE name = ? COLLATE NOCASE", (name.strip(),)
        ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self, *, include_disabled: bool = True) -> list[User]:
        sql = "SELECT * FROM users"
        if not include_disabled:
            sql += " WHERE status = 'active'"
        sql += " ORDER BY name COLLATE NOCASE"
        return [self._row_to_user(r) for r in self.conn.execute(sql)]

    def load_template(self, uid: str) -> bytes:
        row = self.conn.execute(
            "SELECT uid, name, engine, template FROM users WHERE uid = ?", (uid,)
        ).fetchone()
        if row is None:
            raise UnknownUser(f"no user with id {uid}")
        return crypto.unseal(
            self.keys.template,
            row["template"],
            self._template_aad(row["uid"], row["name"], row["engine"]),
        )

    def iter_templates(self, *, active_only: bool = True) -> Iterator[tuple[User, bytes]]:
        for user in self.list_users(include_disabled=not active_only):
            yield user, self.load_template(user.uid)

    def set_status(self, uid: str, status: str) -> None:
        if status not in {"active", "disabled"}:
            raise StoreError(f"unknown status {status!r}")
        with self.transaction() as conn:
            conn.execute(
                "UPDATE users SET status=?, updated_at=? WHERE uid=?",
                (status, int(time.time()), uid),
            )
        if status != "active":
            self.revoke_sessions(uid)

    def delete_user(self, uid: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM users WHERE uid = ?", (uid,))

    # -- second factor -----------------------------------------------------

    def set_second_factor(self, uid: str, secret: str) -> None:
        sealed = crypto.seal(self.keys.secret, secret.encode("utf-8"), self._secret_aad(uid))
        with self.transaction() as conn:
            conn.execute(
                "UPDATE users SET totp_secret=?, totp_counter=0, updated_at=? WHERE uid=?",
                (sealed, int(time.time()), uid),
            )

    def get_second_factor(self, uid: str) -> tuple[str, int] | None:
        row = self.conn.execute(
            "SELECT totp_secret, totp_counter FROM users WHERE uid = ?", (uid,)
        ).fetchone()
        if row is None:
            raise UnknownUser(f"no user with id {uid}")
        if row["totp_secret"] is None:
            return None
        secret = crypto.unseal(self.keys.secret, row["totp_secret"], self._secret_aad(uid))
        return secret.decode("utf-8"), int(row["totp_counter"])

    def spend_totp_counter(self, uid: str, counter: int) -> bool:
        """Record a used counter. False means it was already spent (replay)."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE users SET totp_counter=? WHERE uid=? AND totp_counter < ?",
                (counter, uid, counter),
            )
            return cur.rowcount == 1

    def store_recovery_codes(self, uid: str, codes: Iterable[str]) -> None:
        rows = []
        for code in codes:
            salt = crypto.random_bytes(16)
            rows.append((uid, salt, crypto.hash_secret(code, salt)))
        with self.transaction() as conn:
            conn.execute("DELETE FROM recovery_codes WHERE uid = ?", (uid,))
            conn.executemany(
                "INSERT INTO recovery_codes(uid, salt, code_hash) VALUES(?,?,?)", rows
            )

    def spend_recovery_code(self, uid: str, code: str) -> bool:
        rows = self.conn.execute(
            "SELECT id, salt, code_hash FROM recovery_codes WHERE uid=? AND used_at IS NULL",
            (uid,),
        ).fetchall()
        for row in rows:
            candidate = crypto.hash_secret(code, row["salt"])
            if crypto.constant_time_eq(candidate, row["code_hash"]):
                with self.transaction() as conn:
                    conn.execute(
                        "UPDATE recovery_codes SET used_at=? WHERE id=? AND used_at IS NULL",
                        (int(time.time()), row["id"]),
                    )
                return True
        return False

    def recovery_codes_left(self, uid: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n FROM recovery_codes WHERE uid=? AND used_at IS NULL", (uid,)
        ).fetchone()
        return int(row["n"])

    # -- attempts and lockout ---------------------------------------------

    def record_attempt(self, subject: str, *, success: bool, reason: str,
                       uid: str | None = None) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO attempts(subject, uid, ts, success, reason) VALUES(?,?,?,?,?)",
                (subject, uid, int(time.time()), 1 if success else 0, reason),
            )

    def get_lockout(self, subject: str) -> Lockout:
        row = self.conn.execute(
            "SELECT * FROM lockouts WHERE subject = ?", (subject,)
        ).fetchone()
        if row is None:
            return Lockout(subject=subject, failures=0, locked_until=0)
        return Lockout(subject, int(row["failures"]), int(row["locked_until"]))

    def register_failure(self, subject: str, *, threshold: int, base_s: int,
                         max_s: int) -> Lockout:
        """Count a failure and extend the lockout if we're over the threshold."""
        now = int(time.time())
        current = self.get_lockout(subject)
        failures = current.failures + 1
        locked_until = current.locked_until
        if failures >= threshold:
            over = failures - threshold
            delay = min(base_s * (2**over), max_s)
            locked_until = max(locked_until, now + delay)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO lockouts(subject, failures, locked_until, updated_at)"
                " VALUES(?,?,?,?) ON CONFLICT(subject) DO UPDATE SET"
                " failures=excluded.failures, locked_until=excluded.locked_until,"
                " updated_at=excluded.updated_at",
                (subject, failures, locked_until, now),
            )
        return Lockout(subject, failures, locked_until)

    def clear_failures(self, subject: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM lockouts WHERE subject = ?", (subject,))

    # -- sessions ----------------------------------------------------------

    def create_session(self, uid: str, *, ttl_s: int, factors: list[str]) -> str:
        token = crypto.random_token(32)
        now = int(time.time())
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash, uid, issued_at, expires_at, factors)"
                " VALUES(?,?,?,?,?)",
                (
                    crypto.hash_token(self.keys.session, token),
                    uid,
                    now,
                    now + ttl_s,
                    ",".join(factors),
                ),
            )
        return token

    def validate_session(self, token: str) -> User | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE token_hash = ?",
            (crypto.hash_token(self.keys.session, token),),
        ).fetchone()
        if row is None or row["revoked_at"] is not None:
            return None
        if row["expires_at"] <= time.time():
            return None
        user = self.get_user(row["uid"])
        return user if user.active else None

    def revoke_session(self, token: str) -> bool:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE sessions SET revoked_at=? WHERE token_hash=? AND revoked_at IS NULL",
                (int(time.time()), crypto.hash_token(self.keys.session, token)),
            )
            return cur.rowcount == 1

    def revoke_sessions(self, uid: str) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE sessions SET revoked_at=? WHERE uid=? AND revoked_at IS NULL",
                (int(time.time()), uid),
            )
            return cur.rowcount

    def purge_expired_sessions(self) -> int:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
            return cur.rowcount

    # -- audit -------------------------------------------------------------

    def audit(self, action: str, *, actor: str, subject: str = "-", **detail: Any) -> None:
        """Append a tamper-evident record. Never pass raw biometrics in here."""
        now = int(time.time())
        payload = json.dumps(detail, sort_keys=True, separators=(",", ":"))
        with self.transaction() as conn:
            prev = self._meta_get("audit_head") or GENESIS_MAC
            row = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 AS nxt FROM audit").fetchone()
            seq = int(row["nxt"])
            mac = self._audit_mac(seq, now, actor, action, subject, payload, prev)
            conn.execute(
                "INSERT INTO audit(seq, ts, actor, action, subject, detail, prev_mac, mac)"
                " VALUES(?,?,?,?,?,?,?,?)",
                (seq, now, actor, action, subject, payload, prev, mac),
            )
            self._meta_set("audit_head", mac)
        self._mirror(seq, now, actor, action, subject, payload, mac)

    def _audit_mac(self, seq: int, ts: int, actor: str, action: str, subject: str,
                   detail: str, prev: str) -> str:
        digest = crypto.mac(
            self.keys.audit,
            str(seq).encode(),
            str(ts).encode(),
            actor.encode(),
            action.encode(),
            subject.encode(),
            detail.encode(),
            prev.encode(),
        )
        return base64.b64encode(digest).decode("ascii")

    def _mirror(self, seq: int, ts: int, actor: str, action: str, subject: str,
                detail: str, mac: str) -> None:
        """Best-effort append-only copy, so the log outlives the database file."""
        if self.mirror is None:
            return
        line = json.dumps(
            {"seq": seq, "ts": ts, "actor": actor, "action": action,
             "subject": subject, "detail": json.loads(detail), "mac": mac},
            sort_keys=True,
        )
        fd = os.open(self.mirror, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def verify_audit_chain(self) -> tuple[bool, str]:
        """Recompute every MAC. Returns (ok, human-readable explanation)."""
        prev = GENESIS_MAC
        count = 0
        for row in self.conn.execute("SELECT * FROM audit ORDER BY seq"):
            expected = self._audit_mac(
                row["seq"], row["ts"], row["actor"], row["action"], row["subject"],
                row["detail"], prev,
            )
            if row["prev_mac"] != prev:
                return False, f"record {row['seq']} does not follow record {row['seq'] - 1}"
            if not crypto.constant_time_eq(expected.encode(), row["mac"].encode()):
                return False, f"record {row['seq']} has been modified"
            prev = row["mac"]
            count += 1
        head = self._meta_get("audit_head") or GENESIS_MAC
        if head != prev:
            return False, "records have been removed from the end of the log"
        return True, f"{count} records verified"

    def read_audit(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,))
        )
