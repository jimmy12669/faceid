"""Key management and authenticated encryption.

Design in one paragraph: a random 32-byte master key is generated once and
stored wrapped under a key derived from the operator passphrase (scrypt).
Changing the passphrase re-wraps the master key instead of re-encrypting the
whole database. Everything the master key protects gets its own subkey via
HKDF, so a template-decryption bug can't be turned into audit-log forgery.

Nothing here rolls its own primitives -- it's AES-256-GCM, scrypt, HKDF and
HMAC from `cryptography`/`hashlib`, wired together carefully.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

KEYSTORE_VERSION = 1
_KEYSTORE_AAD = b"faceid/keystore/v1"
_NONCE_LEN = 12

# HKDF labels. Add new ones; never reuse an old one for a new purpose.
INFO_TEMPLATE = b"faceid/template-encryption/v1"
INFO_AUDIT = b"faceid/audit-chain/v1"
INFO_SESSION = b"faceid/session-token/v1"
INFO_SECRET = b"faceid/second-factor/v1"


class CryptoError(Exception):
    pass


class BadPassphrase(CryptoError):
    """Raised when unwrapping fails -- wrong passphrase or tampered keystore."""


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


def random_bytes(n: int = 32) -> bytes:
    return secrets.token_bytes(n)


def random_token(n: int = 32) -> str:
    return secrets.token_urlsafe(n)


def constant_time_eq(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


def derive_subkey(master: bytes, info: bytes, length: int = 32) -> bytes:
    """Split the master key into purpose-bound subkeys."""
    return HKDF(algorithm=SHA256(), length=length, salt=None, info=info).derive(master)


def seal(key: bytes, plaintext: bytes, aad: bytes = b"") -> bytes:
    """AES-256-GCM. Output is nonce || ciphertext || tag."""
    nonce = os.urandom(_NONCE_LEN)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def unseal(key: bytes, blob: bytes, aad: bytes = b"") -> bytes:
    if len(blob) <= _NONCE_LEN:
        raise CryptoError("ciphertext too short")
    nonce, body = blob[:_NONCE_LEN], blob[_NONCE_LEN:]
    try:
        return AESGCM(key).decrypt(nonce, body, aad)
    except InvalidTag as exc:
        raise CryptoError("decryption failed: wrong key or tampered data") from exc


def mac(key: bytes, *parts: bytes) -> bytes:
    """HMAC over length-prefixed parts, so ('ab','c') != ('a','bc')."""
    h = hmac.new(key, digestmod=sha256)
    for part in parts:
        h.update(len(part).to_bytes(4, "big"))
        h.update(part)
    return h.digest()


def hash_token(key: bytes, token: str) -> str:
    """Session tokens are stored keyed-hashed; a stolen DB yields no usable token."""
    return _b64(mac(key, token.encode("utf-8")))


def hash_secret(secret: str, salt: bytes, n: int = 1 << 14, r: int = 8, p: int = 1) -> bytes:
    """For recovery codes. Deliberately slow, salted per code."""
    return Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(secret.encode("utf-8"))


@dataclass
class Keyring:
    """Purpose-bound subkeys held in memory for the life of a command."""

    template: bytes
    audit: bytes
    session: bytes
    secret: bytes

    @classmethod
    def from_master(cls, master: bytes) -> "Keyring":
        return cls(
            template=derive_subkey(master, INFO_TEMPLATE),
            audit=derive_subkey(master, INFO_AUDIT),
            session=derive_subkey(master, INFO_SESSION),
            secret=derive_subkey(master, INFO_SECRET),
        )


class KeyStore:
    """The wrapped master key on disk."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def exists(self) -> bool:
        return self.path.exists()

    def create(self, passphrase: str, *, n: int, r: int, p: int) -> Keyring:
        if self.exists():
            raise CryptoError(f"keystore already exists at {self.path}")
        _check_passphrase(passphrase)
        master = random_bytes(32)
        salt = random_bytes(16)
        kek = Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(passphrase.encode("utf-8"))
        doc = {
            "version": KEYSTORE_VERSION,
            "created_at": int(time.time()),
            "kdf": {"name": "scrypt", "salt": _b64(salt), "n": n, "r": r, "p": p},
            "wrapped_key": _b64(seal(kek, master, _KEYSTORE_AAD)),
        }
        self._write(doc)
        return Keyring.from_master(master)

    def unlock(self, passphrase: str) -> Keyring:
        doc = self._read()
        if doc.get("version") != KEYSTORE_VERSION:
            raise CryptoError(f"unsupported keystore version {doc.get('version')!r}")
        kdf = doc["kdf"]
        if kdf.get("name") != "scrypt":
            raise CryptoError(f"unsupported KDF {kdf.get('name')!r}")
        kek = Scrypt(
            salt=_unb64(kdf["salt"]), length=32, n=kdf["n"], r=kdf["r"], p=kdf["p"]
        ).derive(passphrase.encode("utf-8"))
        try:
            master = unseal(kek, _unb64(doc["wrapped_key"]), _KEYSTORE_AAD)
        except CryptoError as exc:
            raise BadPassphrase("passphrase rejected") from exc
        return Keyring.from_master(master)

    def rewrap(self, old_passphrase: str, new_passphrase: str, *, n: int, r: int, p: int) -> None:
        """Change the passphrase without touching any encrypted record."""
        _check_passphrase(new_passphrase)
        doc = self._read()
        kdf = doc["kdf"]
        old_kek = Scrypt(
            salt=_unb64(kdf["salt"]), length=32, n=kdf["n"], r=kdf["r"], p=kdf["p"]
        ).derive(old_passphrase.encode("utf-8"))
        try:
            master = unseal(old_kek, _unb64(doc["wrapped_key"]), _KEYSTORE_AAD)
        except CryptoError as exc:
            raise BadPassphrase("passphrase rejected") from exc

        salt = random_bytes(16)
        new_kek = Scrypt(salt=salt, length=32, n=n, r=r, p=p).derive(new_passphrase.encode("utf-8"))
        doc["kdf"] = {"name": "scrypt", "salt": _b64(salt), "n": n, "r": r, "p": p}
        doc["wrapped_key"] = _b64(seal(new_kek, master, _KEYSTORE_AAD))
        doc["rewrapped_at"] = int(time.time())
        self._write(doc)

    def _read(self) -> dict:
        if not self.exists():
            raise CryptoError(f"no keystore at {self.path} -- run `faceid init` first")
        mode = self.path.stat().st_mode & 0o777
        if mode & 0o077:
            raise CryptoError(
                f"{self.path} is group/world readable ({mode:o}); refusing to use it. "
                "Fix with: chmod 600 " + str(self.path)
            )
        return json.loads(self.path.read_text("utf-8"))

    def _write(self, doc: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        # Create with 0600 from the start -- never a window where it's readable.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)


def _check_passphrase(passphrase: str) -> None:
    """Minimum bar for the key that protects every template on the box."""
    if len(passphrase) < 12:
        raise CryptoError("passphrase must be at least 12 characters")
    if passphrase.strip() != passphrase or not passphrase.strip():
        raise CryptoError("passphrase must not start or end with whitespace")
