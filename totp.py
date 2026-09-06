"""TOTP (RFC 6238) and one-time recovery codes.

Face recognition is a convenience factor: you leave your face in every photo
you've ever been in, and you can't rotate it after a breach. So for anything
that matters we pair it with something the user *has*. Standard TOTP means any
authenticator app works -- no custom client, no shared secret over the wire.

Verified against the RFC 6238 appendix B vectors in tests/test_totp.py.
"""

from __future__ import annotations

import base64
import hmac
import secrets
import struct
import time
from hashlib import sha1, sha256, sha512
from urllib.parse import quote

_DIGEST = {"SHA1": sha1, "SHA256": sha256, "SHA512": sha512}
RECOVERY_CODE_COUNT = 10


def generate_secret(length: int = 20) -> str:
    """Base32 secret, unpadded. 160 bits is what RFC 4226 recommends."""
    return base64.b32encode(secrets.token_bytes(length)).decode("ascii").rstrip("=")


def _decode_secret(secret: str) -> bytes:
    cleaned = secret.strip().replace(" ", "").upper()
    padding = "=" * (-len(cleaned) % 8)
    try:
        return base64.b32decode(cleaned + padding, casefold=True)
    except Exception as exc:  # binascii.Error and friends
        raise ValueError("second-factor secret is not valid base32") from exc


def hotp(secret: str, counter: int, digits: int = 6, algorithm: str = "SHA1") -> str:
    key = _decode_secret(secret)
    digest = hmac.new(key, struct.pack(">Q", counter), _DIGEST[algorithm.upper()]).digest()
    offset = digest[-1] & 0x0F
    code = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(code % (10**digits)).zfill(digits)


def counter_for(timestamp: float | None = None, period: int = 30, t0: int = 0) -> int:
    now = time.time() if timestamp is None else timestamp
    return int((now - t0) // period)


def totp(secret: str, timestamp: float | None = None, *, digits: int = 6,
         period: int = 30, algorithm: str = "SHA1") -> str:
    return hotp(secret, counter_for(timestamp, period), digits, algorithm)


def verify(secret: str, code: str, *, timestamp: float | None = None, digits: int = 6,
           period: int = 30, algorithm: str = "SHA1", window: int = 1,
           last_counter: int | None = None) -> int | None:
    """Return the counter the code matched, or None.

    `last_counter` is the highest counter already spent by this user. Passing it
    blocks replay: a code shoulder-surfed (or captured from a phishing page) is
    dead the moment it's been used once, instead of staying valid for the rest
    of its 30-second step plus the drift window.
    """
    code = code.strip().replace(" ", "")
    if not code.isdigit() or len(code) != digits:
        return None
    current = counter_for(timestamp, period)
    for drift in range(-window, window + 1):
        counter = current + drift
        if counter < 0:
            continue
        if last_counter is not None and counter <= last_counter:
            continue
        if hmac.compare_digest(hotp(secret, counter, digits, algorithm), code):
            return counter
    return None


def provisioning_uri(secret: str, account: str, issuer: str = "faceid",
                     digits: int = 6, period: int = 30, algorithm: str = "SHA1") -> str:
    """otpauth:// URI for enrolment in an authenticator app."""
    label = quote(f"{issuer}:{account}", safe="")
    params = (
        f"secret={secret}&issuer={quote(issuer, safe='')}"
        f"&algorithm={algorithm.upper()}&digits={digits}&period={period}"
    )
    return f"otpauth://totp/{label}?{params}"


def generate_recovery_codes(count: int = RECOVERY_CODE_COUNT) -> list[str]:
    """Human-typeable single-use codes for when the phone is lost or broken."""
    alphabet = "23456789abcdefghjkmnpqrstuvwxyz"  # no 0/o/1/l/i
    codes = []
    for _ in range(count):
        raw = "".join(secrets.choice(alphabet) for _ in range(16))
        codes.append(f"{raw[:4]}-{raw[4:8]}-{raw[8:12]}-{raw[12:]}")
    return codes


def normalise_recovery_code(code: str) -> str:
    return code.strip().lower().replace(" ", "").replace("-", "")
