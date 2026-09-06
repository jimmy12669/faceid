import base64
import time

import pytest

import totp

# RFC 6238, appendix B. Secret is the ASCII string "12345678901234567890".
RFC_SECRET = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
RFC_VECTORS = [
    (59, "94287082"),
    (1111111109, "07081804"),
    (1111111111, "14050471"),
    (1234567890, "89005924"),
    (2000000000, "69279037"),
    (20000000000, "65353130"),
]


@pytest.mark.parametrize("timestamp,expected", RFC_VECTORS)
def test_rfc6238_vectors(timestamp, expected):
    assert totp.totp(RFC_SECRET, timestamp, digits=8) == expected


def test_verify_accepts_current_code():
    secret = totp.generate_secret()
    assert totp.verify(secret, totp.totp(secret)) is not None


def test_verify_rejects_garbage():
    secret = totp.generate_secret()
    assert totp.verify(secret, "000000") is None
    assert totp.verify(secret, "abcdef") is None
    assert totp.verify(secret, "") is None


def test_drift_window():
    secret = totp.generate_secret()
    now = time.time()
    previous = totp.totp(secret, now - 30)
    assert totp.verify(secret, previous, timestamp=now) is not None
    ancient = totp.totp(secret, now - 300)
    assert totp.verify(secret, ancient, timestamp=now) is None


def test_used_counter_cannot_be_replayed():
    secret = totp.generate_secret()
    code = totp.totp(secret)
    counter = totp.verify(secret, code)
    assert counter is not None
    assert totp.verify(secret, code, last_counter=counter) is None


def test_secret_is_long_enough():
    secret = totp.generate_secret()
    assert len(base64.b32decode(secret + "=" * (-len(secret) % 8))) == 20


def test_recovery_codes_are_unique_and_typeable():
    codes = totp.generate_recovery_codes(10)
    assert len(set(codes)) == 10
    for code in codes:
        assert len(totp.normalise_recovery_code(code)) == 16
        assert not set("01lio") & set(totp.normalise_recovery_code(code))


def test_provisioning_uri_round_trips():
    uri = totp.provisioning_uri("ABCDEFGHIJKLMNOP", "alice smith", issuer="faceid")
    assert uri.startswith("otpauth://totp/faceid%3Aalice%20smith?")
    assert "secret=ABCDEFGHIJKLMNOP" in uri
