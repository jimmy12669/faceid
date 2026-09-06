import json

import pytest

import crypto


def test_seal_roundtrip_and_aad_binding(keyring):
    blob = crypto.seal(keyring.template, b"secret payload", b"uid=alice")
    assert crypto.unseal(keyring.template, blob, b"uid=alice") == b"secret payload"
    with pytest.raises(crypto.CryptoError):
        crypto.unseal(keyring.template, blob, b"uid=bob")


def test_ciphertext_is_not_plaintext(keyring):
    blob = crypto.seal(keyring.template, b"face template goes here")
    assert b"face template" not in blob


def test_flipping_one_bit_is_detected(keyring):
    blob = bytearray(crypto.seal(keyring.template, b"payload"))
    blob[-1] ^= 0x01
    with pytest.raises(crypto.CryptoError):
        crypto.unseal(keyring.template, bytes(blob))


def test_subkeys_are_independent(keyring):
    assert len({keyring.template, keyring.audit, keyring.session, keyring.secret}) == 4
    blob = crypto.seal(keyring.template, b"x")
    with pytest.raises(crypto.CryptoError):
        crypto.unseal(keyring.audit, blob)


def test_wrong_passphrase_rejected(settings, keyring):
    ks = crypto.KeyStore(settings.keystore_path)
    with pytest.raises(crypto.BadPassphrase):
        ks.unlock("not the passphrase")
    assert ks.unlock("test passphrase 123").template == keyring.template


def test_rewrap_keeps_the_master_key(settings, keyring):
    ks = crypto.KeyStore(settings.keystore_path)
    ks.rewrap("test passphrase 123", "a different passphrase", n=settings.scrypt_n, r=8, p=1)
    assert ks.unlock("a different passphrase").template == keyring.template
    with pytest.raises(crypto.BadPassphrase):
        ks.unlock("test passphrase 123")


def test_short_passphrase_refused(tmp_path, settings):
    ks = crypto.KeyStore(tmp_path / "ks.json")
    with pytest.raises(crypto.CryptoError, match="12 characters"):
        ks.create("short", n=settings.scrypt_n, r=8, p=1)


def test_keystore_file_is_private(settings, keyring):
    assert settings.keystore_path.stat().st_mode & 0o077 == 0


def test_keystore_never_holds_the_raw_key(settings, keyring):
    doc = json.loads(settings.keystore_path.read_text())
    assert set(doc) >= {"kdf", "wrapped_key", "version"}
    assert "master" not in json.dumps(doc)


def test_mac_is_not_ambiguous_across_fields(keyring):
    # Length-prefixing means ("ab","c") and ("a","bc") are different messages.
    assert crypto.mac(keyring.audit, b"ab", b"c") != crypto.mac(keyring.audit, b"a", b"bc")


def test_world_readable_keystore_is_refused(settings, keyring):
    settings.keystore_path.chmod(0o644)
    with pytest.raises(crypto.CryptoError, match="readable"):
        crypto.KeyStore(settings.keystore_path).unlock("test passphrase 123")
