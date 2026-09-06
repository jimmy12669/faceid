import time

import pytest

import crypto
import database


def make_user(store, name="alice", template=b"template-bytes"):
    return store.create_user(name, engine="test", template=template, sample_count=5, quality=0.9)


def test_template_roundtrip(store):
    user = make_user(store)
    assert store.load_template(user.uid) == b"template-bytes"


def test_template_is_encrypted_on_disk(store, settings):
    make_user(store, template=b"UNIQUE-BIOMETRIC-MARKER")
    raw = settings.db_path.read_bytes()
    assert b"UNIQUE-BIOMETRIC-MARKER" not in raw


def test_template_cannot_be_moved_between_users(store):
    alice = make_user(store, "alice", b"alice-template")
    bob = make_user(store, "bob", b"bob-template")
    stolen = store.conn.execute(
        "SELECT template FROM users WHERE uid = ?", (alice.uid,)
    ).fetchone()[0]
    store.conn.execute("UPDATE users SET template = ? WHERE uid = ?", (stolen, bob.uid))
    # The AAD binds ciphertext to the row it belongs to, so this is not a
    # silent identity swap -- it is a decryption failure.
    with pytest.raises(crypto.CryptoError):
        store.load_template(bob.uid)


def test_renaming_a_row_invalidates_its_template(store):
    alice = make_user(store)
    store.conn.execute("UPDATE users SET name = 'mallory' WHERE uid = ?", (alice.uid,))
    with pytest.raises(crypto.CryptoError):
        store.load_template(alice.uid)


def test_duplicate_names_refused(store):
    make_user(store, "alice")
    with pytest.raises(database.StoreError):
        make_user(store, "ALICE")


def test_audit_chain_detects_edits(store):
    make_user(store)
    store.audit("auth.success", actor="cli", subject="alice", similarity=0.9)
    store.audit("auth.failure", actor="cli", subject="mallory", reason="no_match")
    assert store.verify_audit_chain()[0]

    store.conn.execute("UPDATE audit SET subject = 'alice' WHERE action = 'auth.failure'")
    ok, why = store.verify_audit_chain()
    assert not ok and "modified" in why


def test_audit_chain_detects_deletions(store):
    store.audit("one", actor="cli")
    store.audit("two", actor="cli")
    store.audit("three", actor="cli")
    store.conn.execute("DELETE FROM audit WHERE action = 'two'")
    assert not store.verify_audit_chain()[0]


def test_audit_chain_detects_truncation(store):
    store.audit("one", actor="cli")
    store.audit("two", actor="cli")
    store.conn.execute("DELETE FROM audit WHERE action = 'two'")
    ok, why = store.verify_audit_chain()
    assert not ok and "removed" in why


def test_audit_mirror_is_append_only_copy(store, settings):
    store.audit("one", actor="cli", subject="alice")
    store.audit("two", actor="cli", subject="bob")
    lines = settings.audit_mirror_path.read_text().strip().splitlines()
    assert len(lines) == 2 and '"action": "two"' in lines[1]


def test_lockout_backs_off_exponentially(store):
    delays = []
    for _ in range(8):
        lock = store.register_failure("alice", threshold=3, base_s=10, max_s=120)
        delays.append(lock.seconds_left)
    assert delays[:2] == [0, 0]        # under the threshold, no delay
    assert delays[2] > 0               # threshold reached
    assert delays[3] > delays[2]       # and it grows
    assert max(delays) <= 120          # but is capped


def test_success_clears_the_lockout(store):
    for _ in range(5):
        store.register_failure("alice", threshold=3, base_s=10, max_s=120)
    assert store.get_lockout("alice").locked
    store.clear_failures("alice")
    assert not store.get_lockout("alice").locked


def test_lockout_survives_a_restart(settings, keyring):
    with database.Store(settings.db_path, keyring) as first:
        for _ in range(6):
            first.register_failure("alice", threshold=3, base_s=30, max_s=300)
        assert first.get_lockout("alice").locked
    # Reopening the process is not a way out of the penalty box.
    with database.Store(settings.db_path, keyring) as second:
        assert second.get_lockout("alice").locked


def test_sessions(store):
    user = make_user(store)
    token = store.create_session(user.uid, ttl_s=60, factors=["face", "totp"])
    assert store.validate_session(token).uid == user.uid
    assert store.validate_session("some-other-token") is None
    assert store.revoke_session(token)
    assert store.validate_session(token) is None


def test_session_tokens_are_not_stored_verbatim(store, settings):
    user = make_user(store)
    token = store.create_session(user.uid, ttl_s=60, factors=["face"])
    assert token.encode() not in settings.db_path.read_bytes()


def test_expired_session_is_rejected(store):
    user = make_user(store)
    token = store.create_session(user.uid, ttl_s=1, factors=["face"])
    store.conn.execute(
        "UPDATE sessions SET expires_at = ?", (int(time.time()) - 1,)
    )
    assert store.validate_session(token) is None


def test_disabling_a_user_kills_their_sessions(store):
    user = make_user(store)
    token = store.create_session(user.uid, ttl_s=600, factors=["face"])
    store.set_status(user.uid, "disabled")
    assert store.validate_session(token) is None


def test_second_factor_secret_is_encrypted(store, settings):
    user = make_user(store)
    store.set_second_factor(user.uid, "JBSWY3DPEHPK3PXP")
    assert b"JBSWY3DPEHPK3PXP" not in settings.db_path.read_bytes()
    secret, counter = store.get_second_factor(user.uid)
    assert (secret, counter) == ("JBSWY3DPEHPK3PXP", 0)


def test_totp_counter_only_moves_forward(store):
    user = make_user(store)
    store.set_second_factor(user.uid, "JBSWY3DPEHPK3PXP")
    assert store.spend_totp_counter(user.uid, 100)
    assert not store.spend_totp_counter(user.uid, 100)
    assert not store.spend_totp_counter(user.uid, 99)
    assert store.spend_totp_counter(user.uid, 101)


def test_recovery_codes_are_single_use(store):
    user = make_user(store)
    store.store_recovery_codes(user.uid, ["aaaabbbbccccdddd", "eeeeffffgggghhhh"])
    assert store.recovery_codes_left(user.uid) == 2
    assert store.spend_recovery_code(user.uid, "aaaabbbbccccdddd")
    assert not store.spend_recovery_code(user.uid, "aaaabbbbccccdddd")
    assert store.recovery_codes_left(user.uid) == 1


def test_recovery_codes_are_hashed(store, settings):
    user = make_user(store)
    store.store_recovery_codes(user.uid, ["aaaabbbbccccdddd"])
    assert b"aaaabbbbccccdddd" not in settings.db_path.read_bytes()


def test_group_readable_database_is_refused(settings, keyring, store):
    store.close()
    settings.db_path.chmod(0o640)
    with pytest.raises(database.StoreError, match="readable"):
        database.Store(settings.db_path, keyring)
