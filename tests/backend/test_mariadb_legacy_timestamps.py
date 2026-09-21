"""Migration coverage for the legacy Synology VARCHAR timestamp schema."""
import json
from datetime import datetime
from pathlib import Path

import pymysql
import pytest

from reclive.database_dialect import DatabaseDialect, detect_database_dialect
from reclive import migrations
from reclive.migrations import MigrationError, run_migrations
from tests.backend.test_migrate import MIGRATIONS, direct_migration_settings


SOURCE_TIMESTAMPS = [
    "2026-09-20T12:30:00.123456+02:00",
    "2026-09-20T10:00:00.654321-05:00",
    "2026-09-20T23:45:00.000001+14:00",
    "2026-09-20T23:15:00.999999-12:00",
    "2026-09-20T09:45:00.000001+00:00",
]


def create_text_timestamp_rules(settings, timestamps=SOURCE_TIMESTAMPS):
    connection = pymysql.connect(**settings)
    try:
        if detect_database_dialect(connection) is not DatabaseDialect.MARIADB1011:
            pytest.skip("legacy text timestamp adapter is MariaDB-specific")
        with connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE push_rules (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    endpoint VARCHAR(2048) NOT NULL,
                    subscription_json LONGTEXT NOT NULL,
                    facility_id INT NOT NULL,
                    section_key VARCHAR(128) NOT NULL,
                    threshold INT NOT NULL,
                    created_at VARCHAR(64) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """)
            for timestamp, endpoint_key in zip(timestamps, ("a", "a", "b", "c", "b")):
                endpoint = f"https://push.reclive-notify.net/{endpoint_key}"
                cursor.execute(
                    "INSERT INTO push_rules (endpoint, subscription_json, facility_id, "
                    "section_key, threshold, created_at) VALUES (%s, %s, 1186, 'fitness', 40, %s)",
                    (endpoint, json.dumps({"endpoint": endpoint, "keys": {"auth": "a", "p256dh": "b"}}), timestamp),
                )
    finally:
        connection.close()


@pytest.mark.mysql
@pytest.mark.parametrize("checkpoint", [
    None,
    "mariadb_timestamps:after_shadow_column",
    "mariadb_timestamps:after_shadow_updates",
    "mariadb_timestamps:before_swap",
    "mariadb_timestamps:after_swap",
    "after_backfill_updates",
    "after_endpoint_drop",
])
def test_text_timestamp_cutover_preserves_instants_ordering_and_recovery(
    clean_test_database, monkeypatch, checkpoint,
):
    create_text_timestamp_rules(clean_test_database)
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "migration-test-key-with-at-least-thirty-two-bytes")
    settings = direct_migration_settings(clean_test_database)

    class Interrupted(RuntimeError):
        pass

    selected = None if checkpoint is None else f"0003_push_rule_lifecycle.sql:{checkpoint}"

    def interrupt(observed):
        if observed == selected:
            raise Interrupted("synthetic timestamp cutover interruption")

    if selected is not None:
        with pytest.raises(Interrupted):
            run_migrations(settings, MIGRATIONS, fault_injector=interrupt)
        connection = pymysql.connect(**clean_test_database)
        try:
            with connection.cursor() as cursor:
                cursor.execute("SHOW TABLES LIKE 'push_rules'")
                assert cursor.fetchone() is None
                cursor.execute("SELECT effective_checksum, started_at FROM schema_migration_attempts")
                original_attempt = cursor.fetchone()
        finally:
            connection.close()

    run_migrations(settings, MIGRATIONS)
    assert run_migrations(settings, MIGRATIONS) == []
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, created_at, status FROM push_rules ORDER BY id")
            assert cursor.fetchall() == (
                (1, datetime(2026, 9, 20, 10, 30, 0, 123456), "cancelled"),
                (2, datetime(2026, 9, 20, 15, 0, 0, 654321), "pending"),
                (3, datetime(2026, 9, 20, 9, 45, 0, 1), "cancelled"),
                (4, datetime(2026, 9, 21, 11, 15, 0, 999999), "pending"),
                (5, datetime(2026, 9, 20, 9, 45, 0, 1), "pending"),
            )
            cursor.execute("SHOW COLUMNS FROM push_rules")
            assert not any(row[0].startswith("_reclive_") for row in cursor.fetchall())
            if selected is not None:
                cursor.execute("SELECT effective_checksum, started_at FROM schema_migration_attempts")
                assert cursor.fetchone() == original_attempt
    finally:
        connection.close()


@pytest.mark.mysql
@pytest.mark.parametrize("invalid", [
    "2026-09-20T10:00:00", "2026-02-30T10:00:00+00:00", "private-invalid-timestamp",
    "2026-09-20T10:00:00+00:60",
])
def test_invalid_text_timestamp_fails_closed_without_rewriting_sources(
    clean_test_database, monkeypatch, invalid,
):
    timestamps = [*SOURCE_TIMESTAMPS[:4], invalid]
    create_text_timestamp_rules(clean_test_database, timestamps)
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "migration-test-key-with-at-least-thirty-two-bytes")
    with pytest.raises(RuntimeError, match="legacy created_at") as failure:
        run_migrations(direct_migration_settings(clean_test_database), MIGRATIONS)
    assert invalid not in str(failure.value)
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT created_at, endpoint_hash FROM _reclive_push_rules_cutover ORDER BY id")
            assert cursor.fetchall() == tuple((value, None) for value in timestamps)
            cursor.execute("SHOW TABLES LIKE 'push_rules'")
            assert cursor.fetchone() is None
    finally:
        connection.close()


@pytest.mark.mysql
def test_timestamp_helper_executes_hashed_snapshot_and_health_detects_changed_bytes(
    clean_test_database, monkeypatch, tmp_path,
):
    from server.reclive.health_repository import _migration_evidence

    create_text_timestamp_rules(clean_test_database)
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "migration-test-key-with-at-least-thirty-two-bytes")
    source = migrations.default_mariadb_timestamp_artifact_path().read_bytes()
    helper = tmp_path / "mariadb_legacy_timestamps.py"
    helper.write_bytes(source)
    monkeypatch.setattr(migrations, "default_mariadb_timestamp_artifact_path", lambda: helper)
    original_read = Path.read_bytes
    changed = False

    def replace_after_snapshot(path):
        nonlocal changed
        result = original_read(path)
        if path == helper and not changed:
            helper.write_bytes(b"raise AssertionError('changed helper executed')\n")
            changed = True
        return result

    monkeypatch.setattr(Path, "read_bytes", replace_after_snapshot)
    run_migrations(direct_migration_settings(clean_test_database), MIGRATIONS)
    assert changed
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT created_at, status FROM push_rules WHERE id = 2")
            assert cursor.fetchone() == (datetime(2026, 9, 20, 15, 0, 0, 654321), "pending")
        assert _migration_evidence(connection, MIGRATIONS) == "stale"
        with pytest.raises(MigrationError, match="checksum mismatch"):
            run_migrations(direct_migration_settings(clean_test_database), MIGRATIONS)
        helper.write_bytes(source)
        assert _migration_evidence(connection, MIGRATIONS) == "ready"
    finally:
        connection.close()


@pytest.mark.mysql
def test_changed_timestamp_helper_cannot_resume_a_recorded_attempt(
    clean_test_database, monkeypatch, tmp_path,
):
    create_text_timestamp_rules(clean_test_database)
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "migration-test-key-with-at-least-thirty-two-bytes")
    source = migrations.default_mariadb_timestamp_artifact_path().read_bytes()
    helper = tmp_path / "mariadb_legacy_timestamps.py"
    helper.write_bytes(source)
    monkeypatch.setattr(migrations, "default_mariadb_timestamp_artifact_path", lambda: helper)

    class Interrupted(RuntimeError):
        pass

    def interrupt(checkpoint):
        if checkpoint.endswith("mariadb_timestamps:after_shadow_column"):
            raise Interrupted

    settings = direct_migration_settings(clean_test_database)
    with pytest.raises(Interrupted):
        run_migrations(settings, MIGRATIONS, fault_injector=interrupt)
    helper.write_bytes(b"raise AssertionError('changed helper executed')\n")
    with pytest.raises(MigrationError, match="checksum mismatch"):
        run_migrations(settings, MIGRATIONS)
    helper.write_bytes(source)
    assert run_migrations(settings, MIGRATIONS) == ["0003_push_rule_lifecycle.sql", "0004_rate_limits.sql"]


@pytest.mark.mysql
def test_timestamp_conversion_refuses_to_drop_an_existing_created_at_index(
    clean_test_database, monkeypatch,
):
    create_text_timestamp_rules(clean_test_database)
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("CREATE INDEX legacy_created ON push_rules (created_at)")
        monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
        monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "migration-test-key-with-at-least-thirty-two-bytes")
        with pytest.raises(RuntimeError, match="legacy created_at index"):
            run_migrations(direct_migration_settings(clean_test_database), MIGRATIONS)
        with connection.cursor() as cursor:
            cursor.execute("SHOW INDEX FROM _reclive_push_rules_cutover WHERE Key_name = 'legacy_created'")
            assert cursor.fetchone() is not None
            cursor.execute("SELECT created_at FROM _reclive_push_rules_cutover ORDER BY id")
            assert cursor.fetchall() == tuple((value,) for value in SOURCE_TIMESTAMPS)
    finally:
        connection.close()
