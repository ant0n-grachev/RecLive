import hashlib
import hmac
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import pymysql
import pytest

from reclive import migrations as migration_module
from reclive.database_dialect import DatabaseDialect, detect_database_dialect
from reclive.migrations import (
    LOCK_NAME,
    MigrationError,
    MigrationSettings,
    migration_files,
    run_migrations,
)
from reclive.push_identity import (
    endpoint_hash,
    normalize_push_endpoint,
    rate_limit_subject_hash,
)

ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = ROOT / "server" / "migrations"
PUSH_BACKFILL = ROOT / "server" / "reclive" / "push_rule_backfill.py"
PUSH_IDENTITY = ROOT / "server" / "reclive" / "push_identity.py"


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.result = None

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def execute(self, statement: str, params=None) -> None:
        if statement == "SELECT VERSION(), @@version_comment":
            self.result = self.connection.server_identity
        elif statement.startswith("SELECT GET_LOCK"):
            self.connection.events.append("lock")
            self.result = (1,)
        elif statement.startswith("SELECT RELEASE_LOCK"):
            self.connection.events.append("release")
            if self.connection.release_error is not None:
                raise self.connection.release_error
            self.result = (1,)
        elif "CREATE TABLE IF NOT EXISTS schema_migrations" in statement:
            self.connection.events.append("create_history")
        elif "CREATE TABLE IF NOT EXISTS schema_migration_attempts" in statement:
            self.connection.events.append("create_attempt_history")
        elif statement.startswith("SELECT checksum FROM schema_migrations"):
            self.connection.events.append("read_checksum")
            selected_checksum = self.connection.existing_checksums.get(
                str(params[0]), self.connection.existing_checksum
            )
            if selected_checksum is None:
                self.result = None
            else:
                self.result = (selected_checksum,)
        elif statement.startswith("SELECT filename FROM schema_migrations"):
            self.connection.events.append("read_applied_filenames")
            self.result = tuple(
                (filename,) for filename in self.connection.applied_filenames
            )
        elif statement.startswith(
            "SELECT effective_checksum, started_at, key_identifier "
            "FROM schema_migration_attempts"
        ):
            self.connection.events.append("read_attempt")
            if self.connection.attempt_checksum is None:
                self.result = None
            else:
                self.result = (
                    self.connection.attempt_checksum,
                    self.connection.attempt_started_at,
                    self.connection.attempt_key_identifier,
                )
        elif statement.startswith(
            "SELECT COUNT(*) FROM information_schema.tables"
        ):
            self.connection.events.append("read_attempt_table")
            self.result = (int(self.connection.attempt_checksum is not None),)
        elif statement.startswith("INSERT INTO schema_migration_attempts"):
            self.connection.events.append("record_attempt")
            self.connection.attempt_checksum = params[1]
            self.connection.attempt_key_identifier = params[2]
            self.result = None
        elif statement.startswith("INSERT INTO schema_migrations"):
            self.connection.events.append("record_migration")
            self.connection.recorded_checksum = params[1]
        else:
            self.connection.events.append("execute_sql")
            self.connection.executed_statements.append(statement)

    def fetchone(self):
        return self.result

    def fetchall(self):
        return self.result or ()


class FakeConnection:
    def __init__(
        self,
        *,
        existing_checksum: str | None = None,
        existing_checksums: dict[str, str] | None = None,
        attempt_checksum: str | None = None,
        attempt_key_identifier: bytes | None = None,
        release_error: Exception | None = None,
        applied_filenames: tuple[str, ...] = (),
        server_identity: tuple[str, str] = ("8.4.7", "MySQL Community Server - GPL"),
    ) -> None:
        self.server_identity = server_identity
        self.existing_checksum = existing_checksum
        self.existing_checksums = existing_checksums or {}
        self.attempt_checksum = attempt_checksum
        self.attempt_key_identifier = attempt_key_identifier
        self.attempt_started_at = datetime(2026, 8, 31, 12, 0, 0)
        self.release_error = release_error
        self.applied_filenames = applied_filenames
        self.events: list[str] = []
        self.executed_statements: list[str] = []
        self.recorded_checksum: str | None = None
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        self.events.append("commit")

    def rollback(self) -> None:
        self.events.append("rollback")

    def close(self) -> None:
        self.events.append("close")
        self.closed = True


def fake_settings() -> MigrationSettings:
    return MigrationSettings(
        host="127.0.0.1",
        port=3306,
        user="reclive",
        password="synthetic-test-password",
        database="reclive_test",
        lock_timeout_seconds=1,
    )


@pytest.mark.parametrize("version", [
    "10.11.11-MariaDB",
    "10.11.11-MariaDB-0+deb12u1",
    "5.5.5-10.11.11-MariaDB",
])
def test_mariadb_executes_compatible_sql_and_binds_ledger_to_dialect(
    version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    migration = tmp_path / "0001_example.sql"
    original = b"CREATE TABLE example (id INT) COLLATE=utf8mb4_0900_ai_ci;\n"
    migration.write_bytes(original)
    connection = FakeConnection(server_identity=(version, "Source distribution"))
    monkeypatch.setattr(migration_module, "connect", lambda _: connection)

    assert run_migrations(fake_settings(), tmp_path) == [migration.name]
    assert connection.executed_statements == [
        "CREATE TABLE example (id INT) COLLATE=utf8mb4_unicode_ci"
    ]
    assert migration.read_bytes() == original
    assert connection.recorded_checksum != hashlib.sha256(original).hexdigest()
    assert len(connection.recorded_checksum) == 64

    mysql = FakeConnection(existing_checksum=connection.recorded_checksum)
    monkeypatch.setattr(migration_module, "connect", lambda _: mysql)
    with pytest.raises(MigrationError, match="checksum mismatch"):
        run_migrations(fake_settings(), tmp_path)
    assert mysql.executed_statements == []


def test_mariadb_execution_checksum_binds_unchanged_frozen_artifacts(tmp_path: Path) -> None:
    migration = tmp_path / "0003_push_rule_lifecycle.sql"
    migration.write_bytes(b"SELECT 'COLLATE=utf8mb4_0900_ai_ci';\n")
    artifact_paths = tuple(
        (name, tmp_path / name) for name in migration_module.PUSH_ARTIFACT_NAMES
    )
    for _name, path in artifact_paths:
        path.write_bytes(b"# frozen original\n")
    original = migration_module.snapshot_migration(migration, artifact_paths=artifact_paths)
    translated = migration_module.execution_snapshot(original, DatabaseDialect.MARIADB1011)
    assert translated.artifacts == original.artifacts
    assert translated.sql_bytes == b"SELECT 'COLLATE=utf8mb4_unicode_ci';\n"
    assert translated.checksum != original.checksum
    assert migration_module.execution_snapshot(original, DatabaseDialect.MYSQL8) is original

    for _name, path in artifact_paths:
        path.write_bytes(b"# edited frozen helper\n")
        edited = migration_module.snapshot_migration(migration, artifact_paths=artifact_paths)
        assert migration_module.execution_snapshot(
            edited, DatabaseDialect.MARIADB1011
        ).checksum != translated.checksum
        path.write_bytes(b"# frozen original\n")


def test_homebrew_mysql84_preserves_original_execution_and_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"CREATE TABLE example (id INT) COLLATE=utf8mb4_0900_ai_ci;\n"
    (tmp_path / "0001_example.sql").write_bytes(original)
    connection = FakeConnection(server_identity=("8.4.11", "Homebrew"))
    monkeypatch.setattr(migration_module, "connect", lambda _: connection)
    assert run_migrations(fake_settings(), tmp_path) == ["0001_example.sql"]
    assert connection.executed_statements == [original.decode().strip().rstrip(";")]
    assert connection.recorded_checksum == hashlib.sha256(original).hexdigest()


@pytest.mark.parametrize("table", ["push_rules", "_reclive_push_rules_cutover"])
def test_mariadb_hook_adapter_preserves_exclusive_lock_for_frozen_alter(table: str) -> None:
    from reclive import database_dialect

    connection = FakeConnection()
    adapter = database_dialect.migration_hook_connection
    wrapped = adapter(connection, DatabaseDialect.MARIADB1011)
    with wrapped.cursor() as cursor:
        cursor.execute(
            f"ALTER TABLE `{table}` MODIFY endpoint_hash BINARY(32) NOT NULL, "
            "ALGORITHM=INPLACE, LOCK=EXCLUSIVE"
        )
    assert connection.executed_statements == [
        f"ALTER TABLE `{table}` MODIFY endpoint_hash BINARY(32) NOT NULL, "
        "ALGORITHM=COPY, LOCK=EXCLUSIVE"
    ]
    assert adapter(connection, DatabaseDialect.MYSQL8) is connection


@pytest.mark.parametrize("identity", [
    ("10.6.20-MariaDB", "Source distribution"),
    ("11.4.4-MariaDB", "Source distribution"),
    ("5.7.44", "MySQL Community Server - GPL"),
    ("8.0.30-TiDB-v7.5.0", "TiDB Server"),
    ("8.4.7", "unexpected vendor"),
    ("", ""),
])
def test_unsupported_database_fails_before_any_migration_writes(
    identity: tuple[str, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "0001_example.sql").write_text("SELECT 1;\n")
    connection = FakeConnection(server_identity=identity)
    monkeypatch.setattr(migration_module, "connect", lambda _: connection)
    with pytest.raises(MigrationError, match="Unsupported database"):
        run_migrations(fake_settings(), tmp_path)
    assert connection.events == ["close"]


def test_mariadb_rejects_unrecognized_mysql_collation_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "0001_example.sql").write_text(
        "CREATE TABLE example (id INT) COLLATE=utf8mb4_0900_as_cs;\n"
    )
    connection = FakeConnection(server_identity=("10.11.11-MariaDB", "Source distribution"))
    monkeypatch.setattr(migration_module, "connect", lambda _: connection)
    with pytest.raises(MigrationError, match="Unsupported MariaDB migration collation"):
        run_migrations(fake_settings(), tmp_path)
    assert connection.events == ["close"]


def write_applied_prior_migration_stubs(
    migration_dir: Path,
) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for filename in ("0001_prior.sql", "0002_prior.sql"):
        contents = b"SELECT 1;\n"
        (migration_dir / filename).write_bytes(contents)
        checksums[filename] = hashlib.sha256(contents).hexdigest()
    return checksums


def run_migrate(
    settings: dict[str, object], migration_dir: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "GYM_DB_HOST": str(settings["host"]),
        "GYM_DB_PORT": str(settings["port"]),
        "GYM_DB_USER": str(settings["user"]),
        "GYM_DB_PASSWORD": str(settings["password"]),
        "GYM_DB_NAME": str(settings["database"]),
        "MIGRATION_LOCK_TIMEOUT_SECONDS": str(
            settings.get("lock_timeout_seconds", 5)
        ),
    }
    cutover_ready = settings.get("cutover_ready", True)
    if cutover_ready:
        env["PUSH_RULE_SCHEMA_CUTOVER_READY"] = "1"
    else:
        env.pop("PUSH_RULE_SCHEMA_CUTOVER_READY", None)
    hash_key = settings.get(
        "hash_key", "migration-test-key-with-at-least-thirty-two-bytes"
    )
    if hash_key:
        env["PUSH_ENDPOINT_HASH_KEY"] = str(hash_key)
    else:
        env.pop("PUSH_ENDPOINT_HASH_KEY", None)
    return subprocess.run(
        [sys.executable, "server/migrate.py", "--migrations-dir", str(migration_dir)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def create_legacy_push_rules(settings: dict[str, object]) -> None:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE push_rules (
                    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    endpoint VARCHAR(2048) NOT NULL,
                    subscription_json JSON NOT NULL,
                    facility_id INT NOT NULL,
                    section_key VARCHAR(80) NOT NULL,
                    threshold INT NOT NULL,
                    created_at DATETIME(6) NOT NULL
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
            cursor.execute(
                "CREATE INDEX idx_legacy_push_rules_endpoint "
                "ON push_rules (endpoint(191))"
            )
            rows = [
                (
                    "https://push.reclive-notify.net/a",
                    '{"endpoint":"https://push.reclive-notify.net/a",'
                    '"keys":{"p256dh":"x","auth":"y"}}',
                    1186,
                    "fitness",
                    40,
                    "2026-08-31 10:00:00.000000",
                ),
                (
                    "https://push.reclive-notify.net/a",
                    '{"endpoint":"https://push.reclive-notify.net/a",'
                    '"keys":{"p256dh":"x2","auth":"y2"}}',
                    1186,
                    "fitness",
                    40,
                    "2026-08-31 10:01:00.000000",
                ),
                (
                    "https://push.reclive-notify.net/b",
                    '{"endpoint":"https://push.reclive-notify.net/b",'
                    '"keys":{"p256dh":"z","auth":"q"}}',
                    1656,
                    "overall",
                    55,
                    "2026-08-31 10:02:00.000000",
                ),
                (
                    "https://push.reclive-notify.net/c",
                    '{"endpoint":"https://push.reclive-notify.net/c",'
                    '"keys":{"p256dh":"bad","auth":"bad"}}',
                    1186,
                    "fitness",
                    101,
                    "2026-08-31 10:03:00.000000",
                ),
                (
                    "https://push.reclive-notify.net/d",
                    '{"endpoint":"https://push.reclive-notify.net/other",'
                    '"keys":{"p256dh":"mismatch","auth":"mismatch"}}',
                    1186,
                    "fitness",
                    1000,
                    "2026-08-31 10:04:00.000000",
                ),
            ]
            cursor.executemany(
                "INSERT INTO push_rules "
                "(endpoint, subscription_json, facility_id, section_key, "
                "threshold, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                rows,
            )
        connection.commit()
    finally:
        connection.close()


def direct_migration_settings(settings: dict[str, object]) -> MigrationSettings:
    return MigrationSettings(
        host=str(settings["host"]),
        port=int(settings["port"]),
        user=str(settings["user"]),
        password=str(settings["password"]),
        database=str(settings["database"]),
        lock_timeout_seconds=5,
    )


def add_second_legacy_endpoint_index(settings: dict[str, object]) -> None:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE INDEX idx_legacy_push_rules_endpoint_facility "
                "ON push_rules (endpoint(191), facility_id)"
            )
    finally:
        connection.close()


def drop_all_disposable_tables(settings: dict[str, object]) -> None:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            table_names = [str(row[0]) for row in cursor.fetchall()]
            for table_name in table_names:
                if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
                    raise RuntimeError("unsafe disposable test table name")
                cursor.execute(f"DROP TABLE `{table_name}`")
    finally:
        connection.close()


def normalized_push_rule_contract(settings: dict[str, object]) -> dict[str, object]:
    connection = pymysql.connect(**settings)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT effective_checksum, started_at "
                "FROM schema_migration_attempts "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            attempt_checksum, started_at = cursor.fetchone()
            cursor.execute(
                "SELECT checksum FROM schema_migrations "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            recorded_checksum = cursor.fetchone()[0]
            cursor.execute(
                "SELECT id, HEX(endpoint_hash), subscription_json, facility_id, "
                "section_key, threshold, created_at, "
                "TIMESTAMPDIFF(MICROSECOND, %s, expires_at), status, claimed_at, "
                "sent_at, TIMESTAMPDIFF(MICROSECOND, %s, finalized_at), "
                "failure_code FROM push_rules ORDER BY id",
                (started_at, started_at),
            )
            rows = cursor.fetchall()
            cursor.execute("SHOW COLUMNS FROM push_rules")
            columns = cursor.fetchall()
            cursor.execute(
                "SELECT index_name, seq_in_index, column_name, non_unique "
                "FROM information_schema.statistics "
                "WHERE table_schema = DATABASE() AND table_name = 'push_rules' "
                "ORDER BY index_name, seq_in_index"
            )
            indexes = cursor.fetchall()
            cursor.execute(
                "SELECT constraint_name, constraint_type "
                "FROM information_schema.table_constraints "
                "WHERE table_schema = DATABASE() AND table_name = 'push_rules' "
                "ORDER BY constraint_name"
            )
            constraints = cursor.fetchall()
    finally:
        connection.close()
    return {
        "attempt_checksum": attempt_checksum,
        "recorded_checksum": recorded_checksum,
        "rows": rows,
        "columns": columns,
        "indexes": indexes,
        "constraints": constraints,
    }


def table_column_contract(
    cursor, table: str, *, mariadb: bool = False,
) -> dict[str, tuple[object, ...]]:
    cursor.execute(
        "SELECT column_name, LOWER(column_type), is_nullable, "
        "LOWER(COALESCE(column_default, '<null>')), LOWER(extra), "
        "datetime_precision FROM information_schema.columns "
        "WHERE table_schema = DATABASE() AND table_name = %s "
        "ORDER BY ordinal_position",
        (table,),
    )
    result = {}
    for name, column_type, nullable, default, extra, precision in cursor.fetchall():
        if mariadb:
            # MariaDB retains integer display widths and represents a SQL NULL
            # default as text. Neither changes the stored value contract.
            column_type = re.sub(r"\b(bigint|int)\(\d+\)", r"\1", column_type)
            if default == "null":
                default = "<null>"
        result[str(name)] = (column_type, nullable, default, extra, precision)
    return result


def table_index_contract(cursor, table: str) -> dict[str, tuple[str, ...]]:
    cursor.execute(
        "SELECT index_name, column_name FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = %s "
        "ORDER BY index_name, seq_in_index",
        (table,),
    )
    indexes: dict[str, list[str]] = {}
    for index_name, column_name in cursor.fetchall():
        indexes.setdefault(str(index_name), []).append(str(column_name))
    return {name: tuple(columns) for name, columns in indexes.items()}


def test_0003_uses_canonical_endpoint_hmac_and_a_domain_separated_rate_limit_subject(
    monkeypatch,
) -> None:
    key = "migration-test-key-with-at-least-thirty-two-bytes"
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", key)
    expected = hmac.new(
        key.encode("utf-8"),
        b"reclive:push:endpoint:v1\x00https://push.reclive-notify.net/a?x=1",
        sha256,
    ).digest()
    assert endpoint_hash(" HTTPS://PUSH.RECLIVE-NOTIFY.NET:443/a?x=1 ") == expected
    assert rate_limit_subject_hash(
        "endpoint", "https://push.reclive-notify.net/a?x=1"
    ) != expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://PUSH.RECLIVE-NOTIFY.NET", "https://push.reclive-notify.net/"),
        (
            "https://push.reclive-notify.net.:443/a?x=1",
            "https://push.reclive-notify.net/a?x=1",
        ),
        (
            "https://push.reclive-notify.net:8443/a?x=1",
            "https://push.reclive-notify.net:8443/a?x=1",
        ),
        ("https://bücher.de/a", "https://xn--bcher-kva.de/a"),
        (
            "https://[2606:4700:4700:0:0:0:0:1111]:443/a",
            "https://[2606:4700:4700::1111]/a",
        ),
        ("https://8.8.8.8/a", "https://8.8.8.8/a"),
    ],
)
def test_0003_normalizes_equivalent_https_authorities(
    value: str, expected: str
) -> None:
    assert normalize_push_endpoint(value) == expected


def test_0003_aliases_share_endpoint_and_rate_limit_identities(monkeypatch) -> None:
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "migration-test-key-with-at-least-thirty-two-bytes",
    )
    aliases = [
        ("https://bücher.de/a", "https://xn--bcher-kva.de/a"),
        (
            "https://[2606:4700:4700:0:0:0:0:1111]/a",
            "https://[2606:4700:4700::1111]/a",
        ),
        (
            "https://push.reclive-notify.net./a",
            "https://PUSH.RECLIVE-NOTIFY.NET:443/a",
        ),
    ]
    for first, second in aliases:
        assert endpoint_hash(first) == endpoint_hash(second)
        assert rate_limit_subject_hash(
            "endpoint", first
        ) == rate_limit_subject_hash("endpoint", second)


@pytest.mark.parametrize(
    "value",
    [
        "http://example.test/a",
        "https://",
        "https://user@example.test/a",
        "https://:password@example.test/a",
        "https://example.test:/a",
        "https://example.test:0/a",
        "https://example.test:65536/a",
        "https://example.test:not-a-port/a",
        "https://example.test/a#",
        "https://example.test/a#fragment",
        "https://[2001:db8::1/a",
        "https://2001:db8::1/a",
        "https://[fe80::1%25eth0]/a",
        "https://[192.0.2.1]/a",
        "https://-bad.example/a",
        "https://bad-.example/a",
        "https://bad..example/a",
        "https://bad_name.example/a",
        f"https://{'a' * 64}.example/a",
    ],
)
def test_0003_rejects_noncanonical_or_malformed_endpoint_authorities(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_push_endpoint(value)


def test_0003_enforces_trimmed_input_and_canonical_output_byte_limits() -> None:
    prefix = "https://push.reclive-notify.net/"
    exact = prefix + ("a" * (2048 - len(prefix.encode("utf-8"))))
    assert len(exact.encode("utf-8")) == 2048
    assert normalize_push_endpoint(f"  {exact}  ") == exact

    with pytest.raises(ValueError, match="too long"):
        normalize_push_endpoint(exact + "a")

    expanding_prefix = "https://éxample.de/"
    expanding = expanding_prefix + (
        "a" * (2048 - len(expanding_prefix.encode("utf-8")))
    )
    assert len(expanding.encode("utf-8")) == 2048
    with pytest.raises(ValueError, match="too long"):
        normalize_push_endpoint(expanding)


@pytest.mark.parametrize(
    "legacy_host",
    [
        "2130706433",
        "017700000001",
        "0x7f000001",
        "0x7f.0.0.1",
        "0177.0.0.1",
    ],
)
def test_0003_rejects_legacy_numeric_ipv4_aliases_for_normalization_and_hashing(
    monkeypatch, legacy_host
) -> None:
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "migration-test-key-with-at-least-thirty-two-bytes",
    )
    endpoint = f"https://{legacy_host}/push"

    with pytest.raises(ValueError):
        normalize_push_endpoint(endpoint)
    with pytest.raises(ValueError):
        endpoint_hash(endpoint)


@pytest.mark.parametrize("host", ["8.8.8.8", "push.reclive-notify.net"])
def test_0003_keeps_global_ip_literals_and_public_dns_hosts(
    monkeypatch, host: str
) -> None:
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "migration-test-key-with-at-least-thirty-two-bytes",
    )
    assert normalize_push_endpoint(f"https://{host}/push") == f"https://{host}/push"
    assert endpoint_hash(f"https://{host}/push") == endpoint_hash(
        f"https://{host}/push"
    )


@pytest.mark.parametrize(
    "host",
    [
        "localhost",
        "push.localhost",
        "intranet",
        "push.internal",
        "push.local",
        "push.localdomain",
        "push.lan",
        "push.home",
        "push.home.arpa",
        "push.corp",
        "push.intranet",
        "push.private",
        "push.test",
        "push.invalid",
        "push.example",
        "push.example.com",
        "127.0.0.1",
        "10.0.0.1",
        "169.254.1.1",
        "224.0.0.1",
        "0.0.0.0",
        "192.0.2.1",
        "[::1]",
        "[fc00::1]",
        "[fe80::1]",
        "[ff02::1]",
        "[::]",
        "[2001:db8::1]",
    ],
)
def test_0003_rejects_statically_unsafe_push_destinations(
    monkeypatch, host: str
) -> None:
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "migration-test-key-with-at-least-thirty-two-bytes",
    )
    endpoint = f"https://{host}/push"

    with pytest.raises(ValueError):
        normalize_push_endpoint(endpoint)
    with pytest.raises(ValueError):
        endpoint_hash(endpoint)


@pytest.mark.mysql
def test_0003_creates_finalized_clean_push_rule_contract(
    clean_test_database, tmp_path
) -> None:
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        maria = detect_database_dialect(connection) is DatabaseDialect.MARIADB1011
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM push_rules")
            columns = {row[0]: row for row in cursor.fetchall()}
            assert {
                "id",
                "endpoint_hash",
                "active_identity",
                "subscription_json",
                "facility_id",
                "section_key",
                "threshold",
                "created_at",
                "expires_at",
                "status",
                "claimed_at",
                "sent_at",
                "finalized_at",
                "failure_code",
            } <= set(columns)
            assert columns["endpoint_hash"][2] == "NO"
            assert columns["threshold"][1:6] == (
                "tinyint(3) unsigned" if maria else "tinyint unsigned",
                "NO",
                "",
                None,
                "",
            )
            assert "endpoint" not in columns
            cursor.execute(
                "SHOW INDEX FROM push_rules "
                "WHERE Key_name = 'uq_push_rules_identity'"
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SHOW INDEX FROM push_rules WHERE Key_name = 'idx_push_rules_pending'"
            )
            assert cursor.fetchone() is not None
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_backfills_legacy_rows_cancels_deterministic_duplicates_and_removes_raw_endpoint(
    clean_test_database, tmp_path
) -> None:
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        maria = detect_database_dialect(connection) is DatabaseDialect.MARIADB1011
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, endpoint_hash, subscription_json, status, finalized_at "
                "FROM push_rules ORDER BY id"
            )
            rules = cursor.fetchall()
            assert len(rules) == 5
            expected_hash = hmac.new(
                b"migration-test-key-with-at-least-thirty-two-bytes",
                b"reclive:push:endpoint:v1\x00https://push.reclive-notify.net/a",
                sha256,
            ).digest()
            assert (
                rules[0][0] == 1
                and rules[0][1] == expected_hash
                and rules[0][3] == "cancelled"
                and rules[0][4] is not None
            )
            assert "https://push.reclive-notify.net/a" in rules[0][2]
            assert (
                rules[1][0] == 2
                and rules[1][3] == "pending"
                and rules[1][4] is None
            )
            assert rules[3][3] == "cancelled" and rules[4][3] == "cancelled"
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'threshold'")
            assert cursor.fetchone()[1:6] == (
                "tinyint(3) unsigned" if maria else "tinyint unsigned",
                "NO",
                "",
                None,
                "",
            )
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint'")
            assert cursor.fetchone() is None
            cursor.execute(
                "SHOW INDEX FROM push_rules "
                "WHERE Key_name = 'idx_legacy_push_rules_endpoint'"
            )
            assert cursor.fetchone() is None
            cursor.execute(
                "UPDATE push_rules SET status = 'sent', "
                "sent_at = UTC_TIMESTAMP(6), finalized_at = UTC_TIMESTAMP(6) "
                "WHERE id = 2"
            )
            cursor.execute(
                "INSERT INTO push_rules "
                "(endpoint_hash, subscription_json, facility_id, section_key, "
                "threshold, expires_at, status) "
                "VALUES (%s, %s, %s, %s, %s, "
                "DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 24 HOUR), 'pending')",
                (
                    expected_hash,
                    '{"endpoint":"https://push.reclive-notify.net/a",'
                    '"keys":{"p256dh":"new","auth":"new"}}',
                    1186,
                    "fitness",
                    40,
                ),
            )
            with pytest.raises(pymysql.err.IntegrityError):
                cursor.execute(
                    "INSERT INTO push_rules "
                    "(endpoint_hash, subscription_json, facility_id, section_key, "
                    "threshold, expires_at, status) "
                    "VALUES (%s, %s, %s, %s, %s, "
                    "DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 24 HOUR), 'pending')",
                    (
                        expected_hash,
                        '{"endpoint":"https://push.reclive-notify.net/a",'
                        '"keys":{"p256dh":"newer","auth":"newer"}}',
                        1186,
                        "fitness",
                        40,
                    ),
                )
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_backfill_cancels_an_unsafe_legacy_destination(
    clean_test_database, tmp_path
) -> None:
    create_legacy_push_rules(clean_test_database)
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO push_rules "
                "(endpoint, subscription_json, facility_id, section_key, "
                "threshold, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    "https://127.0.0.1/private",
                    '{"endpoint":"https://127.0.0.1/private",'
                    '"keys":{"p256dh":"legacy","auth":"legacy"}}',
                    1186,
                    "fitness",
                    45,
                    "2026-08-31 10:05:00.000000",
                ),
            )
    finally:
        connection.close()

    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT status, failure_code, LENGTH(endpoint_hash) "
                "FROM push_rules WHERE id = 6"
            )
            assert cursor.fetchone() == (
                "cancelled",
                "migration_invalid_subscription",
                32,
            )
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint'")
            assert cursor.fetchone() is None
    finally:
        connection.close()


@pytest.mark.mysql
@pytest.mark.parametrize(
    "hash_key, forbidden_output",
    [
        (None, None),
        ("secret-short-key", "secret-short-key"),
    ],
)
def test_0003_fails_closed_before_recording_when_legacy_rows_exist_without_hash_key(
    clean_test_database, tmp_path, hash_key, forbidden_output
) -> None:
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    environment = {**clean_test_database, "hash_key": hash_key}
    result = run_migrate(environment, migration_dir)
    assert result.returncode == 1
    assert "PUSH_ENDPOINT_HASH_KEY" in result.stderr
    if forbidden_output is not None:
        assert forbidden_output not in result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT filename FROM schema_migrations "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            assert cursor.fetchone() is None
            cursor.execute("SHOW TABLES LIKE 'push_rules'")
            assert cursor.fetchone() is None
            cursor.execute(
                "SHOW COLUMNS FROM _reclive_push_rules_cutover LIKE 'endpoint'"
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SHOW COLUMNS FROM _reclive_push_rules_cutover "
                "LIKE 'endpoint_hash'"
            )
            assert cursor.fetchone() is None
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_requires_explicit_cutover_gate_before_mutating_legacy_push_rules(
    clean_test_database, tmp_path
) -> None:
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    environment = {**clean_test_database, "cutover_ready": False}

    result = run_migrate(environment, migration_dir)

    assert result.returncode == 1
    assert "PUSH_RULE_SCHEMA_CUTOVER_READY" in result.stderr
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM push_rules")
            assert [row[0] for row in cursor.fetchall()] == [
                "id",
                "endpoint",
                "subscription_json",
                "facility_id",
                "section_key",
                "threshold",
                "created_at",
            ]
            cursor.execute("SHOW TABLES LIKE 'schema_migration_attempts'")
            assert cursor.fetchone() is None
            cursor.execute(
                "SELECT filename FROM schema_migrations "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            assert cursor.fetchone() is None
            cursor.execute(
                "INSERT INTO push_rules "
                "(endpoint, subscription_json, facility_id, section_key, "
                "threshold, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    "https://push.reclive-notify.net/still-legacy",
                    '{"endpoint":"https://push.reclive-notify.net/still-legacy",'
                    '"keys":{"p256dh":"legacy","auth":"legacy"}}',
                    1186,
                    "fitness",
                    45,
                    "2026-08-31 10:05:00.000000",
                ),
            )
            cursor.execute("SELECT COUNT(*) FROM push_rules")
            assert cursor.fetchone()[0] == 6
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_replays_completed_conversion_without_changing_retained_rows(
    clean_test_database, tmp_path
) -> None:
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    first = run_migrate(clean_test_database, migration_dir)
    assert first.returncode == 0, first.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, endpoint_hash, subscription_json, facility_id, "
                "section_key, threshold, created_at, expires_at, status, "
                "claimed_at, sent_at, finalized_at, failure_code "
                "FROM push_rules ORDER BY id"
            )
            before = cursor.fetchall()
            cursor.execute(
                "DELETE FROM schema_migrations WHERE filename IN "
                "('0003_push_rule_lifecycle.sql', '0004_rate_limits.sql')"
            )

        second = run_migrate(clean_test_database, migration_dir)
        assert second.returncode == 0, second.stderr

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, endpoint_hash, subscription_json, facility_id, "
                "section_key, threshold, created_at, expires_at, status, "
                "claimed_at, sent_at, finalized_at, failure_code "
                "FROM push_rules ORDER BY id"
            )
            assert cursor.fetchall() == before
    finally:
        connection.close()


def assert_legacy_push_rule_dml_is_blocked(
    settings: dict[str, object]
) -> None:
    endpoint = "https://push.reclive-notify.net/stale"
    subscription = (
        '{"endpoint":"https://push.reclive-notify.net/stale",'
        '"keys":{"p256dh":"stale","auth":"stale"}}'
    )
    statements = [
        (
            "INSERT INTO push_rules "
            "(endpoint, subscription_json, facility_id, section_key, threshold, "
            "created_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                endpoint,
                subscription,
                1186,
                "fitness",
                50,
                datetime(2026, 8, 31, 10, 6),
            ),
        ),
        (
            "INSERT INTO push_rules "
            "(endpoint, subscription_json, facility_id, section_key, threshold, "
            "created_at) VALUES (%s, %s, %s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE threshold = VALUES(threshold)",
            (
                endpoint,
                subscription,
                1186,
                "fitness",
                50,
                datetime(2026, 8, 31, 10, 6),
            ),
        ),
        (
            "UPDATE push_rules SET threshold = 50 WHERE endpoint = %s",
            (endpoint,),
        ),
        ("DELETE FROM push_rules WHERE endpoint = %s", (endpoint,)),
    ]
    writer = pymysql.connect(**settings)
    try:
        with writer.cursor() as cursor:
            for statement, params in statements:
                with pytest.raises(pymysql.MySQLError):
                    cursor.execute(statement, params)
    finally:
        writer.close()


@pytest.mark.mysql
@pytest.mark.parametrize(
    "fault_checkpoint",
    [
        "0003_push_rule_lifecycle.sql:after_sql:0",
        "0003_push_rule_lifecycle.sql:after_sql:4",
        "0003_push_rule_lifecycle.sql:after_backfill_updates",
        "0003_push_rule_lifecycle.sql:before_endpoint_hash_not_null",
        "0003_push_rule_lifecycle.sql:after_endpoint_hash_not_null",
        "0003_push_rule_lifecycle.sql:after_endpoint_index_drop",
        "0003_push_rule_lifecycle.sql:after_endpoint_drop",
        "0003_push_rule_lifecycle.sql:after_final_columns",
        "0003_push_rule_lifecycle.sql:after_constraint:chk_push_rules_threshold",
        "0003_push_rule_lifecycle.sql:after_index:uq_push_rules_identity",
        "0003_push_rule_lifecycle.sql:before_cutover_barrier_release",
        "0003_push_rule_lifecycle.sql:after_cutover_barrier_release",
        "0003_push_rule_lifecycle.sql:before_record",
    ],
)
def test_0003_blocks_all_legacy_dml_through_finalization_and_resumes_cleanly(
    clean_test_database, tmp_path, monkeypatch, fault_checkpoint
) -> None:
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "migration-test-key-with-at-least-thirty-two-bytes",
    )
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    settings = direct_migration_settings(clean_test_database)

    class InjectedMigrationFault(RuntimeError):
        pass

    injected = False

    def block_stale_writer_then_stop(checkpoint: str) -> None:
        nonlocal injected
        if checkpoint != fault_checkpoint or injected:
            return
        injected = True
        assert_legacy_push_rule_dml_is_blocked(clean_test_database)
        raise InjectedMigrationFault(checkpoint)

    with pytest.raises(InjectedMigrationFault, match=re.escape(fault_checkpoint)):
        run_migrations(
            settings,
            migration_dir,
            fault_injector=block_stale_writer_then_stop,
        )
    assert injected is True

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = DATABASE() AND table_name IN "
                "('push_rules', '_reclive_push_rules_cutover') "
                "ORDER BY table_name"
            )
            table_names = [row[0] for row in cursor.fetchall()]
            if fault_checkpoint.endswith("after_cutover_barrier_release") or (
                fault_checkpoint.endswith("before_record")
            ):
                assert table_names == ["push_rules"]
            else:
                assert table_names == ["_reclive_push_rules_cutover"]
    finally:
        connection.close()

    assert run_migrations(settings, migration_dir) == [
        "0003_push_rule_lifecycle.sql",
        "0004_rate_limits.sql",
    ]
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE '_reclive_push_rules_cutover'")
            assert cursor.fetchone() is None
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint'")
            assert cursor.fetchone() is None
            cursor.execute("SELECT COUNT(*) FROM push_rules")
            assert cursor.fetchone()[0] == 5
            cursor.execute(
                "UPDATE push_rules SET status = 'cancelled', "
                "finalized_at = UTC_TIMESTAMP(6) WHERE id = 2"
            )
            assert cursor.rowcount == 1
    finally:
        connection.close()
    assert_legacy_push_rule_dml_is_blocked(clean_test_database)


@pytest.mark.mysql
@pytest.mark.parametrize(
    "fault_checkpoint",
    [
        "0003_push_rule_lifecycle.sql:before_endpoint_hash_not_null",
        "0003_push_rule_lifecycle.sql:after_endpoint_hash_not_null",
    ],
)
def test_0003_incomplete_attempt_is_bound_to_the_configured_hash_key(
    clean_test_database, tmp_path, monkeypatch, fault_checkpoint
) -> None:
    key_a = "migration-key-A-with-at-least-thirty-two-bytes"
    key_b = "migration-key-B-with-at-least-thirty-two-bytes"
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", key_a)
    create_legacy_push_rules(clean_test_database)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    settings = direct_migration_settings(clean_test_database)

    class InjectedMigrationFault(RuntimeError):
        pass

    injected = False

    def fail_attempt(checkpoint: str) -> None:
        nonlocal injected
        if checkpoint != fault_checkpoint or injected:
            return
        injected = True
        raise InjectedMigrationFault(checkpoint)

    with pytest.raises(InjectedMigrationFault):
        run_migrations(
            settings,
            migration_dir,
            fault_injector=fail_attempt,
        )
    assert injected is True

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT key_identifier FROM schema_migration_attempts "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            assert cursor.fetchone()[0] == hmac.new(
                key_a.encode("utf-8"),
                b"reclive:push:migration-key-id:v1",
                sha256,
            ).digest()
            cursor.execute("SHOW TABLES LIKE 'push_rules'")
            assert cursor.fetchone() is None
            cursor.execute(
                "SHOW COLUMNS FROM _reclive_push_rules_cutover LIKE 'endpoint'"
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SHOW COLUMNS FROM _reclive_push_rules_cutover "
                "LIKE 'endpoint_hash'"
            )
            nullable_before_mismatch = cursor.fetchone()[2]
            cursor.execute("SELECT COUNT(*) FROM _reclive_push_rules_cutover")
            rows_before_mismatch = cursor.fetchone()[0]
    finally:
        connection.close()

    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", key_b)
    with pytest.raises(MigrationError) as mismatch:
        run_migrations(settings, migration_dir)
    message = str(mismatch.value)
    assert "PUSH_ENDPOINT_HASH_KEY" in message
    assert "0003" in message
    assert key_a not in message
    assert key_b not in message

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'push_rules'")
            assert cursor.fetchone() is None
            cursor.execute(
                "SHOW COLUMNS FROM _reclive_push_rules_cutover LIKE 'endpoint'"
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SHOW COLUMNS FROM _reclive_push_rules_cutover "
                "LIKE 'endpoint_hash'"
            )
            assert cursor.fetchone()[2] == nullable_before_mismatch
            cursor.execute("SELECT COUNT(*) FROM _reclive_push_rules_cutover")
            assert cursor.fetchone()[0] == rows_before_mismatch
            cursor.execute(
                "SELECT COUNT(*) FROM schema_migrations "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()

    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", key_a)
    assert run_migrations(settings, migration_dir) == [
        "0003_push_rule_lifecycle.sql",
        "0004_rate_limits.sql",
    ]
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW COLUMNS FROM push_rules LIKE 'endpoint'")
            assert cursor.fetchone() is None
            cursor.execute("SELECT COUNT(*) FROM push_rules")
            assert cursor.fetchone()[0] == rows_before_mismatch
    finally:
        connection.close()


@pytest.mark.mysql
def test_0003_clean_empty_migration_does_not_require_or_bind_a_hash_key(
    clean_test_database, tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.delenv("PUSH_ENDPOINT_HASH_KEY", raising=False)
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)

    assert run_migrations(
        direct_migration_settings(clean_test_database), migration_dir
    ) == [
        "0001_core_history.sql",
        "0002_snapshot_and_ingestion.sql",
        "0003_push_rule_lifecycle.sql",
        "0004_rate_limits.sql",
    ]

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT key_identifier FROM schema_migration_attempts "
                "WHERE filename = '0003_push_rule_lifecycle.sql'"
            )
            assert cursor.fetchone()[0] is None
    finally:
        connection.close()


@pytest.mark.mysql
@pytest.mark.parametrize(
    "fault_checkpoint",
    [
        "0003_push_rule_lifecycle.sql:after_sql:4",
        "0003_push_rule_lifecycle.sql:after_backfill_updates",
        "0003_push_rule_lifecycle.sql:after_endpoint_hash_not_null",
        "0003_push_rule_lifecycle.sql:after_endpoint_index_drop",
        "0003_push_rule_lifecycle.sql:after_endpoint_drop",
        "0003_push_rule_lifecycle.sql:after_final_columns",
        "0003_push_rule_lifecycle.sql:after_constraint:chk_push_rules_threshold",
        "0003_push_rule_lifecycle.sql:after_index:uq_push_rules_identity",
        "0003_push_rule_lifecycle.sql:before_record",
    ],
)
def test_0003_fault_resume_matches_uninterrupted_rows_and_metadata(
    clean_test_database, tmp_path, monkeypatch, fault_checkpoint
) -> None:
    monkeypatch.setenv("PUSH_RULE_SCHEMA_CUTOVER_READY", "1")
    monkeypatch.setenv(
        "PUSH_ENDPOINT_HASH_KEY",
        "migration-test-key-with-at-least-thirty-two-bytes",
    )
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    settings = direct_migration_settings(clean_test_database)
    create_legacy_push_rules(clean_test_database)
    add_second_legacy_endpoint_index(clean_test_database)

    class InjectedMigrationFault(RuntimeError):
        pass

    injected = False

    def fail_once(checkpoint: str) -> None:
        nonlocal injected
        if checkpoint == fault_checkpoint and not injected:
            injected = True
            raise InjectedMigrationFault(checkpoint)

    with pytest.raises(InjectedMigrationFault, match=re.escape(fault_checkpoint)):
        run_migrations(settings, migration_dir, fault_injector=fail_once)
    assert injected is True
    assert run_migrations(settings, migration_dir) == [
        "0003_push_rule_lifecycle.sql",
        "0004_rate_limits.sql",
    ]
    resumed_contract = normalized_push_rule_contract(clean_test_database)

    drop_all_disposable_tables(clean_test_database)
    create_legacy_push_rules(clean_test_database)
    add_second_legacy_endpoint_index(clean_test_database)
    assert run_migrations(settings, migration_dir) == [
        "0001_core_history.sql",
        "0002_snapshot_and_ingestion.sql",
        "0003_push_rule_lifecycle.sql",
        "0004_rate_limits.sql",
    ]
    uninterrupted_contract = normalized_push_rule_contract(clean_test_database)

    assert resumed_contract == uninterrupted_contract


@pytest.mark.mysql
def test_runner_applies_once_then_rejects_an_applied_checksum_change(
    clean_test_database, tmp_path
) -> None:
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "CREATE TABLE location_history (id BIGINT UNSIGNED NOT NULL "
                "AUTO_INCREMENT PRIMARY KEY, location_id INT NOT NULL, "
                "last_updated DATETIME(6) NULL) ENGINE=InnoDB"
            )
            cursor.execute(
                "INSERT INTO location_history (location_id, last_updated) "
                "VALUES (1186, '2026-08-31 10:00:00.123456')"
            )
        connection.commit()
    finally:
        connection.close()
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    for later_migration in migration_dir.glob("000[2-9]_*.sql"):
        later_migration.unlink()

    first = run_migrate(clean_test_database, migration_dir)
    assert first.returncode == 0, first.stderr
    second = run_migrate(clean_test_database, migration_dir)
    assert second.returncode == 0, second.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT filename FROM schema_migrations ORDER BY filename")
            assert [row[0] for row in cursor.fetchall()] == ["0001_core_history.sql"]
            cursor.execute(
                "SHOW INDEX FROM location_history "
                "WHERE Key_name = 'idx_location_history_location_fetched'"
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SHOW INDEX FROM location_history "
                "WHERE Key_name = "
                "'idx_location_history_location_source_updated'"
            )
            assert cursor.fetchone() is not None
            cursor.execute(
                "SELECT source_updated_at FROM location_history WHERE location_id = 1186"
            )
            assert (
                cursor.fetchone()[0].strftime("%Y-%m-%d %H:%M:%S.%f")
                == "2026-08-31 10:00:00.123456"
            )
    finally:
        connection.close()

    core_history = migration_dir / "0001_core_history.sql"
    core_history.write_text(
        core_history.read_text(encoding="utf-8") + "\nSELECT 1;\n",
        encoding="utf-8",
    )
    changed = run_migrate(clean_test_database, migration_dir)
    assert changed.returncode == 1
    assert "checksum mismatch" in changed.stderr.lower()


def test_migration_files_include_only_exact_numeric_sql_names(tmp_path) -> None:
    expected = [tmp_path / "0001_alpha.sql", tmp_path / "0002_beta.sql"]
    for path in [
        *expected,
        tmp_path / "001_short.sql",
        tmp_path / "0003_CAPS.sql",
        tmp_path / "0004-dash.sql",
        tmp_path / "0005_wrong.SQL",
        tmp_path / "notes.sql",
    ]:
        path.write_text("SELECT 1;\n", encoding="utf-8")

    assert migration_files(tmp_path) == expected


@pytest.mark.parametrize(
    "filenames",
    [
        ("0001_alpha.sql", "0001_duplicate.sql"),
        ("0002_starts_late.sql",),
        ("0001_alpha.sql", "0003_gap.sql"),
    ],
)
def test_migration_files_reject_duplicate_or_noncontiguous_versions(
    tmp_path, filenames
) -> None:
    for filename in filenames:
        (tmp_path / filename).write_text("SELECT 1;\n", encoding="utf-8")

    with pytest.raises(MigrationError, match="contiguous.*0001"):
        migration_files(tmp_path)


def test_runner_rejects_missing_applied_migration_before_executing_later_work(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "0001_current.sql").write_text("SELECT 1;\n", encoding="utf-8")
    (tmp_path / "0002_later.sql").write_text("SELECT 2;\n", encoding="utf-8")
    connection = FakeConnection(applied_filenames=("0001_removed.sql",))
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    with pytest.raises(MigrationError, match="missing applied migration file"):
        run_migrations(fake_settings(), tmp_path)

    assert connection.executed_statements == []
    assert "record_migration" not in connection.events


def test_0003_effective_checksum_covers_exact_sql_backfill_and_identity_bytes(
    tmp_path,
) -> None:
    migration = tmp_path / "0003_push_rule_lifecycle.sql"
    backfill = tmp_path / "push_rule_backfill.py"
    identity = tmp_path / "push_identity.py"
    migration.write_bytes(b"SELECT 'sql';\n")
    backfill.write_bytes(b"BACKFILL = 'version-a'\n")
    identity.write_bytes(b"IDENTITY = 'version-a'\n")
    artifact_paths = (
        ("push_rule_backfill.py", backfill),
        ("push_identity.py", identity),
    )

    snapshot = migration_module.snapshot_migration(
        migration, artifact_paths=artifact_paths
    )
    digest = sha256()
    digest.update(b"reclive:effective-migration:v1\x00")
    for name, contents in (
        (migration.name, migration.read_bytes()),
        ("push_rule_backfill.py", backfill.read_bytes()),
        ("push_identity.py", identity.read_bytes()),
    ):
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    assert snapshot.checksum == digest.hexdigest()

    backfill.write_bytes(b"BACKFILL = 'version-b'\n")
    changed_backfill = migration_module.snapshot_migration(
        migration, artifact_paths=artifact_paths
    )
    assert changed_backfill.checksum != snapshot.checksum

    backfill.write_bytes(b"BACKFILL = 'version-a'\n")
    identity.write_bytes(b"IDENTITY = 'version-b'\n")
    changed_identity = migration_module.snapshot_migration(
        migration, artifact_paths=artifact_paths
    )
    assert changed_identity.checksum != snapshot.checksum

    ordinary = tmp_path / "0004_ordinary.sql"
    ordinary.write_bytes(b"SELECT 'ordinary';\n")
    assert migration_module.snapshot_migration(ordinary).checksum == hashlib.sha256(
        ordinary.read_bytes()
    ).hexdigest()


def test_0003_raw_snapshot_does_not_execute_artifact_top_level_code(
    tmp_path,
) -> None:
    migration = tmp_path / "0003_push_rule_lifecycle.sql"
    backfill = tmp_path / "push_rule_backfill.py"
    identity = tmp_path / "push_identity.py"
    migration.write_bytes(b"SELECT 'sql';\n")
    backfill.write_bytes(
        b"raise AssertionError('raw backfill snapshot was executed')\n"
    )
    identity.write_bytes(
        b"raise AssertionError('raw identity snapshot was executed')\n"
    )

    snapshot = migration_module.snapshot_migration(
        migration,
        artifact_paths=(
            ("push_rule_backfill.py", backfill),
            ("push_identity.py", identity),
        ),
    )

    assert len(snapshot.checksum) == 64
    assert [name for name, _path, _source in snapshot.artifacts] == [
        "push_rule_backfill.py",
        "push_identity.py",
    ]


def test_0003_completed_run_refuses_changed_hook_checksum(
    monkeypatch, tmp_path
) -> None:
    migration = tmp_path / "0003_push_rule_lifecycle.sql"
    migration.write_bytes((MIGRATIONS / migration.name).read_bytes())
    backfill = tmp_path / "push_rule_backfill.py"
    identity = tmp_path / "push_identity.py"
    backfill.write_bytes(PUSH_BACKFILL.read_bytes())
    identity.write_bytes(PUSH_IDENTITY.read_bytes())
    artifact_paths = (
        ("push_rule_backfill.py", backfill),
        ("push_identity.py", identity),
    )
    applied_checksum = migration_module.snapshot_migration(
        migration, artifact_paths=artifact_paths
    ).checksum
    existing_checksums = write_applied_prior_migration_stubs(tmp_path)
    existing_checksums[migration.name] = applied_checksum
    backfill.write_bytes(
        backfill.read_bytes()
        + b"\nraise AssertionError('changed backfill executed')\n"
    )
    connection = FakeConnection(
        existing_checksums=existing_checksums,
        applied_filenames=tuple(existing_checksums),
    )
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    with pytest.raises(MigrationError, match="checksum mismatch"):
        run_migrations(
            fake_settings(), tmp_path, artifact_paths=artifact_paths
        )

    assert "execute_sql" not in connection.events
    assert "create_attempt_history" not in connection.events
    assert "record_migration" not in connection.events


def test_0003_incomplete_run_refuses_changed_identity_checksum(
    monkeypatch, tmp_path
) -> None:
    migration = tmp_path / "0003_push_rule_lifecycle.sql"
    migration.write_bytes((MIGRATIONS / migration.name).read_bytes())
    backfill = tmp_path / "push_rule_backfill.py"
    identity = tmp_path / "push_identity.py"
    backfill.write_bytes(PUSH_BACKFILL.read_bytes())
    identity.write_bytes(PUSH_IDENTITY.read_bytes())
    artifact_paths = (
        ("push_rule_backfill.py", backfill),
        ("push_identity.py", identity),
    )
    attempt_checksum = migration_module.snapshot_migration(
        migration, artifact_paths=artifact_paths
    ).checksum
    existing_checksums = write_applied_prior_migration_stubs(tmp_path)
    identity.write_bytes(
        identity.read_bytes()
        + b"\nraise AssertionError('changed identity executed')\n"
    )
    connection = FakeConnection(
        existing_checksums=existing_checksums,
        applied_filenames=tuple(existing_checksums),
        attempt_checksum=attempt_checksum,
    )
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    with pytest.raises(MigrationError, match="checksum mismatch"):
        run_migrations(
            fake_settings(), tmp_path, artifact_paths=artifact_paths
        )

    assert "execute_sql" not in connection.events
    assert "create_attempt_history" not in connection.events
    assert "record_migration" not in connection.events


def test_0003_executes_the_same_hook_artifact_snapshot_that_it_hashes(
    monkeypatch, tmp_path
) -> None:
    migration = tmp_path / "0003_push_rule_lifecycle.sql"
    migration.write_bytes(b"SELECT 'snapshotted sql';\n")
    identity = tmp_path / "push_identity.py"
    backfill = tmp_path / "push_rule_backfill.py"
    identity.write_bytes(
        b"SNAPSHOT_MARKER = 'version-a'\n"
        b"class PushHashKeyConfigurationError(RuntimeError):\n    pass\n"
        b"def migration_hash_key_identifier():\n    return b'x' * 32\n"
    )
    backfill.write_bytes(
        b"from reclive.push_identity import (\n"
        b"    SNAPSHOT_MARKER, PushHashKeyConfigurationError,\n"
        b")\n"
        b"class PushRuleCutoverNotReadyError(RuntimeError):\n    pass\n"
        b"def preflight_legacy_push_rules(connection, settings):\n"
        b"    connection.events.append('pre:' + SNAPSHOT_MARKER)\n"
        b"def backfill_push_rule_lifecycle(\n"
        b"    connection, settings, started_at, fault_injector=None,\n"
        b"):\n"
        b"    connection.events.append('post:' + SNAPSHOT_MARKER)\n"
    )
    artifact_paths = (
        ("push_rule_backfill.py", backfill),
        ("push_identity.py", identity),
    )
    existing_checksums = write_applied_prior_migration_stubs(tmp_path)
    original_read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        contents = original_read_bytes(path)
        if path == backfill:
            path.write_bytes(contents.replace(b"version-a", b"version-b"))
        elif path == identity:
            path.write_bytes(contents.replace(b"version-a", b"version-b"))
        return contents

    connection = FakeConnection(
        existing_checksums=existing_checksums,
        applied_filenames=tuple(existing_checksums),
    )
    monkeypatch.setattr(Path, "read_bytes", read_then_replace)
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    assert run_migrations(
        fake_settings(), tmp_path, artifact_paths=artifact_paths
    ) == [migration.name]
    assert "pre:version-a" in connection.events
    assert "post:version-a" in connection.events
    assert "pre:version-b" not in connection.events
    assert "post:version-b" not in connection.events


def test_runner_closes_connection_when_release_lock_raises(
    monkeypatch, tmp_path
) -> None:
    migration = tmp_path / "0001_already_applied.sql"
    migration_bytes = b"SELECT 'already applied';\n"
    migration.write_bytes(migration_bytes)
    connection = FakeConnection(
        existing_checksum=hashlib.sha256(migration_bytes).hexdigest(),
        release_error=RuntimeError("release failed"),
    )
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    with pytest.raises(RuntimeError, match="release failed"):
        run_migrations(fake_settings(), tmp_path)

    assert connection.closed is True
    assert connection.events[-1] == "close"


def test_runner_preserves_primary_error_when_release_lock_also_fails(
    monkeypatch, tmp_path
) -> None:
    connection = FakeConnection(release_error=RuntimeError("cleanup failed"))
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    with pytest.raises(MigrationError, match="No numeric SQL migrations found"):
        run_migrations(fake_settings(), tmp_path)

    assert connection.closed is True


def test_pre_hook_failure_rolls_back_before_release_and_close(
    monkeypatch, tmp_path
) -> None:
    (tmp_path / "0001_pre_hook.sql").write_text("SELECT 1;\n", encoding="utf-8")
    connection = FakeConnection()
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)

    def fail_pre_hook(filename, migration_connection, settings, hooks=None) -> None:
        raise RuntimeError("pre-hook failed")

    monkeypatch.setattr(migration_module, "run_pre_sql_hook", fail_pre_hook)

    with pytest.raises(RuntimeError, match="pre-hook failed"):
        run_migrations(fake_settings(), tmp_path)

    assert connection.events[-3:] == ["rollback", "release", "close"]
    assert connection.events.count("commit") == 1


def test_runner_hashes_and_executes_the_same_file_snapshot(
    monkeypatch, tmp_path
) -> None:
    migration = tmp_path / "0001_snapshot.sql"
    original = b"SELECT 'version a';\n"
    replacement = b"SELECT 'version b';\n"
    migration.write_bytes(original)
    connection = FakeConnection()
    monkeypatch.setattr(migration_module, "connect", lambda settings: connection)
    original_read_bytes = Path.read_bytes

    def read_then_replace(path: Path) -> bytes:
        contents = original_read_bytes(path)
        if path == migration:
            path.write_bytes(replacement)
        return contents

    monkeypatch.setattr(Path, "read_bytes", read_then_replace)

    assert run_migrations(fake_settings(), tmp_path) == [migration.name]
    assert connection.executed_statements == ["SELECT 'version a'"]
    assert connection.recorded_checksum == hashlib.sha256(original).hexdigest()


@pytest.mark.mysql
def test_runner_reports_advisory_lock_timeout_without_applying(
    clean_test_database,
) -> None:
    connection = pymysql.connect(**clean_test_database)
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT GET_LOCK(%s, 0)", (LOCK_NAME,))
            assert cursor.fetchone()[0] == 1

        settings = {**clean_test_database, "lock_timeout_seconds": 1}
        result = run_migrate(settings, MIGRATIONS)

        assert result.returncode == 1
        assert result.stdout == ""
        assert result.stderr == (
            "migration failed: Timed out acquiring migration advisory lock\n"
        )
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'schema_migrations'")
            assert cursor.fetchone() is None
    finally:
        connection.close()


def test_cli_sanitizes_unexpected_exception_details(tmp_path) -> None:
    sentinel = "secret-like-sentinel-4f65b0e1"
    settings: dict[str, object] = {
        "host": "127.0.0.1",
        "port": sentinel,
        "user": "reclive",
        "password": "reclive-ci-password",
        "database": "reclive_test",
    }

    result = run_migrate(settings, tmp_path)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr == "migration failed: unexpected migration error\n"
    assert sentinel not in result.stderr


@pytest.mark.mysql
def test_0002_creates_exact_snapshot_and_ingestion_contract(
    clean_test_database, tmp_path
) -> None:
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        maria = detect_database_dialect(connection) is DatabaseDialect.MARIADB1011
        generated = "" if maria else "default_generated"
        on_update = (
            "on update current_timestamp(6)" if maria
            else "default_generated on update current_timestamp(6)"
        )
        with connection.cursor() as cursor:
            snapshot_columns = table_column_contract(cursor, "location_snapshot", mariadb=maria)
            assert snapshot_columns == {
                "location_id": ("int", "NO", "<null>", "", None),
                "is_closed": ("tinyint(1)", "YES", "<null>", "", None),
                "current_capacity": ("int", "YES", "<null>", "", None),
                "max_capacity": ("int", "YES", "<null>", "", None),
                "source_updated_at": (
                    "datetime(6)",
                    "YES",
                    "<null>",
                    "",
                    6,
                ),
                "fetched_at": ("datetime(6)", "NO", "<null>", "", 6),
                "created_at": (
                    "datetime(6)",
                    "NO",
                    "current_timestamp(6)",
                    generated,
                    6,
                ),
                "updated_at": (
                    "datetime(6)",
                    "NO",
                    "current_timestamp(6)",
                    on_update,
                    6,
                ),
            }
            run_columns = table_column_contract(cursor, "ingestion_runs", mariadb=maria)
            assert run_columns == {
                "id": ("bigint unsigned", "NO", "<null>", "auto_increment", None),
                "started_at": ("datetime(6)", "NO", "<null>", "", 6),
                "completed_at": ("datetime(6)", "YES", "<null>", "", 6),
                "status": ("varchar(16)", "NO", "<null>", "", None),
                "received_count": ("int unsigned", "NO", "0", "", None),
                "valid_count": ("int unsigned", "NO", "0", "", None),
                "history_inserted_count": (
                    "int unsigned",
                    "NO",
                    "0",
                    "",
                    None,
                ),
                "snapshot_updated_count": (
                    "int unsigned",
                    "NO",
                    "0",
                    "",
                    None,
                ),
                "observed_location_ids": (
                    "longtext" if maria else "json",
                    "NO",
                    "json_array()",
                    generated,
                    None,
                ),
                "error_category": ("varchar(64)", "YES", "<null>", "", None),
                "error_message": ("varchar(240)", "YES", "<null>", "", None),
            }
            assert table_index_contract(cursor, "location_snapshot") == {
                "PRIMARY": ("location_id",),
                "idx_location_snapshot_fetched_at": ("fetched_at",),
            }
            assert table_index_contract(cursor, "ingestion_runs") == {
                "PRIMARY": ("id",),
                "idx_ingestion_runs_status_completed": (
                    "status",
                    "completed_at",
                ),
            }
            cursor.execute(
                "SELECT constraint_type FROM information_schema.table_constraints "
                "WHERE table_schema = DATABASE() "
                "AND table_name = 'ingestion_runs' "
                "AND constraint_name = 'chk_ingestion_runs_status'"
            )
            assert cursor.fetchone() == ("CHECK",)
            with pytest.raises(pymysql.MySQLError) as invalid_status:
                cursor.execute(
                    "INSERT INTO ingestion_runs (started_at, status) "
                    "VALUES (UTC_TIMESTAMP(6), 'unsafe')"
                )
            assert invalid_status.value.args[0] == (4025 if maria else 3819)
            cursor.execute(
                "INSERT INTO ingestion_runs (started_at, status) "
                "VALUES (UTC_TIMESTAMP(6), 'running')"
            )
            cursor.execute(
                "SELECT JSON_LENGTH(observed_location_ids) FROM ingestion_runs"
            )
            assert cursor.fetchone()[0] == 0
    finally:
        connection.close()


@pytest.mark.mysql
def test_0004_creates_exact_hashed_rate_limit_counter(
    clean_test_database, tmp_path
) -> None:
    migration_dir = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, migration_dir)
    result = run_migrate(clean_test_database, migration_dir)
    assert result.returncode == 0, result.stderr

    connection = pymysql.connect(**clean_test_database)
    try:
        maria = detect_database_dialect(connection) is DatabaseDialect.MARIADB1011
        with connection.cursor() as cursor:
            assert table_column_contract(cursor, "push_rate_limits", mariadb=maria) == {
                "subject_hash": ("binary(32)", "NO", "<null>", "", None),
                "window_started_at": (
                    "datetime(6)",
                    "NO",
                    "<null>",
                    "",
                    6,
                ),
                "request_count": (
                    "int unsigned",
                    "NO",
                    "0",
                    "",
                    None,
                ),
                "updated_at": (
                    "datetime(6)",
                    "NO",
                    "current_timestamp(6)",
                    "on update current_timestamp(6)" if maria
                    else "default_generated on update current_timestamp(6)",
                    6,
                ),
            }
            assert table_index_contract(cursor, "push_rate_limits") == {
                "PRIMARY": ("subject_hash", "window_started_at"),
                "idx_push_rate_limits_cleanup": ("updated_at",),
            }
    finally:
        connection.close()
