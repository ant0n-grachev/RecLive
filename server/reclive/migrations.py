from __future__ import annotations

import hashlib
import hmac
import os
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType

import pymysql

from .database_dialect import (
    MARIADB_HOOK_REWRITES,
    DatabaseDialect,
    UnsupportedDatabaseError,
    detect_database_dialect,
    migration_hook_connection,
)

MIGRATION_NAME = re.compile(r"^\d{4}_[a-z0-9_]+\.sql$")
LOCK_NAME = "reclive_schema_migrations"
EFFECTIVE_MIGRATION_DOMAIN = b"reclive:effective-migration:v1\x00"
PUSH_MIGRATION_NAME = "0003_push_rule_lifecycle.sql"
PUSH_ARTIFACT_NAMES = ("push_rule_backfill.py", "push_identity.py")
MARIADB_MIGRATION_DOMAIN = b"reclive:mariadb-10.11:utf8mb4-unicode-ci:v1\x00"

FaultInjector = Callable[[str], None]


class MigrationError(RuntimeError):
    """An operator-safe migration failure."""


@dataclass(frozen=True)
class MigrationSettings:
    host: str
    port: int
    user: str
    password: str
    database: str
    lock_timeout_seconds: int

    @classmethod
    def from_environment(cls) -> MigrationSettings:
        required = {
            name: os.environ.get(name, "").strip()
            for name in (
                "GYM_DB_HOST",
                "GYM_DB_PORT",
                "GYM_DB_USER",
                "GYM_DB_PASSWORD",
                "GYM_DB_NAME",
            )
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise MigrationError(
                f"Missing required environment variables: {', '.join(missing)}"
            )
        return cls(
            host=required["GYM_DB_HOST"],
            port=int(required["GYM_DB_PORT"]),
            user=required["GYM_DB_USER"],
            password=required["GYM_DB_PASSWORD"],
            database=required["GYM_DB_NAME"],
            lock_timeout_seconds=max(
                1, int(os.environ.get("MIGRATION_LOCK_TIMEOUT_SECONDS", "30"))
            ),
        )


@dataclass(frozen=True)
class MigrationHooks:
    preflight: Callable[[object, MigrationSettings], bytes | None]
    backfill: Callable[
        [object, MigrationSettings, object, FaultInjector | None], None
    ]
    key_identifier: Callable[[], bytes]
    operator_safe_errors: tuple[type[BaseException], ...]


@dataclass(frozen=True)
class MigrationSnapshot:
    path: Path
    sql_bytes: bytes
    checksum: str
    artifacts: tuple[tuple[str, Path, bytes], ...] = ()
    dialect_artifact: tuple[Path, bytes] | None = None


def migration_files(migration_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in migration_dir.iterdir()
        if path.is_file() and MIGRATION_NAME.fullmatch(path.name)
    )
    if not files:
        raise MigrationError(f"No numeric SQL migrations found in {migration_dir}")
    validate_contiguous_versions([path.name for path in files])
    return files


def validate_contiguous_versions(filenames: list[str]) -> None:
    versions: list[int] = []
    for filename in filenames:
        if not MIGRATION_NAME.fullmatch(filename):
            raise MigrationError("Migration history contains an invalid filename")
        versions.append(int(filename[:4]))
    if versions != list(range(1, len(versions) + 1)):
        raise MigrationError(
            "Migration versions must be unique and contiguous beginning at 0001"
        )


def validate_applied_history(connection, available_filenames: list[str]) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT filename FROM schema_migrations ORDER BY filename")
        applied_filenames = [str(row[0]) for row in cursor.fetchall()]
    if applied_filenames:
        validate_contiguous_versions(applied_filenames)
    available = set(available_filenames)
    if any(filename not in available for filename in applied_filenames):
        raise MigrationError("Migration history has a missing applied migration file")


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def default_push_artifact_paths() -> tuple[tuple[str, Path], ...]:
    module_dir = Path(__file__).resolve().parent
    return tuple((name, module_dir / name) for name in PUSH_ARTIFACT_NAMES)


def snapshot_migration(
    path: Path,
    *,
    artifact_paths: tuple[tuple[str, Path], ...] | None = None,
) -> MigrationSnapshot:
    sql_bytes = path.read_bytes()
    if path.name != PUSH_MIGRATION_NAME:
        return MigrationSnapshot(
            path=path,
            sql_bytes=sql_bytes,
            checksum=hashlib.sha256(sql_bytes).hexdigest(),
        )

    selected_paths = artifact_paths or default_push_artifact_paths()
    if tuple(name for name, _artifact_path in selected_paths) != PUSH_ARTIFACT_NAMES:
        raise MigrationError(
            "0003 effective migration artifacts must be backfill then identity"
        )
    artifacts = tuple(
        (name, artifact_path, artifact_path.read_bytes())
        for name, artifact_path in selected_paths
    )
    parts = (
        (path.name, sql_bytes),
        *((name, contents) for name, _path, contents in artifacts),
    )
    digest = hashlib.sha256()
    digest.update(EFFECTIVE_MIGRATION_DOMAIN)
    for name, contents in parts:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    effective_checksum = digest.hexdigest()
    return MigrationSnapshot(
        path=path,
        sql_bytes=sql_bytes,
        checksum=effective_checksum,
        artifacts=artifacts,
    )


def load_snapshot_hooks(snapshot: MigrationSnapshot) -> MigrationHooks:
    if tuple(name for name, _path, _contents in snapshot.artifacts) != (
        PUSH_ARTIFACT_NAMES
    ):
        raise MigrationError("0003 effective migration artifacts are unavailable")
    backfill_name, backfill_path, backfill_bytes = snapshot.artifacts[0]
    identity_name, identity_path, identity_bytes = snapshot.artifacts[1]
    if (
        backfill_name != "push_rule_backfill.py"
        or identity_name != "push_identity.py"
    ):
        raise MigrationError("0003 effective migration artifacts are unavailable")
    return load_push_migration_hooks(
        identity_bytes=identity_bytes,
        identity_path=identity_path,
        backfill_bytes=backfill_bytes,
        backfill_path=backfill_path,
        effective_checksum=snapshot.checksum,
    )


def execution_snapshot(
    snapshot: MigrationSnapshot, dialect: DatabaseDialect,
) -> MigrationSnapshot:
    """Bind the executed SQL and immutable source artifacts to the dialect."""
    if dialect is DatabaseDialect.MYSQL8:
        return snapshot
    if dialect is not DatabaseDialect.MARIADB1011:
        raise MigrationError("Unsupported database migration dialect")
    sql_bytes = snapshot.sql_bytes.replace(
        b"COLLATE=utf8mb4_0900_ai_ci", b"COLLATE=utf8mb4_unicode_ci"
    )
    if b"utf8mb4_0900" in sql_bytes.lower():
        raise MigrationError("Unsupported MariaDB migration collation")
    digest = hashlib.sha256(
        MARIADB_MIGRATION_DOMAIN + bytes.fromhex(snapshot.checksum) + sql_bytes
    )
    for source, target in sorted(MARIADB_HOOK_REWRITES.items()):
        for statement in (source, target):
            encoded = statement.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    dialect_artifact = None
    if snapshot.path.name == PUSH_MIGRATION_NAME:
        artifact_path = default_mariadb_timestamp_artifact_path()
        artifact_bytes = artifact_path.read_bytes()
        dialect_artifact = (artifact_path, artifact_bytes)
        for part in (b"mariadb_legacy_timestamps.py", artifact_bytes):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return replace(
        snapshot, sql_bytes=sql_bytes, checksum=digest.hexdigest(),
        dialect_artifact=dialect_artifact,
    )


def default_mariadb_timestamp_artifact_path() -> Path:
    return Path(__file__).with_name("mariadb_legacy_timestamps.py")


def run_dialect_pre_backfill(snapshot: MigrationSnapshot, connection, fault_injector) -> None:
    if snapshot.dialect_artifact is None:
        return
    path, contents = snapshot.dialect_artifact
    module = execute_snapshot_module(
        f"_reclive_0003_mariadb_timestamps_{snapshot.checksum}", contents, path,
    )
    module.prepare_legacy_timestamps(connection, fault_injector)


def load_push_migration_hooks(
    *,
    identity_bytes: bytes,
    identity_path: Path,
    backfill_bytes: bytes,
    backfill_path: Path,
    effective_checksum: str,
) -> MigrationHooks:
    identity_module = execute_snapshot_module(
        f"_reclive_0003_identity_{effective_checksum}",
        identity_bytes,
        identity_path,
    )
    canonical_identity_name = "reclive.push_identity"
    previous_identity = sys.modules.get(canonical_identity_name)
    sys.modules[canonical_identity_name] = identity_module
    try:
        backfill_module = execute_snapshot_module(
            f"_reclive_0003_backfill_{effective_checksum}",
            backfill_bytes,
            backfill_path,
        )
    finally:
        if previous_identity is None:
            sys.modules.pop(canonical_identity_name, None)
        else:
            sys.modules[canonical_identity_name] = previous_identity

    key_error = getattr(identity_module, "PushHashKeyConfigurationError")
    gate_error = getattr(backfill_module, "PushRuleCutoverNotReadyError")
    return MigrationHooks(
        preflight=getattr(backfill_module, "preflight_legacy_push_rules"),
        backfill=getattr(backfill_module, "backfill_push_rule_lifecycle"),
        key_identifier=getattr(
            identity_module, "migration_hash_key_identifier"
        ),
        operator_safe_errors=(key_error, gate_error),
    )


def execute_snapshot_module(
    name: str, source_bytes: bytes, source_path: Path
) -> ModuleType:
    module = ModuleType(name)
    module.__file__ = str(source_path)
    module.__package__ = "reclive"
    source = source_bytes.decode("utf-8")
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    return module


def connect(settings: MigrationSettings):
    return pymysql.connect(
        host=settings.host,
        port=settings.port,
        user=settings.user,
        password=settings.password,
        database=settings.database,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
    )


def split_statements(sql: str) -> list[str]:
    without_full_line_comments = "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )
    statements = [statement.strip() for statement in without_full_line_comments.split(";")]
    return [statement for statement in statements if statement]


def run_post_sql_hook(
    filename: str,
    connection,
    settings: MigrationSettings,
    hooks: MigrationHooks | None,
    started_at,
    fault_injector: FaultInjector | None,
) -> None:
    if filename != PUSH_MIGRATION_NAME:
        return
    if hooks is None or started_at is None:
        raise MigrationError("0003 effective migration hooks are unavailable")
    hooks.backfill(connection, settings, started_at, fault_injector)


def run_pre_sql_hook(
    filename: str,
    connection,
    settings: MigrationSettings,
    hooks: MigrationHooks | None = None,
) -> bytes | None:
    if filename != PUSH_MIGRATION_NAME:
        return None
    if hooks is None:
        raise MigrationError("0003 effective migration hooks are unavailable")
    try:
        return hooks.preflight(connection, settings)
    except hooks.operator_safe_errors as exc:
        raise MigrationError(str(exc)) from None


def configured_attempt_key_identifier(hooks: MigrationHooks) -> bytes:
    try:
        identifier = hooks.key_identifier()
    except hooks.operator_safe_errors as exc:
        raise MigrationError(str(exc)) from None
    if not isinstance(identifier, bytes) or len(identifier) != 32:
        raise MigrationError("0003 migration key identifier is invalid")
    return identifier


def migration_attempt_started_at(
    connection,
    filename: str,
    effective_checksum: str,
    key_identifier: bytes | None,
):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migration_attempts (
                filename VARCHAR(255) NOT NULL,
                effective_checksum CHAR(64) NOT NULL,
                started_at DATETIME(6) NOT NULL,
                key_identifier BINARY(32) NULL,
                PRIMARY KEY (filename, effective_checksum)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        )
    connection.commit()

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT effective_checksum, started_at, key_identifier "
            "FROM schema_migration_attempts "
            "WHERE filename = %s",
            (filename,),
        )
        recorded = cursor.fetchone()
        if recorded is not None:
            if recorded[0] != effective_checksum:
                raise MigrationError(
                    f"Migration checksum mismatch for {filename}"
                )
            if recorded[2] != key_identifier:
                raise MigrationError(
                    "PUSH_ENDPOINT_HASH_KEY does not match the key bound to "
                    "incomplete 0003 migration attempt"
                )
            return recorded[1]
        cursor.execute(
            "INSERT INTO schema_migration_attempts "
            "(filename, effective_checksum, started_at, key_identifier) "
            "VALUES (%s, %s, UTC_TIMESTAMP(6), %s)",
            (filename, effective_checksum, key_identifier),
        )
    connection.commit()

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT effective_checksum, started_at, key_identifier "
            "FROM schema_migration_attempts "
            "WHERE filename = %s",
            (filename,),
        )
        recorded = cursor.fetchone()
    if (
        recorded is None
        or recorded[0] != effective_checksum
        or recorded[2] != key_identifier
    ):
        raise MigrationError(f"Migration attempt recording failed for {filename}")
    return recorded[1]


def existing_migration_attempt(connection, filename: str):
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() "
            "AND table_name = 'schema_migration_attempts'"
        )
        if int(cursor.fetchone()[0]) == 0:
            return None
        cursor.execute(
            "SELECT effective_checksum, started_at, key_identifier "
            "FROM schema_migration_attempts "
            "WHERE filename = %s",
            (filename,),
        )
        return cursor.fetchone()


def run_migrations(
    settings: MigrationSettings,
    migration_dir: Path,
    *,
    artifact_paths: tuple[tuple[str, Path], ...] | None = None,
    fault_injector: FaultInjector | None = None,
) -> list[str]:
    connection = connect(settings)
    locked = False
    applied: list[str] = []
    primary_error: BaseException | None = None
    cleanup_error: BaseException | None = None
    try:
        try:
            dialect = detect_database_dialect(connection)
        except UnsupportedDatabaseError as exc:
            raise MigrationError(str(exc)) from None
        hook_connection = migration_hook_connection(connection, dialect)
        snapshots = [
            execution_snapshot(
                snapshot_migration(
                    path,
                    artifact_paths=(
                        artifact_paths
                        if path.name == PUSH_MIGRATION_NAME
                        else None
                    ),
                ),
                dialect,
            )
            for path in migration_files(migration_dir)
        ]
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT GET_LOCK(%s, %s)",
                (LOCK_NAME, settings.lock_timeout_seconds),
            )
            locked = bool(cursor.fetchone()[0])
            if not locked:
                raise MigrationError("Timed out acquiring migration advisory lock")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    filename VARCHAR(255) NOT NULL PRIMARY KEY,
                    checksum CHAR(64) NOT NULL,
                    applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        connection.commit()
        validate_applied_history(
            connection, [snapshot.path.name for snapshot in snapshots]
        )

        for snapshot in snapshots:
            path = snapshot.path
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT checksum FROM schema_migrations WHERE filename = %s",
                        (path.name,),
                    )
                    recorded = cursor.fetchone()
                if recorded:
                    if recorded[0] != snapshot.checksum:
                        raise MigrationError(
                            f"Migration checksum mismatch for {path.name}"
                        )
                    continue

                existing_attempt = None
                hooks = None
                if path.name == PUSH_MIGRATION_NAME:
                    existing_attempt = existing_migration_attempt(
                        connection, path.name
                    )
                    if (
                        existing_attempt is not None
                        and existing_attempt[0] != snapshot.checksum
                    ):
                        raise MigrationError(
                            f"Migration checksum mismatch for {path.name}"
                        )
                    hooks = load_snapshot_hooks(snapshot)
                    if (
                        existing_attempt is not None
                        and existing_attempt[2] is not None
                    ):
                        current_identifier = configured_attempt_key_identifier(
                            hooks
                        )
                        if not hmac.compare_digest(
                            bytes(existing_attempt[2]), current_identifier
                        ):
                            raise MigrationError(
                                "PUSH_ENDPOINT_HASH_KEY does not match the key "
                                "bound to incomplete 0003 migration attempt"
                            )
                preflight_identifier = run_pre_sql_hook(
                    path.name, hook_connection, settings, hooks
                )
                started_at = None
                if path.name == PUSH_MIGRATION_NAME:
                    if existing_attempt is None:
                        started_at = migration_attempt_started_at(
                            connection,
                            path.name,
                            snapshot.checksum,
                            preflight_identifier,
                        )
                    else:
                        if (
                            existing_attempt[2] is None
                            and preflight_identifier is not None
                        ):
                            raise MigrationError(
                                "PUSH_ENDPOINT_HASH_KEY cannot be verified for "
                                "incomplete 0003 migration attempt"
                            )
                        started_at = existing_attempt[1]
                statements = split_statements(snapshot.sql_bytes.decode("utf-8"))
                with connection.cursor() as cursor:
                    for index, statement in enumerate(statements):
                        cursor.execute(statement)
                        inject_fault(
                            fault_injector,
                            f"{path.name}:after_sql:{index}",
                        )
                def scoped_fault(checkpoint: str) -> None:
                    inject_fault(
                        fault_injector, f"{path.name}:{checkpoint}"
                    )

                run_dialect_pre_backfill(snapshot, connection, scoped_fault)
                run_post_sql_hook(
                    path.name,
                    hook_connection,
                    settings,
                    hooks,
                    started_at,
                    scoped_fault,
                )
                inject_fault(fault_injector, f"{path.name}:before_record")
                with connection.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO schema_migrations (filename, checksum) "
                        "VALUES (%s, %s)",
                        (path.name, snapshot.checksum),
                    )
                connection.commit()
            except BaseException:
                try:
                    connection.rollback()
                except BaseException:
                    pass
                raise
            applied.append(path.name)
    except BaseException as exc:
        primary_error = exc
    finally:
        if locked:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT RELEASE_LOCK(%s)", (LOCK_NAME,))
            except BaseException as exc:
                cleanup_error = exc
        try:
            connection.close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc

    if primary_error is not None:
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_error is not None:
        raise cleanup_error.with_traceback(cleanup_error.__traceback__)
    return applied


def inject_fault(
    fault_injector: FaultInjector | None, checkpoint: str
) -> None:
    if fault_injector is not None:
        fault_injector(checkpoint)


sys.modules.setdefault("server.reclive.migrations", sys.modules[__name__])
sys.modules.setdefault("reclive.migrations", sys.modules[__name__])
