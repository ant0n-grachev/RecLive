from __future__ import annotations

import json
import os
import re
from collections.abc import Callable

from reclive.push_identity import (
    cancelled_legacy_endpoint_hash,
    configured_push_hash_key,
    endpoint_hash,
    migration_hash_key_identifier,
    normalize_push_endpoint,
)

IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
CUTOVER_GATE_NAME = "PUSH_RULE_SCHEMA_CUTOVER_READY"
CUTOVER_TABLE_NAME = "_reclive_push_rules_cutover"
PUBLIC_TABLE_NAME = "push_rules"
LEGACY_COLUMNS = {
    "endpoint",
    "subscription_json",
    "facility_id",
    "section_key",
    "threshold",
    "created_at",
}

FaultInjector = Callable[[str], None]


class PushRuleCutoverNotReadyError(RuntimeError):
    """A fixed, operator-safe push schema cutover error."""


def table_columns(connection, table: str) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s",
            (table,),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def table_exists(connection, table: str) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name = %s",
            (table,),
        )
        return int(cursor.fetchone()[0]) == 1


def migration_table(connection) -> str:
    has_cutover = table_exists(connection, CUTOVER_TABLE_NAME)
    has_public = table_exists(connection, PUBLIC_TABLE_NAME)
    if has_cutover and has_public:
        raise RuntimeError(
            "push_rules cutover failed: public and cutover tables both exist"
        )
    if has_cutover:
        return CUTOVER_TABLE_NAME
    return PUBLIC_TABLE_NAME


def quoted_table(table: str) -> str:
    if table not in {CUTOVER_TABLE_NAME, PUBLIC_TABLE_NAME}:
        raise RuntimeError("push_rules cutover failed: unsafe table name")
    return f"`{table}`"


def require_hash_key() -> bytes:
    return configured_push_hash_key()


def preflight_legacy_push_rules(connection, _settings) -> bytes | None:
    if os.environ.get(CUTOVER_GATE_NAME) != "1":
        raise PushRuleCutoverNotReadyError(
            f"{CUTOVER_GATE_NAME}=1 is required for the push-rule schema cutover"
        )
    has_cutover = table_exists(connection, CUTOVER_TABLE_NAME)
    has_public = table_exists(connection, PUBLIC_TABLE_NAME)
    if has_cutover and has_public:
        raise RuntimeError(
            "push_rules cutover failed: public and cutover tables both exist"
        )
    if not has_cutover and not has_public:
        return None
    if not has_cutover:
        columns = table_columns(connection, PUBLIC_TABLE_NAME)
        if "endpoint" not in columns:
            return None
        if not LEGACY_COLUMNS <= columns:
            raise RuntimeError(
                "push_rules legacy preflight failed: "
                "required legacy columns are missing"
            )
        with connection.cursor() as cursor:
            cursor.execute(
                f"RENAME TABLE {quoted_table(PUBLIC_TABLE_NAME)} "
                f"TO {quoted_table(CUTOVER_TABLE_NAME)}"
            )
    columns = table_columns(connection, CUTOVER_TABLE_NAME)
    if not LEGACY_COLUMNS <= columns:
        if "endpoint" in columns:
            raise RuntimeError(
                "push_rules legacy preflight failed: "
                "required legacy columns are missing"
            )
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT COUNT(*) FROM {quoted_table(CUTOVER_TABLE_NAME)}"
        )
        has_rows = int(cursor.fetchone()[0]) > 0
    if has_rows:
        return migration_hash_key_identifier()
    return None


def backfill_push_rule_lifecycle(
    connection,
    _settings,
    started_at,
    fault_injector: FaultInjector | None = None,
) -> None:
    table = migration_table(connection)
    sql_table = quoted_table(table)
    columns = table_columns(connection, table)
    has_legacy_endpoint = "endpoint" in columns
    if not has_legacy_endpoint:
        finalize_empty_or_resumed_table(
            connection, started_at, fault_injector, table
        )
        return

    preflight_legacy_push_rules(connection, _settings)
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, endpoint, subscription_json, facility_id, section_key, "
            "threshold, created_at, endpoint_hash, status, expires_at, "
            f"finalized_at, failure_code FROM {sql_table} ORDER BY id"
        )
        legacy_rows = cursor.fetchall()
    if not legacy_rows:
        finalize_empty_or_resumed_table(
            connection, started_at, fault_injector, table
        )
        return

    require_hash_key()
    candidates: list[tuple[object, ...]] = []
    new_invalid_rows: list[tuple[object, ...]] = []
    for (
        rule_id,
        endpoint,
        subscription_json,
        facility_id,
        section_key,
        threshold,
        created_at,
        stored_digest,
        status,
        _expires_at,
        _finalized_at,
        _stored_failure_code,
    ) in legacy_rows:
        if stored_digest is not None:
            if status == "pending":
                candidates.append(
                    (
                        int(rule_id),
                        bytes(stored_digest),
                        int(facility_id),
                        str(section_key or "legacy").strip() or "legacy",
                        int(threshold),
                        created_at,
                        True,
                    )
                )
            continue

        raw_endpoint = str(endpoint or "")
        try:
            normalized_endpoint = normalize_push_endpoint(raw_endpoint)
            decoded_subscription = (
                json.loads(subscription_json)
                if isinstance(subscription_json, str)
                else subscription_json
            )
            subscription_endpoint = (
                decoded_subscription.get("endpoint")
                if isinstance(decoded_subscription, dict)
                else None
            )
            valid_subscription = (
                isinstance(subscription_endpoint, str)
                and normalize_push_endpoint(subscription_endpoint)
                == normalized_endpoint
            )
            valid_threshold = 1 <= int(threshold) <= 100
            failure_code = (
                None
                if valid_subscription and valid_threshold
                else (
                    "migration_invalid_threshold"
                    if not valid_threshold
                    else "migration_invalid_subscription"
                )
            )
            digest = endpoint_hash(normalized_endpoint)
        except (ValueError, TypeError, json.JSONDecodeError):
            digest = cancelled_legacy_endpoint_hash(int(rule_id))
            failure_code = "migration_invalid_subscription"

        safe_threshold = int(threshold) if 1 <= int(threshold) <= 100 else 1
        if failure_code is None:
            candidates.append(
                (
                    int(rule_id),
                    digest,
                    int(facility_id),
                    str(section_key or "legacy").strip() or "legacy",
                    safe_threshold,
                    created_at,
                    False,
                )
            )
        else:
            new_invalid_rows.append(
                (int(rule_id), digest, safe_threshold, failure_code)
            )

    winners: dict[tuple[bytes, int, str, int], tuple[object, ...]] = {}
    duplicates: list[tuple[object, ...]] = []
    for row in sorted(
        candidates, key=lambda item: (item[5], item[0]), reverse=True
    ):
        identity = (row[1], row[2], row[3], row[4])
        if identity in winners:
            duplicates.append(row)
        else:
            winners[identity] = row

    with connection.cursor() as cursor:
        for (
            rule_id,
            digest,
            _facility_id,
            _section_key,
            _threshold,
            _created_at,
            was_staged,
        ) in winners.values():
            if was_staged:
                continue
            cursor.execute(
                f"UPDATE {sql_table} SET endpoint_hash = %s, "
                "expires_at = COALESCE(expires_at, "
                "DATE_ADD(%s, INTERVAL 24 HOUR)), "
                "status = 'pending', claimed_at = NULL, sent_at = NULL, "
                "finalized_at = NULL, "
                "failure_code = NULL WHERE id = %s",
                (digest, started_at, rule_id),
            )
        for (
            rule_id,
            digest,
            _facility_id,
            _section_key,
            threshold,
            _created_at,
            _was_staged,
        ) in duplicates:
            cursor.execute(
                f"UPDATE {sql_table} SET endpoint_hash = %s, threshold = %s, "
                "expires_at = %s, status = 'cancelled', "
                "claimed_at = NULL, sent_at = NULL, "
                "finalized_at = COALESCE(finalized_at, %s), "
                "failure_code = COALESCE(failure_code, 'migration_duplicate') "
                "WHERE id = %s",
                (
                    digest,
                    threshold,
                    started_at,
                    started_at,
                    rule_id,
                ),
            )
        for rule_id, digest, threshold, failure_code in new_invalid_rows:
            cursor.execute(
                f"UPDATE {sql_table} SET endpoint_hash = %s, threshold = %s, "
                "expires_at = %s, status = 'cancelled', "
                "claimed_at = NULL, sent_at = NULL, "
                "finalized_at = COALESCE(finalized_at, %s), "
                "failure_code = COALESCE(failure_code, %s) WHERE id = %s",
                (
                    digest,
                    threshold,
                    started_at,
                    started_at,
                    failure_code,
                    rule_id,
                ),
            )

    validate_prepared_rows(connection, table)
    inject_fault(fault_injector, "after_backfill_updates")
    finalize_empty_or_resumed_table(
        connection, started_at, fault_injector, table
    )


def finalize_empty_or_resumed_table(
    connection,
    _started_at,
    fault_injector: FaultInjector | None = None,
    table: str | None = None,
) -> None:
    selected_table = table or migration_table(connection)
    sql_table = quoted_table(selected_table)
    validate_prepared_rows(connection, selected_table)
    inject_fault(fault_injector, "before_endpoint_hash_not_null")
    with connection.cursor() as cursor:
        cursor.execute(
            f"ALTER TABLE {sql_table} "
            "MODIFY endpoint_hash BINARY(32) NOT NULL, "
            "ALGORITHM=INPLACE, LOCK=EXCLUSIVE"
        )
    inject_fault(fault_injector, "after_endpoint_hash_not_null")
    validate_prepared_rows(connection, selected_table)

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT index_name FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = %s "
            "AND column_name = 'endpoint'",
            (selected_table,),
        )
        endpoint_indexes = sorted({str(row[0]) for row in cursor.fetchall()})
        for name in endpoint_indexes:
            if not IDENTIFIER.fullmatch(name):
                raise RuntimeError(
                    "push_rules backfill validation failed: "
                    "unsafe legacy index name"
                )
            cursor.execute(f"DROP INDEX `{name}` ON {sql_table}")
            inject_fault(fault_injector, "after_endpoint_index_drop")
        if "endpoint" in table_columns(connection, selected_table):
            cursor.execute(f"ALTER TABLE {sql_table} DROP COLUMN endpoint")
            inject_fault(fault_injector, "after_endpoint_drop")
        cursor.execute(
            f"ALTER TABLE {sql_table} "
            "MODIFY created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6), "
            "MODIFY threshold TINYINT UNSIGNED NOT NULL, "
            "MODIFY expires_at DATETIME(6) NOT NULL, "
            "MODIFY status VARCHAR(32) NOT NULL"
        )
        inject_fault(fault_injector, "after_final_columns")
        ensure_constraint(
            cursor,
            selected_table,
            "chk_push_rules_threshold",
            f"ALTER TABLE {sql_table} "
            "ADD CONSTRAINT chk_push_rules_threshold "
            "CHECK (threshold BETWEEN 1 AND 100)",
        )
        inject_fault(
            fault_injector,
            "after_constraint:chk_push_rules_threshold",
        )
        ensure_constraint(
            cursor,
            selected_table,
            "chk_push_rules_status",
            f"ALTER TABLE {sql_table} ADD CONSTRAINT chk_push_rules_status "
            "CHECK (status IN ('pending', 'claimed', 'sent', 'failed', "
            "'expired', 'invalid_subscription', 'cancelled'))",
        )
        inject_fault(
            fault_injector, "after_constraint:chk_push_rules_status"
        )
        ensure_index(
            cursor,
            selected_table,
            "uq_push_rules_identity",
            f"CREATE UNIQUE INDEX uq_push_rules_identity ON {sql_table} "
            "(endpoint_hash, facility_id, section_key, threshold, active_identity)",
        )
        inject_fault(
            fault_injector, "after_index:uq_push_rules_identity"
        )
        ensure_index(
            cursor,
            selected_table,
            "idx_push_rules_pending",
            f"CREATE INDEX idx_push_rules_pending ON {sql_table} "
            "(status, expires_at, id)",
        )
        inject_fault(fault_injector, "after_index:idx_push_rules_pending")
    validate_prepared_rows(connection, selected_table)
    inject_fault(fault_injector, "before_cutover_barrier_release")
    if selected_table == CUTOVER_TABLE_NAME:
        if table_exists(connection, PUBLIC_TABLE_NAME):
            raise RuntimeError(
                "push_rules cutover failed: public table appeared during cutover"
            )
        with connection.cursor() as cursor:
            cursor.execute(
                f"RENAME TABLE {quoted_table(CUTOVER_TABLE_NAME)} "
                f"TO {quoted_table(PUBLIC_TABLE_NAME)}"
            )
    inject_fault(fault_injector, "after_cutover_barrier_release")


def validate_prepared_rows(connection, table: str) -> None:
    sql_table = quoted_table(table)
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT COUNT(*) FROM {sql_table} "
            "WHERE endpoint_hash IS NULL OR LENGTH(endpoint_hash) <> 32"
        )
        if int(cursor.fetchone()[0]) != 0:
            raise RuntimeError(
                "push_rules backfill validation failed: "
                "endpoint hashes are incomplete"
            )
        cursor.execute(
            "SELECT COUNT(*) FROM ("
            "SELECT endpoint_hash, facility_id, section_key, threshold, "
            f"active_identity FROM {sql_table} "
            "GROUP BY endpoint_hash, facility_id, "
            "section_key, threshold, active_identity HAVING active_identity = 1 "
            "AND COUNT(*) > 1) AS duplicate_identities"
        )
        if int(cursor.fetchone()[0]) != 0:
            raise RuntimeError(
                "push_rules backfill validation failed: duplicate identities remain"
            )


def inject_fault(
    fault_injector: FaultInjector | None, checkpoint: str
) -> None:
    if fault_injector is not None:
        fault_injector(checkpoint)


def ensure_constraint(
    cursor, table: str, name: str, statement: str
) -> None:
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.table_constraints "
        "WHERE table_schema = DATABASE() AND table_name = %s "
        "AND constraint_name = %s",
        (table, name),
    )
    if int(cursor.fetchone()[0]) == 0:
        cursor.execute(statement)


def ensure_index(cursor, table: str, name: str, statement: str) -> None:
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.statistics "
        "WHERE table_schema = DATABASE() AND table_name = %s "
        "AND index_name = %s",
        (table, name),
    )
    if int(cursor.fetchone()[0]) == 0:
        cursor.execute(statement)
