"""Immutable MariaDB 0003 timestamp preparation, executed from hashed bytes."""
from __future__ import annotations

import re
from datetime import datetime, timezone


TABLE = "_reclive_push_rules_cutover"
SHADOW = "_reclive_created_at_utc"
SHADOW_MARKER = "reclive:mariadb-created-at-utc:v1"
AWARE_ISO = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9])"
)


def utc_timestamp(value: object) -> datetime:
    try:
        if not isinstance(value, str) or AWARE_ISO.fullmatch(value) is None:
            raise ValueError
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        converted = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        if converted.year < 1000:
            raise ValueError
        return converted
    except (ValueError, OverflowError):
        raise RuntimeError("Invalid MariaDB legacy created_at timestamp") from None


def prepare_legacy_timestamps(connection, fault_injector=None) -> None:
    def checkpoint(name: str) -> None:
        if fault_injector is not None:
            fault_injector(f"mariadb_timestamps:{name}")

    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT data_type, character_maximum_length FROM information_schema.columns "
            "WHERE table_schema = DATABASE() AND table_name = %s AND column_name = 'created_at'",
            (TABLE,),
        )
        column = cursor.fetchone()
        if column is None or column[0] == "datetime":
            return
        if tuple(column) != ("varchar", 64):
            raise RuntimeError("Unsupported MariaDB legacy created_at column")
        cursor.execute(
            "SELECT COUNT(*) FROM information_schema.statistics "
            "WHERE table_schema = DATABASE() AND table_name = %s AND column_name = 'created_at'",
            (TABLE,),
        )
        if cursor.fetchone()[0]:
            raise RuntimeError("Unsupported MariaDB legacy created_at index")
        cursor.execute(f"SELECT id, created_at FROM `{TABLE}` ORDER BY id")
        source_rows = cursor.fetchall()
        # Parse every value before creating or writing a conversion column.
        converted_rows = [(rule_id, source, utc_timestamp(source)) for rule_id, source in source_rows]
        cursor.execute(
            "SELECT data_type, is_nullable, datetime_precision, column_comment "
            "FROM information_schema.columns WHERE table_schema = DATABASE() "
            "AND table_name = %s AND column_name = %s",
            (TABLE, SHADOW),
        )
        shadow = cursor.fetchone()
        if shadow is not None and tuple(shadow) != ("datetime", "YES", 6, SHADOW_MARKER):
            raise RuntimeError("Unexpected MariaDB legacy created_at conversion column")
        if shadow is None:
            cursor.execute(
                f"ALTER TABLE `{TABLE}` ADD COLUMN `{SHADOW}` DATETIME(6) NULL "
                f"COMMENT '{SHADOW_MARKER}', ALGORITHM=COPY, LOCK=EXCLUSIVE"
            )
        checkpoint("after_shadow_column")
        for rule_id, _source, converted in converted_rows:
            cursor.execute(
                f"UPDATE `{TABLE}` SET `{SHADOW}` = %s WHERE id = %s",
                (converted, rule_id),
            )
        checkpoint("after_shadow_updates")
        cursor.execute(f"SELECT id, created_at, `{SHADOW}` FROM `{TABLE}` ORDER BY id")
        if list(cursor.fetchall()) != converted_rows:
            raise RuntimeError("MariaDB legacy created_at conversion validation failed")
        checkpoint("before_swap")
        # One atomic DDL changes the column identity. A crash leaves either
        # the original strings plus a retryable shadow, or the complete UTC
        # DATETIME column; never partially rewritten timezone-free strings.
        cursor.execute(
            f"ALTER TABLE `{TABLE}` DROP COLUMN created_at, "
            f"CHANGE COLUMN `{SHADOW}` created_at DATETIME(6) NOT NULL COMMENT '', "
            "ALGORITHM=COPY, LOCK=EXCLUSIVE"
        )
        checkpoint("after_swap")
