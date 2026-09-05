from __future__ import annotations

import sys as _import_sys
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence
import pymysql
from fastapi import HTTPException
from server.reclive.push_identity import endpoint_hash, normalize_push_endpoint
from server.reclive.runtime import current_runtime, runtime_scope
from server.reclive import db as _owner_db
from server.reclive import runtime as _owner_runtime
from server.reclive import sections as _owner_sections
from typing import Protocol


class ValidatedSubscription(Protocol):
    endpoint: str
    subscription: dict[str, Any]


class PushRuleRepository:
    """Transaction methods share the caller's already locked connection."""

    def __init__(self, connection, runtime=None):
        self.connection = connection
        self.runtime = runtime or current_runtime()

    def load_candidates(self, now):
        with runtime_scope(self.runtime):
            return load_evaluator_candidates(self.connection, now)

    def claim(self, rule_id, now):
        with runtime_scope(self.runtime):
            return claim_pending_rule(self.connection, rule_id, now)

    def finalize(self, rule_id, now, status, failure_code=None):
        with runtime_scope(self.runtime):
            return finalize_claimed_rule(
                self.connection, rule_id, now, status, failure_code
            )


PUSH_RULE_SELECT_COLUMNS = "\n    id,\n    endpoint_hash,\n    subscription_json,\n    facility_id,\n    section_key,\n    threshold,\n    created_at,\n    expires_at,\n    status,\n    active_identity\n"


def push_rules_table_name() -> str:
    return _owner_db.safe_sql_identifier(
        current_runtime().settings.push.table, "PUSH_RULES_TABLE"
    )


@dataclass(frozen=True)
class PushRuleRecord:
    id: int
    endpoint_hash: bytes
    subscription_json: object
    facility_id: int
    section_key: str
    threshold: int
    created_at: datetime
    expires_at: datetime
    status: str
    active_identity: int | None


def _mysql_utc_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("push rule timestamp must be UTC")
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=timezone.utc)
    if value.utcoffset() != timedelta(0):
        raise ValueError("push rule timestamp must be UTC")
    return value.astimezone(timezone.utc)


def _mysql_utc_bind(value: datetime) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
        or (value.utcoffset() != timedelta(0))
    ):
        raise ValueError("push rule timestamp must be aware UTC")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _push_rule_from_row(row: Sequence[object]) -> PushRuleRecord:
    if len(row) != 10:
        raise ValueError("invalid push rule row")
    (
        rule_id,
        stored_digest,
        subscription_json,
        facility_id,
        section_key,
        threshold,
        created_at,
        expires_at,
        status,
        active_identity,
    ) = row
    if type(rule_id) is not int or rule_id <= 0:
        raise ValueError("invalid push rule row")
    try:
        digest = bytes(stored_digest)
    except (TypeError, ValueError):
        raise ValueError("invalid push rule row") from None
    if len(digest) != 32:
        raise ValueError("invalid push rule row")
    if type(facility_id) is not int or facility_id not in {1186, 1656}:
        raise ValueError("invalid push rule row")
    if type(section_key) is not str:
        raise ValueError("invalid push rule row")
    if type(threshold) is not int or not 1 <= threshold <= 100:
        raise ValueError("invalid push rule row")
    if type(status) is not str:
        raise ValueError("invalid push rule row")
    if active_identity is not None and type(active_identity) is not int:
        raise ValueError("invalid push rule row")
    return PushRuleRecord(
        id=rule_id,
        endpoint_hash=digest,
        subscription_json=subscription_json,
        facility_id=facility_id,
        section_key=section_key,
        threshold=threshold,
        created_at=_mysql_utc_datetime(created_at),
        expires_at=_mysql_utc_datetime(expires_at),
        status=status,
        active_identity=active_identity,
    )


def _canonical_subscription_json(subscription: "ValidatedSubscription") -> str:
    return json.dumps(
        subscription.subscription, separators=(",", ":"), ensure_ascii=True
    )


def _safe_rollback(connection: Any) -> None:
    try:
        connection.rollback()
    except Exception:
        pass


def _safe_close_connection(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _rule_store_unavailable() -> HTTPException:
    return _push_http_error(503, "push_rule_store_unavailable")


def _endpoint_lock_name(digest: bytes) -> str:
    if len(digest) != 32:
        raise ValueError("invalid push endpoint digest")
    return f"reclive:push:{digest.hex()[:48]}"


def _acquire_endpoint_lock(cursor: Any, lock_name: str) -> None:
    cursor.execute("SELECT GET_LOCK(%s, 2)", (lock_name,))
    row = cursor.fetchone()
    if row and type(row[0]) is int and (row[0] == 1):
        return
    if row and type(row[0]) is int and (row[0] == 0):
        raise _push_http_error(503, "push_rule_store_busy")
    raise _rule_store_unavailable()


def _release_endpoint_lock(connection: Any, lock_name: str) -> None:
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
    except Exception:
        pass


def db_select_rule_by_id(cursor: Any, rule_id: int) -> PushRuleRecord | None:
    table_name = push_rules_table_name()
    cursor.execute(
        f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE id = %s LIMIT 1",
        (int(rule_id),),
    )
    row = cursor.fetchone()
    return _push_rule_from_row(row) if row is not None else None


def _select_active_identity_rule(
    cursor: Any, digest: bytes, facility_id: int, section_key: str, threshold: int
) -> PushRuleRecord | None:
    table_name = push_rules_table_name()
    cursor.execute(
        f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE endpoint_hash = %s AND facility_id = %s AND section_key = %s AND threshold = %s AND active_identity IS NOT NULL ORDER BY id DESC LIMIT 1 FOR UPDATE",
        (digest, facility_id, section_key, threshold),
    )
    row = cursor.fetchone()
    return _push_rule_from_row(row) if row is not None else None


def _expire_owned_pending_rules(
    cursor: Any,
    subscription: "ValidatedSubscription",
    digest: bytes,
    now_bound: datetime,
) -> int:
    table_name = push_rules_table_name()
    cursor.execute(
        f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE endpoint_hash = %s AND status = 'pending' AND active_identity IS NOT NULL AND expires_at <= %s FOR UPDATE",
        (digest, now_bound),
    )
    expired = 0
    for row in cursor.fetchall():
        try:
            rule = _push_rule_from_row(row)
        except ValueError:
            continue
        if not current_runtime().rule_ownership(rule, subscription.endpoint, digest):
            continue
        cursor.execute(
            f"UPDATE {table_name} SET status = 'expired', finalized_at = %s WHERE id = %s AND status = 'pending' AND expires_at <= %s",
            (now_bound, rule.id, now_bound),
        )
        expired += int(cursor.rowcount or 0)
    return expired


def db_rules_count() -> int:
    table_name = push_rules_table_name()
    conn = None
    try:
        conn = current_runtime().connect()
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row = cur.fetchone()
            return _owner_sections._int_or_default(row[0] if row else 0, 0)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail="Push rule store DB is unavailable"
        ) from exc
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def db_subscribe_rule(
    subscription: "ValidatedSubscription",
    facility_id: int,
    section_key: str,
    threshold: int,
    ttl_seconds: int | None,
) -> tuple[bool, PushRuleRecord]:
    table_name = push_rules_table_name()
    canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
    if not hmac.compare_digest(
        canonical_endpoint.encode("utf-8"), subscription.endpoint.encode("utf-8")
    ):
        raise _push_http_error(422, "invalid_push_subscription")
    normalized_key = _owner_sections.canonical_section_key(section_key)
    if (
        type(facility_id) is not int
        or facility_id not in {1186, 1656}
        or type(threshold) is not int
        or (not 1 <= threshold <= 100)
        or (normalized_key != section_key)
        or (not _owner_sections.location_ids_for_section(facility_id, normalized_key))
    ):
        raise _push_http_error(422, "invalid_push_request")
    ttl = (
        current_runtime().settings.push.default_rule_ttl_seconds
        if ttl_seconds is None
        else ttl_seconds
    )
    if (
        type(ttl) is not int
        or not 1 <= ttl <= current_runtime().settings.push.max_rule_ttl_seconds
    ):
        raise _push_http_error(422, "invalid_push_request")
    try:
        now = _owner_runtime.now_utc()
        now_bound = _mysql_utc_bind(now)
        expires_bound = _mysql_utc_bind(now + timedelta(seconds=ttl))
        digest = endpoint_hash(canonical_endpoint)
        lock_name = _endpoint_lock_name(digest)
        subscription_json = _canonical_subscription_json(subscription)
    except HTTPException:
        raise
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    locked = False
    committed = False
    try:
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            _acquire_endpoint_lock(cur, lock_name)
            locked = True
            _expire_owned_pending_rules(cur, subscription, digest, now_bound)
            existing = _select_active_identity_rule(
                cur, digest, facility_id, normalized_key, threshold
            )
            if existing is not None:
                if not current_runtime().rule_ownership(
                    existing, canonical_endpoint, digest
                ):
                    raise _push_http_error(409, "push_identity_conflict")
                if existing.status == "claimed":
                    raise _push_http_error(409, "push_rule_in_progress")
                if existing.status != "pending" or existing.expires_at <= now:
                    raise _push_http_error(409, "push_identity_conflict")
                conn.commit()
                committed = True
                return (False, existing)
            cur.execute(
                f"SELECT COUNT(*) FROM {table_name} WHERE endpoint_hash = %s AND active_identity IS NOT NULL AND status IN ('pending', 'claimed')",
                (digest,),
            )
            count_row = cur.fetchone()
            if not count_row or type(count_row[0]) is not int or count_row[0] < 0:
                raise RuntimeError("invalid active push rule count")
            if (
                count_row[0]
                >= current_runtime().settings.push.max_active_rules_per_endpoint
            ):
                raise _push_http_error(409, "push_rule_limit_reached")
            try:
                cur.execute(
                    f"\n                    INSERT INTO {table_name}\n                        (endpoint_hash, subscription_json, facility_id,\n                         section_key, threshold, created_at, expires_at, status)\n                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')\n                    ",
                    (
                        digest,
                        subscription_json,
                        facility_id,
                        normalized_key,
                        threshold,
                        now_bound,
                        expires_bound,
                    ),
                )
            except pymysql.err.IntegrityError as exc:
                if not exc.args or exc.args[0] != 1062:
                    raise
                recovered = _select_active_identity_rule(
                    cur, digest, facility_id, normalized_key, threshold
                )
                if recovered is None:
                    raise RuntimeError("active push identity unavailable") from None
                if not current_runtime().rule_ownership(
                    recovered, canonical_endpoint, digest
                ):
                    raise _push_http_error(409, "push_identity_conflict")
                if recovered.status == "claimed":
                    raise _push_http_error(409, "push_rule_in_progress")
                if recovered.status != "pending" or recovered.expires_at <= now:
                    raise _push_http_error(409, "push_identity_conflict")
                conn.commit()
                committed = True
                return (False, recovered)
            inserted_id = int(cur.lastrowid or 0)
            inserted = db_select_rule_by_id(cur, inserted_id)
            if inserted is None or not current_runtime().rule_ownership(
                inserted, canonical_endpoint, digest
            ):
                raise RuntimeError("inserted push rule unavailable")
        conn.commit()
        committed = True
        return (True, inserted)
    except HTTPException:
        if conn is not None and (not committed):
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None and (not committed):
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            if locked:
                _release_endpoint_lock(conn, lock_name)
            _safe_close_connection(conn)


def _db_list_owned_rule_records(
    subscription: "ValidatedSubscription",
) -> List[PushRuleRecord]:
    table_name = push_rules_table_name()
    try:
        canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
        digest = endpoint_hash(canonical_endpoint)
        now_bound = _mysql_utc_bind(_owner_runtime.now_utc())
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    try:
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE endpoint_hash = %s AND status = 'pending' AND active_identity IS NOT NULL AND expires_at > %s ORDER BY created_at, id",
                (digest, now_bound),
            )
            rows = cur.fetchall()
        owned: List[PushRuleRecord] = []
        for row in rows:
            try:
                rule = _push_rule_from_row(row)
            except ValueError:
                continue
            if current_runtime().rule_ownership(rule, canonical_endpoint, digest):
                owned.append(rule)
        conn.commit()
        return owned
    except Exception:
        if conn is not None:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            _safe_close_connection(conn)


def resolve_owned_rule(endpoint: str, rule_id: int) -> PushRuleRecord:
    try:
        canonical_endpoint = normalize_push_endpoint(endpoint)
        digest = endpoint_hash(canonical_endpoint)
    except Exception:
        raise _push_http_error(404, "push_rule_not_found") from None
    conn = None
    try:
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            table_name = push_rules_table_name()
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE endpoint_hash = %s AND id = %s LIMIT 1",
                (digest, int(rule_id)),
            )
            row = cur.fetchone()
            rule = _push_rule_from_row(row) if row is not None else None
        if rule is None or not current_runtime().rule_ownership(
            rule, canonical_endpoint, digest
        ):
            raise _push_http_error(404, "push_rule_not_found")
        conn.commit()
        return rule
    except HTTPException:
        if conn is not None:
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None:
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            _safe_close_connection(conn)


def db_cancel_owned_rule(subscription: "ValidatedSubscription", rule_id: int) -> int:
    table_name = push_rules_table_name()
    try:
        canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
        digest = endpoint_hash(canonical_endpoint)
        lock_name = _endpoint_lock_name(digest)
        now = _owner_runtime.now_utc()
        now_bound = _mysql_utc_bind(now)
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    locked = False
    committed = False
    try:
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            _acquire_endpoint_lock(cur, lock_name)
            locked = True
            _expire_owned_pending_rules(cur, subscription, digest, now_bound)
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE endpoint_hash = %s AND id = %s LIMIT 1 FOR UPDATE",
                (digest, int(rule_id)),
            )
            row = cur.fetchone()
            try:
                rule = _push_rule_from_row(row) if row is not None else None
            except ValueError:
                rule = None
            if (
                rule is None
                or rule.status != "pending"
                or rule.expires_at <= now
                or (
                    not current_runtime().rule_ownership(
                        rule, canonical_endpoint, digest
                    )
                )
            ):
                raise _push_http_error(404, "push_rule_not_found")
            cur.execute(
                f"UPDATE {table_name} SET status = 'cancelled', finalized_at = %s WHERE id = %s AND status = 'pending' AND expires_at > %s",
                (now_bound, rule.id, now_bound),
            )
            cancelled = int(cur.rowcount or 0)
            if cancelled != 1:
                raise _push_http_error(404, "push_rule_not_found")
        conn.commit()
        committed = True
    except HTTPException:
        if conn is not None and (not committed):
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None and (not committed):
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            if locked:
                _release_endpoint_lock(conn, lock_name)
            _safe_close_connection(conn)
    return cancelled


def db_cancel_all_owned_rules(subscription: "ValidatedSubscription") -> int:
    table_name = push_rules_table_name()
    try:
        canonical_endpoint = normalize_push_endpoint(subscription.endpoint)
        digest = endpoint_hash(canonical_endpoint)
        lock_name = _endpoint_lock_name(digest)
        now = _owner_runtime.now_utc()
        now_bound = _mysql_utc_bind(now)
    except Exception:
        raise _rule_store_unavailable() from None
    conn = None
    locked = False
    committed = False
    try:
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            _acquire_endpoint_lock(cur, lock_name)
            locked = True
            _expire_owned_pending_rules(cur, subscription, digest, now_bound)
            cur.execute(
                f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE endpoint_hash = %s AND status = 'pending' AND active_identity IS NOT NULL AND expires_at > %s ORDER BY id FOR UPDATE",
                (digest, now_bound),
            )
            cancelled = 0
            for row in cur.fetchall():
                try:
                    rule = _push_rule_from_row(row)
                except ValueError:
                    continue
                if not current_runtime().rule_ownership(
                    rule, canonical_endpoint, digest
                ):
                    continue
                cur.execute(
                    f"UPDATE {table_name} SET status = 'cancelled', finalized_at = %s WHERE id = %s AND status = 'pending' AND expires_at > %s",
                    (now_bound, rule.id, now_bound),
                )
                cancelled += int(cur.rowcount or 0)
        conn.commit()
        committed = True
        return cancelled
    except HTTPException:
        if conn is not None and (not committed):
            _safe_rollback(conn)
        raise
    except Exception:
        if conn is not None and (not committed):
            _safe_rollback(conn)
        raise _rule_store_unavailable() from None
    finally:
        if conn is not None:
            if locked:
                _release_endpoint_lock(conn, lock_name)
            _safe_close_connection(conn)


class PushEvaluatorStoreError(RuntimeError):
    pass


def db_acquire_evaluator_lock() -> Optional[Any]:
    conn = None
    try:
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT GET_LOCK(%s, 0)",
                (current_runtime().settings.push.evaluator_lock_name,),
            )
            row = cur.fetchone()
            valid_row = (
                isinstance(row, (list, tuple))
                and len(row) == 1
                and (type(row[0]) is int)
            )
            if valid_row and row[0] == 1:
                return conn
            if valid_row and row[0] == 0:
                _safe_close_connection(conn)
                return None
    except Exception:
        if conn is not None:
            _safe_close_connection(conn)
        raise PushEvaluatorStoreError("push evaluator lock unavailable") from None
    if conn is not None:
        _safe_close_connection(conn)
    raise PushEvaluatorStoreError("push evaluator lock result invalid")


def db_release_evaluator_lock(conn: Any) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT RELEASE_LOCK(%s)",
                (current_runtime().settings.push.evaluator_lock_name,),
            )
            row = cur.fetchone()
            if (
                not isinstance(row, (list, tuple))
                or len(row) != 1
                or type(row[0]) is not int
                or (row[0] != 1)
            ):
                raise PushEvaluatorStoreError("push evaluator lock release failed")
    except PushEvaluatorStoreError:
        raise
    except Exception:
        raise PushEvaluatorStoreError("push evaluator lock release failed") from None
    finally:
        _safe_close_connection(conn)


def push_db_available() -> bool:
    table_name = push_rules_table_name()
    conn = None
    try:
        conn = current_runtime().connect()
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            cur.fetchone()
        return True
    except Exception:
        return False
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def load_evaluator_candidates(conn: Any, now: datetime) -> List[PushRuleRecord]:
    table_name = push_rules_table_name()
    now_bound = _mysql_utc_bind(now)
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table_name} SET status = 'expired', finalized_at = %s WHERE status = 'pending' AND expires_at <= %s",
            (now_bound, now_bound),
        )
        cursor.execute(
            f"SELECT {PUSH_RULE_SELECT_COLUMNS} FROM {table_name} WHERE status = 'pending' AND active_identity IS NOT NULL AND expires_at > %s ORDER BY id",
            (now_bound,),
        )
        rows = cursor.fetchall()
    candidates: List[PushRuleRecord] = []
    for row in rows:
        try:
            rule = _push_rule_from_row(row)
        except (TypeError, ValueError):
            continue
        if (
            rule.status == "pending"
            and rule.active_identity is not None
            and (rule.expires_at > now)
        ):
            candidates.append(rule)
    return candidates


def claim_pending_rule(conn: Any, rule_id: int, now: datetime) -> bool:
    table_name = push_rules_table_name()
    now_bound = _mysql_utc_bind(now)
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table_name} SET status = 'claimed', claimed_at = %s WHERE id = %s AND status = 'pending' AND expires_at > %s",
            (now_bound, int(rule_id), now_bound),
        )
        return int(cursor.rowcount or 0) == 1


def finalize_claimed_rule(
    conn: Any, rule_id: int, now: datetime, status: str, failure_code: str | None = None
) -> bool:
    if status not in {"sent", "failed", "invalid_subscription"}:
        raise ValueError("invalid push terminal status")
    if status == "sent" and failure_code is not None:
        raise ValueError("sent rule cannot have a failure code")
    if status != "sent" and failure_code not in {
        "webpush_failed",
        "webpush_404",
        "webpush_410",
    }:
        raise ValueError("invalid push failure code")
    table_name = push_rules_table_name()
    now_bound = _mysql_utc_bind(now)
    sent_at = now_bound if status == "sent" else None
    with conn.cursor() as cursor:
        cursor.execute(
            f"UPDATE {table_name} SET status = %s, sent_at = %s, finalized_at = %s, failure_code = %s WHERE id = %s AND status = 'claimed'",
            (status, sent_at, now_bound, failure_code, int(rule_id)),
        )
        return int(cursor.rowcount or 0) == 1


def _push_http_error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


def record_public_push_write(subject_hash: bytes) -> None:
    conn = None
    try:
        sample = _owner_runtime.now_utc()
        if sample.tzinfo is None or sample.utcoffset() is None:
            raise ValueError("push clock must be timezone-aware")
        sample_utc = sample.astimezone(timezone.utc)
        epoch = int(sample_utc.timestamp())
        window_epoch = (
            epoch - epoch % current_runtime().settings.push.write_rate_window_seconds
        )
        window_started_at = datetime.fromtimestamp(
            window_epoch, tz=timezone.utc
        ).replace(tzinfo=None)
        updated_at = sample_utc.replace(tzinfo=None)
        conn = current_runtime().connect(autocommit=False)
        with conn.cursor() as cur:
            cur.execute(
                "\n                INSERT INTO push_rate_limits\n                    (subject_hash, window_started_at, request_count, updated_at)\n                VALUES (%s, %s, 1, %s)\n                ON DUPLICATE KEY UPDATE\n                    request_count = request_count + 1,\n                    updated_at = VALUES(updated_at)\n                ",
                (subject_hash, window_started_at, updated_at),
            )
            cur.execute(
                "\n                SELECT request_count\n                FROM push_rate_limits\n                WHERE subject_hash = %s AND window_started_at = %s\n                ",
                (subject_hash, window_started_at),
            )
            row = cur.fetchone()
            if not row or type(row[0]) is not int or row[0] < 1:
                raise RuntimeError("invalid push rate-limit counter result")
            request_count = row[0]
        conn.commit()
    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise _push_http_error(503, "push_rate_limit_store_unavailable") from None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    if request_count > current_runtime().settings.push.write_rate_limit:
        raise _push_http_error(429, "push_write_rate_limited")


_import_sys.modules.setdefault(
    "server.reclive.repositories.push_rules", _import_sys.modules[__name__]
)
_import_sys.modules.setdefault(
    "reclive.repositories.push_rules", _import_sys.modules[__name__]
)
