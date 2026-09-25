"""Stable legacy imports and executable API entry point."""

import sys
import hmac as hmac
import json as json
import http as http
from pathlib import Path

if __package__ in {None, ""}:
    _project_root = str(Path(__file__).resolve().parent.parent)
    if _project_root not in sys.path and str(Path.cwd()) != _project_root:
        sys.path.insert(0, _project_root)

from server.reclive.api.app import get_default_app, create_app as create_app
from server.env_loader import (
    validate_production_environment as validate_production_environment,
)

from server.reclive.api.dependencies import (
    OwnedActualHourRepository as OwnedActualHourRepository,
    get_actual_hour_repository as get_actual_hour_repository,
    get_snapshot_repository as get_snapshot_repository,
    require_admin_token as require_admin_token,
)

from server.reclive.api.forecasts import (
    load_forecast as load_forecast,
    generated_age_seconds as generated_age_seconds,
    compact_hour_payload as compact_hour_payload,
    compact_window_payload as compact_window_payload,
    compact_day_payload as compact_day_payload,
    compact_facility_payload as compact_facility_payload,
    parse_chicago_date_key as parse_chicago_date_key,
    to_chicago_datetime as to_chicago_datetime,
    forecast as forecast,
    facilities as facilities,
    facility_forecast as facility_forecast,
    serialize_actual_hour as serialize_actual_hour,
    facility_actual_hours as facility_actual_hours,
)

from server.reclive.api.health import health as health, push_health as push_health

from server.reclive.api.lifespan_compat import lifespan as lifespan

from server.reclive.api.live_counts import (
    live_counts as live_counts,
    utc_iso as utc_iso,
)

from server.reclive.api.push import (
    public_key as public_key,
    push_availability as push_availability,
    subscribe as subscribe,
    push_rules_list as push_rules_list,
    push_rule_cancel as push_rule_cancel,
    push_rules_cancel_all as push_rules_cancel_all,
    dispatch as dispatch,
    evaluate as evaluate,
)

from server.reclive.api.schedules import (
    facility_hours as facility_hours,
    facility_hours_facilities as facility_hours_facilities,
    facility_hours_for_facility as facility_hours_for_facility,
)

from server.reclive.db import (
    open_db_connection as open_db_connection,
    safe_sql_identifier as safe_sql_identifier,
)

from server.reclive.facility_schedule import (
    load_facility_hours as load_facility_hours,
    schedule_health as schedule_health,
    _parse_facility_id as _parse_facility_id,
    get_facility_hours_entry as get_facility_hours_entry,
)

from server.reclive.push import (
    PushRuleResponse as PushRuleResponse,
    push_rule_response as push_rule_response,
    _decode_stored_subscription as _decode_stored_subscription,
    _canonical_stored_endpoint as _canonical_stored_endpoint,
    _rule_is_owned as _rule_is_owned,
    SafePushDispatchError as SafePushDispatchError,
    PushProviderStatusError as PushProviderStatusError,
    _PushAttemptDeadline as _PushAttemptDeadline,
    _BoundedPushExecutor as _BoundedPushExecutor,
    _get_push_executor as _get_push_executor,
    PinnedPushTarget as PinnedPushTarget,
    SafePushResponse as SafePushResponse,
    resolve_endpoint_host as resolve_endpoint_host,
    _is_safe_push_address as _is_safe_push_address,
    resolve_public_push_addresses as resolve_public_push_addresses,
    build_pinned_push_target as build_pinned_push_target,
    PinnedPushSession as PinnedPushSession,
    _send_notification_pinned_attempt as _send_notification_pinned_attempt,
    send_notification_pinned as send_notification_pinned,
    push_identity_configured as push_identity_configured,
    index_live_rows as index_live_rows,
    is_aware_utc_datetime as is_aware_utc_datetime,
    is_fresh_snapshot as is_fresh_snapshot,
    round_nonnegative_percent as round_nonnegative_percent,
    compute_section_metrics as compute_section_metrics,
    compute_fresh_section_metrics as compute_fresh_section_metrics,
    facility_notification_url as facility_notification_url,
    _evaluator_store_unavailable as _evaluator_store_unavailable,
    _base_evaluator_result as _base_evaluator_result,
    _sample_evaluator_now as _sample_evaluator_now,
    _fresh_ingestion_at as _fresh_ingestion_at,
    _rule_schedule_gate as _rule_schedule_gate,
    _rule_snapshot_gate as _rule_snapshot_gate,
    _provider_failure_state as _provider_failure_state,
    evaluate_rules_once as evaluate_rules_once,
    evaluator_loop as evaluator_loop,
    IgnoreExtraModel as IgnoreExtraModel,
    StrictPushModel as StrictPushModel,
    PushKeysInput as PushKeysInput,
    _is_safe_expiration_time as _is_safe_expiration_time,
    PushSubscriptionInput as PushSubscriptionInput,
    PushRuleRequest as PushRuleRequest,
    PushOwnershipRequest as PushOwnershipRequest,
    ValidatedSubscription as ValidatedSubscription,
    _content_length_value as _content_length_value,
    _reject_json_constant as _reject_json_constant,
    _read_limited_push_json as _read_limited_push_json,
    _validate_push_model as _validate_push_model,
    parse_limited_push_body as parse_limited_push_body,
    _decoded_subscription_endpoint as _decoded_subscription_endpoint,
    rate_limit_public_push_write as rate_limit_public_push_write,
    _parse_limited_push_write as _parse_limited_push_write,
    _parse_push_rule_id as _parse_push_rule_id,
    _strict_base64url_decode as _strict_base64url_decode,
    validate_push_subscription as validate_push_subscription,
    _validated_subscription_from_model as _validated_subscription_from_model,
    subscribe_owned_push_rule as subscribe_owned_push_rule,
    list_owned_push_rules as list_owned_push_rules,
    cancel_owned_push_rule as cancel_owned_push_rule,
    cancel_all_owned_push_rules as cancel_all_owned_push_rules,
    PushDispatchRequest as PushDispatchRequest,
)

from server.reclive.repositories.push_rules import (
    PushRuleRecord as PushRuleRecord,
    _mysql_utc_datetime as _mysql_utc_datetime,
    _mysql_utc_bind as _mysql_utc_bind,
    _push_rule_from_row as _push_rule_from_row,
    _canonical_subscription_json as _canonical_subscription_json,
    _safe_rollback as _safe_rollback,
    _safe_close_connection as _safe_close_connection,
    _rule_store_unavailable as _rule_store_unavailable,
    _endpoint_lock_name as _endpoint_lock_name,
    _acquire_endpoint_lock as _acquire_endpoint_lock,
    _release_endpoint_lock as _release_endpoint_lock,
    db_select_rule_by_id as db_select_rule_by_id,
    _select_active_identity_rule as _select_active_identity_rule,
    _expire_owned_pending_rules as _expire_owned_pending_rules,
    db_rules_count as db_rules_count,
    db_subscribe_rule as db_subscribe_rule,
    _db_list_owned_rule_records as _db_list_owned_rule_records,
    resolve_owned_rule as resolve_owned_rule,
    db_cancel_owned_rule as db_cancel_owned_rule,
    db_cancel_all_owned_rules as db_cancel_all_owned_rules,
    PushEvaluatorStoreError as PushEvaluatorStoreError,
    db_acquire_evaluator_lock as db_acquire_evaluator_lock,
    db_release_evaluator_lock as db_release_evaluator_lock,
    push_db_available as push_db_available,
    load_evaluator_candidates as load_evaluator_candidates,
    claim_pending_rule as claim_pending_rule,
    finalize_claimed_rule as finalize_claimed_rule,
    push_rules_table_name as push_rules_table_name,
    _push_http_error as _push_http_error,
)

from server.reclive.runtime import now_utc as now_utc, now_iso as now_iso

from server.reclive.sections import (
    load_facility_sections as load_facility_sections,
    load_runtime_facility_configuration as load_runtime_facility_configuration,
    ensure_runtime_facility_configuration as ensure_runtime_facility_configuration,
    category_location_ids_for_forecast as category_location_ids_for_forecast,
    canonical_section_key as canonical_section_key,
    location_ids_for_section as location_ids_for_section,
    _int_or_default as _int_or_default,
    _str_or_none as _str_or_none,
)

from server.reclive.settings import (
    _read_env as _read_env,
    require_env as require_env,
    env_with_default as env_with_default,
    int_with_default as int_with_default,
    positive_int_with_legacy_alias as positive_int_with_legacy_alias,
    bool_with_default as bool_with_default,
    path_with_default as path_with_default,
    resolve_path as resolve_path,
    app_environment as app_environment,
    push_admin_routes_enabled as push_admin_routes_enabled,
    _validated_admin_token_bytes as _validated_admin_token_bytes,
    validate_push_configuration as validate_push_configuration,
    parse_allowed_origins as parse_allowed_origins,
    evaluator_enabled as evaluator_enabled,
    push_vapid_configured as push_vapid_configured,
    push_admin_configured as push_admin_configured,
    get_vapid_public_key as get_vapid_public_key,
    get_vapid_private_key as get_vapid_private_key,
    get_vapid_claims as get_vapid_claims,
)

from server.reclive.runtime import current_runtime
from server.reclive import push as _push
from server.reclive.settings import SERVER_ROOT

from datetime import datetime as datetime, timedelta as timedelta, timezone as timezone
import asyncio as asyncio
import threading as threading
import pymysql as pymysql
import socket as socket
import ssl as ssl
import time as time
from pywebpush import WebPushException as WebPushException
from server.forecast_shared import normalize_section_key as normalize_section_key
from server.reclive.push_identity import (
    endpoint_hash as endpoint_hash,
    normalize_push_endpoint as normalize_push_endpoint,
    rate_limit_subject_hash as rate_limit_subject_hash,
)
from server.reclive.occupancy_repository import (
    SnapshotRepository as SnapshotRepository,
    SnapshotRow as SnapshotRow,
)
from server.reclive.facility_schedule import (
    official_facility_is_open as official_facility_is_open,
    parse_utc_timestamp as parse_utc_timestamp,
)

app = get_default_app()
SCRIPT_DIR = str(SERVER_ROOT)

FORECAST_JSON_PATH = current_runtime().settings.forecast_json_path

API_HOST = current_runtime().settings.host

API_PORT = current_runtime().settings.port

DB_TIMEZONE_NAME = current_runtime().settings.database.timezone

ACTUAL_HOUR_MIN_COVERAGE = current_runtime().settings.actual_hour_min_coverage

FACILITY_SECTION_CONFIG_PATH = current_runtime().settings.facility_section_config_path

FACILITY_HOURS_JSON_PATH = current_runtime().settings.facility_hours_json_path

SCHEDULE_STALE_AFTER_SECONDS = current_runtime().settings.schedule_stale_after_seconds

MAX_CAP = current_runtime().capacities

FACILITY_NAMES = current_runtime().facility_names

SECTION_IDS = current_runtime().section_ids

_FACILITY_CONFIGURATION_LOADED = current_runtime().configuration_loaded

_FACILITY_CONFIGURATION_LOCK = current_runtime().configuration_lock

EVALUATOR_TASK = current_runtime().task

_push_executor = current_runtime().executor

_push_executor_lock = current_runtime().transport_lock

PUSH_RULES_TABLE = current_runtime().settings.push.table

PUSH_BODY_MAX_BYTES = current_runtime().settings.push.body_max_bytes

PUSH_DEFAULT_RULE_TTL_SECONDS = current_runtime().settings.push.default_rule_ttl_seconds

PUSH_MAX_RULE_TTL_SECONDS = current_runtime().settings.push.max_rule_ttl_seconds

PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT = (
    current_runtime().settings.push.max_active_rules_per_endpoint
)

PUSH_WRITE_RATE_LIMIT = current_runtime().settings.push.write_rate_limit

PUSH_WRITE_RATE_WINDOW_SECONDS = (
    current_runtime().settings.push.write_rate_window_seconds
)

EVALUATOR_INTERVAL_SECONDS = current_runtime().settings.push.evaluator_interval_seconds

PUSH_EVALUATOR_DB_LOCK_NAME = current_runtime().settings.push.evaluator_lock_name

PUSH_VAPID_PUBLIC_KEY = current_runtime().settings.push.vapid_public_key

PUSH_VAPID_PRIVATE_KEY = current_runtime().settings.push.vapid_private_key

PUSH_VAPID_SUBJECT = current_runtime().settings.push.vapid_subject

PUSH_ADMIN_TOKEN = current_runtime().settings.push.admin_token

PUSH_PROVIDER_RESPONSE_MAX_BYTES = _push.PUSH_PROVIDER_RESPONSE_MAX_BYTES

PUSH_TRANSPORT_TIMEOUT_SECONDS = _push.PUSH_TRANSPORT_TIMEOUT_SECONDS

PUSH_TRANSPORT_DEADLINE_SECONDS = _push.PUSH_TRANSPORT_DEADLINE_SECONDS

PUSH_TRANSPORT_WORKER_COUNT = _push.PUSH_TRANSPORT_WORKER_COUNT

PUSH_TRANSPORT_QUEUE_CAPACITY = _push.PUSH_TRANSPORT_QUEUE_CAPACITY

OCCUPANCY_FRESHNESS = _push.OCCUPANCY_FRESHNESS


def main():
    import uvicorn

    uvicorn.run("forecast_api:app", host=API_HOST, port=API_PORT, reload=False)


if __name__ == "__main__":
    main()
else:
    sys.modules.setdefault("forecast_api", sys.modules[__name__])
    sys.modules.setdefault("server.forecast_api", sys.modules[__name__])
    setattr(sys.modules["server"], "forecast_api", sys.modules[__name__])
