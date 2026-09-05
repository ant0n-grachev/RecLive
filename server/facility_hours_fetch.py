"""Executable compatibility entry point for the official schedule service."""

import argparse
import os
import sys
from dataclasses import replace

if not __package__:
    import reclive  # noqa: F401 - initialize the canonical parent for direct scripts.

from server.env_loader import load_project_dotenv, validate_production_environment
from server.reclive.settings import Settings
from server.reclive.facility_schedule import (
    SCRIPT_DIR as SCRIPT_DIR,
    USER_AGENT as USER_AGENT,
    DEFAULT_OUTPUT_FILE as DEFAULT_OUTPUT_FILE,
    DEFAULT_SITE_BASE as DEFAULT_SITE_BASE,
    DEFAULT_FACILITIES as DEFAULT_FACILITIES,
    SUPPORTED_FACILITY_IDENTITIES as SUPPORTED_FACILITY_IDENTITIES,
    DAY_HINT_RE as DAY_HINT_RE,
    DATE_LABEL_RE as DATE_LABEL_RE,
    HOURS_HINT_RE as HOURS_HINT_RE,
    NOTICE_HOURS_RE as NOTICE_HOURS_RE,
    NOTICE_MAINTENANCE_RE as NOTICE_MAINTENANCE_RE,
    TABLE_RE as TABLE_RE,
    DEFINITION_LIST_RE as DEFINITION_LIST_RE,
    DEFINITION_PAIR_RE as DEFINITION_PAIR_RE,
    ROW_RE as ROW_RE,
    CELL_RE as CELL_RE,
    HEADING_RE as HEADING_RE,
    TAG_RE as TAG_RE,
    SCHEDULE_ERROR_CATEGORIES as SCHEDULE_ERROR_CATEGORIES,
    SCHEDULE_SOURCES as SCHEDULE_SOURCES,
    SCHEDULE_REFRESH_ERROR as SCHEDULE_REFRESH_ERROR,
    FACILITY_RECORD_KEYS as FACILITY_RECORD_KEYS,
    FACILITY_RECORD_REQUIRED_KEYS as FACILITY_RECORD_REQUIRED_KEYS,
    SCHEDULE_PAYLOAD_KEYS as SCHEDULE_PAYLOAD_KEYS,
    MAX_SCHEDULE_SECTIONS as MAX_SCHEDULE_SECTIONS,
    MAX_SCHEDULE_ROWS as MAX_SCHEDULE_ROWS,
    MAX_SCHEDULE_URL_LENGTH as MAX_SCHEDULE_URL_LENGTH,
    UNSAFE_URL_CHARACTER_RE as UNSAFE_URL_CHARACTER_RE,
    UNSAFE_ENCODED_URL_CHARACTER_RE as UNSAFE_ENCODED_URL_CHARACTER_RE,
    LOCAL_ISO_TIMESTAMP_RE as LOCAL_ISO_TIMESTAMP_RE,
    ScheduleFetchError as ScheduleFetchError,
    safe_error_message as safe_error_message,
    _bounded_text as _bounded_text,
    _https_origin as _https_origin,
    safe_same_origin_https_url as safe_same_origin_https_url,
    _safe_current_public_url as _safe_current_public_url,
    _valid_local_iso_timestamp as _valid_local_iso_timestamp,
    _copy_valid_sections as _copy_valid_sections,
    _parse_canonical_utc_timestamp as _parse_canonical_utc_timestamp,
    _aware_datetime as _aware_datetime,
    FacilityCandidate as FacilityCandidate,
    env_with_default as env_with_default,
    resolve_path as resolve_path,
    iso_utc as iso_utc,
    now_utc as now_utc,
    now_iso as now_iso,
    _copy_valid_previous_facility as _copy_valid_previous_facility,
    previous_is_valid_facility as previous_is_valid_facility,
    merge_facility_candidate as merge_facility_candidate,
    _copy_json_schedule_sections as _copy_json_schedule_sections,
    _safe_source_site as _safe_source_site,
    _normalized_facility_record as _normalized_facility_record,
    validate_schedule_payload as validate_schedule_payload,
    clean_text as clean_text,
    is_header_row as is_header_row,
    looks_like_hours_row as looks_like_hours_row,
    strip_html as strip_html,
    extract_rows_from_table as extract_rows_from_table,
    find_table_heading as find_table_heading,
    dedupe_sections as dedupe_sections,
    parse_hours_sections_with_bs4 as parse_hours_sections_with_bs4,
    find_heading_before_table as find_heading_before_table,
    parse_hours_sections_with_regex as parse_hours_sections_with_regex,
    parse_schedule_sections as parse_schedule_sections,
    parse_hours_sections as parse_hours_sections,
    looks_like_bot_challenge as looks_like_bot_challenge,
    parse_direct_response as parse_direct_response,
    parse_wp_page_payload as parse_wp_page_payload,
    fetch_direct_html as fetch_direct_html,
    fetch_wp_json_html as fetch_wp_json_html,
    _validated_candidate_identity as _validated_candidate_identity,
    collect_facility_candidate as collect_facility_candidate,
    build_combined_payload as build_combined_payload,
    collect_facility_hours as collect_facility_hours,
    build_facilities as build_facilities,
    _cleanup_atomic_file as _cleanup_atomic_file,
    atomic_write_json as atomic_write_json,
    write_json as write_json,
    _reject_duplicate_json_keys as _reject_duplicate_json_keys,
    load_previous_schedule as load_previous_schedule,
    main_for_output as main_for_output,
    BeautifulSoup as BeautifulSoup,
    requests as requests,
    run_facility_hours_fetch as run_facility_hours_fetch,
)


def main() -> int:
    load_project_dotenv()
    validate_production_environment(
        os.environ,
        required_names=("FACILITY_HOURS_JSON_PATH",),
        cors_name=None,
        admin_enabled=False,
    )
    parser = argparse.ArgumentParser(
        description="Fetch and parse official RecWell hours for Nick and Bakke."
    )
    parser.add_argument(
        "--output",
        default=env_with_default("FACILITY_HOURS_JSON_PATH", DEFAULT_OUTPUT_FILE),
        help="Output JSON file path relative to this script directory (default: facility_hours.json).",
    )
    args = parser.parse_args()

    output_path = resolve_path(str(args.output))
    try:
        settings = replace(
            Settings.for_commands(), facility_hours_json_path=output_path
        )
        exit_code = run_facility_hours_fetch(settings)
        payload = load_previous_schedule(output_path)
        if payload is None:
            raise ValueError("invalid schedule payload")
    except ScheduleFetchError as exc:
        print(
            f"facility_hours_fetch: failed category={exc.category}",
            file=sys.stderr,
        )
        return 1
    except Exception:
        print(
            "facility_hours_fetch: failed category=schema_invalid",
            file=sys.stderr,
        )
        return 1

    facility_payloads = payload["facilities"]
    ok_count = sum(1 for row in facility_payloads if row.get("status") == "ok")
    stale_count = sum(1 for row in facility_payloads if row.get("status") == "stale")
    error_count = sum(1 for row in facility_payloads if row.get("status") == "error")
    print(
        "facility_hours_fetch: published"
        f" ok={ok_count} stale={stale_count} error={error_count}"
        f" total={len(facility_payloads)}"
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

sys.modules.setdefault("server.facility_hours_fetch", sys.modules[__name__])
sys.modules.setdefault("facility_hours_fetch", sys.modules[__name__])
setattr(sys.modules["server"], "facility_hours_fetch", sys.modules[__name__])
