from __future__ import annotations

import builtins
import importlib
import json
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

import facility_hours_fetch
import facility_schedule
import forecast_api


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tests" / "fixtures"
FIXED_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 8, 31, 17, 0, tzinfo=timezone.utc)


@pytest.fixture
def fixture_text() -> Callable[[str], str]:
    def load(relative_path: str) -> str:
        return (FIXTURE_ROOT / relative_path).read_text(encoding="utf-8")

    return load


@pytest.fixture
def fixture_json() -> Callable[[str], object]:
    def load(relative_path: str) -> object:
        return json.loads(
            (FIXTURE_ROOT / relative_path).read_text(encoding="utf-8")
        )

    return load


def facility_config(facility_id: int) -> dict[str, object]:
    identity = {
        1186: ("Nick", "nick"),
        1656: ("Bakke", "bakke"),
    }
    facility_name, slug = identity[facility_id]
    return {
        "facilityId": facility_id,
        "facilityName": facility_name,
        "slug": slug,
        "url": f"https://recwell.example.test/locations/{slug}/",
    }


def schedule_section(title: str = "Building Hours") -> dict[str, object]:
    return {
        "title": title,
        "rows": [{"label": "Mon-Fri", "hours": "6:00 am - 10:00 pm"}],
        "note": None,
    }


def candidate_for(
    facility_id: int,
    *,
    source: str | None,
    error_category: str | None,
    title: str = "Building Hours",
    fetched_at: datetime = FIXED_NOW,
    source_modified_gmt: str | None = None,
) -> facility_hours_fetch.FacilityCandidate:
    config = facility_config(facility_id)
    sections = (schedule_section(title),) if source is not None else ()
    return facility_hours_fetch.FacilityCandidate(
        facility_id=facility_id,
        facility_name=str(config["facilityName"]),
        slug=str(config["slug"]),
        public_url=str(config["url"]),
        source=source,
        resolved_url=str(config["url"]) if source is not None else None,
        source_modified_gmt=source_modified_gmt,
        sections=sections,
        fetched_at=fetched_at,
        error_category=error_category,
    )


def valid_previous_facility(
    facility_id: int = 1186,
    *,
    source: str = "direct_html",
) -> dict[str, object]:
    config = facility_config(facility_id)
    successful_at = FIXED_NOW - timedelta(hours=1)
    return {
        "facilityId": facility_id,
        "facilityName": config["facilityName"],
        "slug": config["slug"],
        "url": config["url"],
        "resolvedUrl": config["url"],
        "status": "ok",
        "source": source,
        "sourceModifiedGmt": (
            "2026-09-01T10:55:00" if source == "wp_json" else None
        ),
        "sections": [schedule_section("Previous Building Hours")],
        "sourceFetchedAt": facility_hours_fetch.iso_utc(successful_at),
        "lastSuccessfulAt": facility_hours_fetch.iso_utc(successful_at),
        "stale": False,
        "error": None,
        "errorCategory": None,
        "updatedAt": facility_hours_fetch.iso_utc(successful_at),
    }


def valid_schedule_payload() -> dict[str, object]:
    return {
        "generatedAt": "2026-09-01T12:00:00Z",
        "sourceSite": "https://recwell.example.test",
        "facilities": [
            valid_previous_facility(1186),
            valid_previous_facility(1656),
        ],
        "okCount": 2,
        "totalCount": 2,
    }


def chicago_datetime(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/Chicago"))


@pytest.fixture
def schedule_test_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[[Mapping[str, object], datetime], TestClient]:
    def build(payload: Mapping[str, object], now: datetime) -> TestClient:
        schedule_path = tmp_path / "facility_hours.json"
        schedule_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(
            forecast_api, "FACILITY_HOURS_JSON_PATH", str(schedule_path)
        )
        monkeypatch.setattr(forecast_api, "now_utc", lambda: now, raising=False)
        monkeypatch.setattr(
            forecast_api,
            "load_forecast",
            lambda: {
                "generatedAt": facility_hours_fetch.iso_utc(now),
                "facilities": [],
                "modelInfo": {"status": "fixture"},
            },
        )
        monkeypatch.setattr(forecast_api, "evaluator_enabled", lambda: False)
        return TestClient(forecast_api.app)

    return build


def test_schedule_api_exposes_safe_freshness_metadata(
    schedule_test_client: Callable[[Mapping[str, object], datetime], TestClient],
) -> None:
    response = schedule_test_client(valid_schedule_payload(), FIXED_NOW).get(
        "/api/facility-hours/facilities/1186"
    )

    assert response.status_code == 200
    assert response.json()["stale"] is False
    assert response.json()["lastSuccessfulAt"] == "2026-09-01T11:00:00Z"
    assert "exception" not in response.text.lower()


def test_health_marks_old_or_stale_schedule_without_paths(
    schedule_test_client: Callable[[Mapping[str, object], datetime], TestClient],
) -> None:
    payload = valid_schedule_payload()
    facilities = payload["facilities"]
    assert isinstance(facilities, list)
    mark_facility_stale(facilities[0])
    payload["okCount"] = 1
    now = FIXED_NOW + timedelta(seconds=21_601)

    response = schedule_test_client(payload, now).get("/health")

    assert response.status_code == 200
    assert response.json()["schedule"] == {
        "state": "stale",
        "ageSeconds": 21_601,
        "facilities": {"1186": "stale", "1656": "ok"},
    }
    assert "facility_hours.json" not in response.text


def test_schedule_health_marks_fractionally_old_artifact_stale() -> None:
    health = forecast_api.schedule_health(
        valid_schedule_payload(),
        FIXED_NOW + timedelta(seconds=21_600, microseconds=900_000),
    )

    assert health == {
        "state": "stale",
        "ageSeconds": 21_600,
        "facilities": {"1186": "ok", "1656": "ok"},
    }


@pytest.mark.parametrize("artifact", ["missing", "unreadable", "invalid"])
def test_schedule_artifact_failures_share_stable_503_category(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact: str,
) -> None:
    path = tmp_path / "facility_hours.json"
    if artifact == "unreadable":
        path.mkdir()
    elif artifact == "invalid":
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(forecast_api, "FACILITY_HOURS_JSON_PATH", str(path))
    monkeypatch.setattr(forecast_api, "now_utc", lambda: FIXED_NOW, raising=False)

    response = TestClient(forecast_api.app).get("/api/facility-hours")

    assert response.status_code == 503
    assert response.json() == {"detail": "schedule_unavailable"}
    assert str(path) not in response.text


def test_health_reports_unavailable_schedule_without_throwing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "facility_hours.json"
    monkeypatch.setattr(
        forecast_api, "FACILITY_HOURS_JSON_PATH", str(missing_path)
    )
    monkeypatch.setattr(forecast_api, "now_utc", lambda: FIXED_NOW, raising=False)
    monkeypatch.setattr(
        forecast_api,
        "load_forecast",
        lambda: {
            "generatedAt": "2026-09-01T12:00:00Z",
            "facilities": [],
            "modelInfo": {"status": "fixture"},
        },
    )

    response = TestClient(forecast_api.app).get("/health")

    assert response.status_code == 200
    assert response.json()["schedule"] == {
        "state": "unavailable",
        "ageSeconds": None,
        "facilities": {},
    }
    assert str(missing_path) not in response.text


def test_schedule_stale_setting_prefers_valid_canonical_without_reading_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHEDULE_STALE_AFTER_SECONDS", "3600")
    monkeypatch.setenv("SCHEDULE_MAX_AGE_SECONDS", "private-invalid-value")

    assert forecast_api.positive_int_with_legacy_alias(
        "SCHEDULE_STALE_AFTER_SECONDS", "SCHEDULE_MAX_AGE_SECONDS", 21_600
    ) == 3600


@pytest.mark.parametrize(
    ("canonical", "legacy", "expected"),
    [(None, "7200", 7200), (None, None, 21_600)],
)
def test_schedule_stale_setting_uses_legacy_only_as_fallback(
    monkeypatch: pytest.MonkeyPatch,
    canonical: str | None,
    legacy: str | None,
    expected: int,
) -> None:
    monkeypatch.delenv("SCHEDULE_STALE_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("SCHEDULE_MAX_AGE_SECONDS", raising=False)
    if canonical is not None:
        monkeypatch.setenv("SCHEDULE_STALE_AFTER_SECONDS", canonical)
    if legacy is not None:
        monkeypatch.setenv("SCHEDULE_MAX_AGE_SECONDS", legacy)

    assert forecast_api.positive_int_with_legacy_alias(
        "SCHEDULE_STALE_AFTER_SECONDS", "SCHEDULE_MAX_AGE_SECONDS", 21_600
    ) == expected


@pytest.mark.parametrize(
    ("canonical", "legacy", "name"),
    [
        ("0", None, "SCHEDULE_STALE_AFTER_SECONDS"),
        ("canonical-secret", None, "SCHEDULE_STALE_AFTER_SECONDS"),
        (None, "bad-secret", "SCHEDULE_MAX_AGE_SECONDS"),
    ],
)
def test_schedule_stale_setting_rejects_nonpositive_or_invalid_without_value(
    monkeypatch: pytest.MonkeyPatch,
    canonical: str | None,
    legacy: str | None,
    name: str,
) -> None:
    monkeypatch.delenv("SCHEDULE_STALE_AFTER_SECONDS", raising=False)
    monkeypatch.delenv("SCHEDULE_MAX_AGE_SECONDS", raising=False)
    if canonical is not None:
        monkeypatch.setenv("SCHEDULE_STALE_AFTER_SECONDS", canonical)
    if legacy is not None:
        monkeypatch.setenv("SCHEDULE_MAX_AGE_SECONDS", legacy)

    with pytest.raises(RuntimeError) as captured:
        forecast_api.positive_int_with_legacy_alias(
            "SCHEDULE_STALE_AFTER_SECONDS", "SCHEDULE_MAX_AGE_SECONDS", 21_600
        )

    assert name in str(captured.value)
    assert "bad-secret" not in str(captured.value)
    assert "canonical-secret" not in str(captured.value)


def mark_facility_stale(record: dict[str, object]) -> None:
    record.update(
        {
            "status": "stale",
            "stale": True,
            "error": "Official hours could not be refreshed.",
            "errorCategory": "upstream_timeout",
        }
    )


def mark_facility_error(record: dict[str, object]) -> None:
    record.update(
        {
            "resolvedUrl": None,
            "status": "error",
            "source": None,
            "sourceModifiedGmt": None,
            "sections": [],
            "sourceFetchedAt": None,
            "lastSuccessfulAt": None,
            "stale": False,
            "error": "Official hours could not be refreshed.",
            "errorCategory": "upstream_http",
        }
    )


def test_one_failed_facility_keeps_valid_previous_while_other_updates() -> None:
    previous_nick = valid_previous_facility(1186)
    nick = facility_hours_fetch.merge_facility_candidate(
        candidate_for(
            1186,
            source=None,
            error_category="anti_bot",
        ),
        previous_nick,
        FIXED_NOW,
    )
    bakke = facility_hours_fetch.merge_facility_candidate(
        candidate_for(
            1656,
            source="direct_html",
            error_category=None,
            title="Bakke fresh rows",
        ),
        None,
        FIXED_NOW,
    )

    assert (
        nick["status"],
        nick["stale"],
        nick["sections"],
        nick["lastSuccessfulAt"],
    ) == (
        "stale",
        True,
        previous_nick["sections"],
        previous_nick["lastSuccessfulAt"],
    )
    assert nick["error"] == "Official hours could not be refreshed."
    assert nick["errorCategory"] == "anti_bot"
    assert (
        bakke["status"],
        bakke["stale"],
        bakke["sections"][0]["title"],
    ) == ("ok", False, "Bakke fresh rows")
    assert bakke["sourceFetchedAt"] == "2026-09-01T12:00:00Z"
    assert bakke["lastSuccessfulAt"] == "2026-09-01T12:00:00Z"


def test_failed_candidate_without_valid_previous_emits_no_invented_schedule() -> None:
    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(
            1186,
            source=None,
            error_category="upstream_timeout",
        ),
        None,
        FIXED_NOW,
    )

    assert result == {
        **facility_config(1186),
        "resolvedUrl": None,
        "status": "error",
        "source": None,
        "sourceModifiedGmt": None,
        "sections": [],
        "sourceFetchedAt": None,
        "lastSuccessfulAt": None,
        "stale": False,
        "error": "Official hours could not be refreshed.",
        "errorCategory": "upstream_timeout",
        "updatedAt": "2026-09-01T12:00:00Z",
    }


def test_consecutive_failures_preserve_an_already_stale_valid_record() -> None:
    first_stale = facility_hours_fetch.merge_facility_candidate(
        candidate_for(1186, source=None, error_category="anti_bot"),
        valid_previous_facility(1186),
        FIXED_NOW,
    )
    later = FIXED_NOW + timedelta(hours=1)

    second_stale = facility_hours_fetch.merge_facility_candidate(
        candidate_for(
            1186,
            source=None,
            error_category="upstream_http",
            fetched_at=later,
        ),
        first_stale,
        later,
    )

    assert second_stale["status"] == "stale"
    assert second_stale["stale"] is True
    assert second_stale["sections"] == first_stale["sections"]
    assert second_stale["sourceFetchedAt"] == first_stale["sourceFetchedAt"]
    assert second_stale["lastSuccessfulAt"] == first_stale["lastSuccessfulAt"]
    assert second_stale["errorCategory"] == "upstream_http"
    assert second_stale["updatedAt"] == "2026-09-01T13:00:00Z"


def test_merge_copies_previous_sections_without_sharing_nested_mutables() -> None:
    previous = valid_previous_facility(1186)
    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(1186, source=None, error_category="parse_empty"),
        previous,
        FIXED_NOW,
    )

    assert result["sections"] == previous["sections"]
    assert result["sections"] is not previous["sections"]
    assert result["sections"][0] is not previous["sections"][0]
    result["sections"][0]["rows"][0]["hours"] = "Closed"
    assert previous["sections"][0]["rows"][0]["hours"] == (
        "6:00 am - 10:00 pm"
    )


def test_fresh_wordpress_candidate_preserves_source_modified_gmt_exactly() -> None:
    source_modified_gmt = "2026-09-01T10:55:00"
    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(
            1186,
            source="wp_json",
            error_category=None,
            source_modified_gmt=source_modified_gmt,
        ),
        valid_previous_facility(1186),
        FIXED_NOW,
    )

    assert result["status"] == "ok"
    assert result["source"] == "wp_json"
    assert result["sourceModifiedGmt"] == source_modified_gmt


@pytest.mark.parametrize("category", sorted(facility_hours_fetch.SCHEDULE_ERROR_CATEGORIES))
def test_safe_error_message_is_fixed_for_every_allowed_category(category: str) -> None:
    assert facility_hours_fetch.safe_error_message(category) == (
        "Official hours could not be refreshed."
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_id",
        "wrong_name",
        "wrong_slug",
        "cross_origin_url",
        "wrong_facility_path",
        "query_url",
        "cross_origin_resolved",
        "error_status",
        "ok_marked_stale",
        "stale_marked_fresh",
        "unknown_source",
        "direct_modified_gmt",
        "future_source_timestamp",
        "naive_updated_timestamp",
        "empty_sections",
        "markup_title",
        "markup_label",
        "markup_hours",
        "markup_note",
        "unknown_section_field",
        "unknown_row_field",
        "unknown_top_level_field",
    ],
)
def test_failed_candidate_rejects_invalid_previous_records(mutation: str) -> None:
    previous = valid_previous_facility(1186)
    if mutation == "bool_id":
        previous["facilityId"] = True
    elif mutation == "wrong_name":
        previous["facilityName"] = "private-marker"
    elif mutation == "wrong_slug":
        previous["slug"] = "bakke"
    elif mutation == "cross_origin_url":
        previous["url"] = "https://recwell.example.test.attacker.invalid/nick/"
    elif mutation == "wrong_facility_path":
        previous["url"] = "https://recwell.example.test/locations/bakke/"
    elif mutation == "query_url":
        previous["url"] = "https://recwell.example.test/nick/?private=marker"
    elif mutation == "cross_origin_resolved":
        previous["resolvedUrl"] = "https://attacker.invalid/nick/"
    elif mutation == "error_status":
        previous["status"] = "error"
    elif mutation == "ok_marked_stale":
        previous["stale"] = True
    elif mutation == "stale_marked_fresh":
        previous.update(
            {
                "status": "stale",
                "stale": False,
                "error": "Official hours could not be refreshed.",
                "errorCategory": "anti_bot",
            }
        )
    elif mutation == "unknown_source":
        previous["source"] = "private_provider"
    elif mutation == "direct_modified_gmt":
        previous["sourceModifiedGmt"] = "2026-09-01T10:55:00"
    elif mutation == "future_source_timestamp":
        previous["sourceFetchedAt"] = "2026-09-01T12:00:01Z"
    elif mutation == "naive_updated_timestamp":
        previous["updatedAt"] = "2026-09-01T11:00:00"
    elif mutation == "empty_sections":
        previous["sections"] = []
    elif mutation == "markup_title":
        previous["sections"][0]["title"] = "<private-marker>"
    elif mutation == "markup_label":
        previous["sections"][0]["rows"][0]["label"] = "<private-marker>"
    elif mutation == "markup_hours":
        previous["sections"][0]["rows"][0]["hours"] = "<private-marker>"
    elif mutation == "markup_note":
        previous["sections"][0]["note"] = "<private-marker>"
    elif mutation == "unknown_section_field":
        previous["sections"][0]["privateHtml"] = "private-section-marker"
    elif mutation == "unknown_row_field":
        previous["sections"][0]["rows"][0]["privateUrl"] = (
            "https://attacker.invalid/"
        )
    elif mutation == "unknown_top_level_field":
        previous["privateDebug"] = "private-top-level-marker"

    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(1186, source=None, error_category="upstream_http"),
        previous,
        FIXED_NOW,
    )

    assert result["status"] == "error"
    assert result["sections"] == []
    assert result["source"] is None
    assert "private" not in str(result).lower()
    assert "attacker" not in str(result).lower()


def test_stale_wordpress_merge_preserves_valid_source_modified_gmt() -> None:
    previous = valid_previous_facility(1186, source="wp_json")
    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(1186, source=None, error_category="anti_bot"),
        previous,
        FIXED_NOW,
    )

    assert result["status"] == "stale"
    assert result["source"] == "wp_json"
    assert result["sourceModifiedGmt"] == "2026-09-01T10:55:00"


def test_explicit_default_https_port_is_same_origin_for_previous_record() -> None:
    previous = valid_previous_facility(1186)
    previous["url"] = "https://recwell.example.test:443/locations/nick/"
    previous["resolvedUrl"] = "https://recwell.example.test:443/nick/"

    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(1186, source=None, error_category="anti_bot"),
        previous,
        FIXED_NOW,
    )

    assert result["status"] == "stale"
    assert result["url"] == previous["url"]
    assert result["resolvedUrl"] == previous["resolvedUrl"]


def test_future_fresh_candidate_is_not_published_as_current() -> None:
    result = facility_hours_fetch.merge_facility_candidate(
        candidate_for(
            1186,
            source="direct_html",
            error_category=None,
            fetched_at=FIXED_NOW + timedelta(seconds=1),
        ),
        None,
        FIXED_NOW,
    )

    assert result["status"] == "error"
    assert result["errorCategory"] == "schema_invalid"
    assert result["sections"] == []


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "http://recwell.example.test/locations/nick/",
        "//recwell.example.test/locations/nick/",
        "https://user:password@recwell.example.test/locations/nick/",
        "https://recwell.example.test.attacker.invalid/locations/nick/",
        "https://recwell.example.test:444/locations/nick/",
        "https://recwell.example.test:/locations/nick/",
        "https://recwell.example.test:not-a-port/locations/nick/",
        "https://recwell.example.test/locations/nick/?private=1",
        "https://recwell.example.test/locations/nick/?",
        "https://recwell.example.test/locations/nick/#private",
        "https://recwell.example.test/locations/nick/#",
        "https://recwell.example.test/locations/\nnick/",
        "https://recwell.example.test/locations/%0anick/",
        "https://recwell.exämple.test/locations/nick/",
        "https://" + ".".join(["a" * 63] * 4) + "/locations/nick/",
        "https://recwell.example.test/" + ("x" * 2_100),
    ],
)
def test_same_origin_url_validation_rejects_unsafe_urls(unsafe_url: str) -> None:
    assert facility_hours_fetch.safe_same_origin_https_url(
        unsafe_url,
        "https://recwell.example.test",
    ) is None


@pytest.mark.parametrize(
    "safe_url",
    [
        "https://RECWELL.EXAMPLE.TEST/locations/nick/",
        "https://recwell.example.test:443/locations/nick/",
    ],
)
def test_same_origin_url_validation_accepts_canonical_https_origin(
    safe_url: str,
) -> None:
    assert facility_hours_fetch.safe_same_origin_https_url(
        safe_url,
        "https://recwell.example.test",
    ) == safe_url


def test_same_origin_url_validation_rejects_oversized_hostname() -> None:
    oversized_hostname = ".".join(["a" * 63] * 4)
    assert facility_hours_fetch.safe_same_origin_https_url(
        f"https://{oversized_hostname}/locations/nick/",
        f"https://{oversized_hostname}",
    ) is None


@pytest.mark.parametrize("field", ["title", "label", "hours", "note"])
def test_candidate_rejects_markup_in_every_schedule_text_field(field: str) -> None:
    section = schedule_section()
    if field == "title":
        section["title"] = "<private-marker>"
    elif field == "note":
        section["note"] = "<private-marker>"
    else:
        section["rows"][0][field] = "<private-marker>"
    config = facility_config(1186)

    with pytest.raises(ValueError) as captured:
        facility_hours_fetch.FacilityCandidate(
            facility_id=1186,
            facility_name="Nick",
            slug="nick",
            public_url=str(config["url"]),
            source="direct_html",
            resolved_url=str(config["url"]),
            source_modified_gmt=None,
            sections=(section,),
            fetched_at=FIXED_NOW,
            error_category=None,
        )

    assert str(captured.value) == "invalid facility candidate"
    assert "private-marker" not in str(captured.value)


def test_direct_collection_wins_without_calling_wordpress(
    monkeypatch: pytest.MonkeyPatch,
    fixture_text: Callable[[str], str],
) -> None:
    wordpress_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_direct_html",
        lambda _url: (
            fixture_text("facility_hours/nick-direct.html"),
            "https://recwell.example.test/locations/nick/",
        ),
    )

    def record_wordpress_call(*args: object, **_kwargs: object) -> object:
        wordpress_calls.append(args)
        raise facility_hours_fetch.ScheduleFetchError("upstream_http")

    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_wp_json_html",
        record_wordpress_call,
    )

    candidate = facility_hours_fetch.collect_facility_candidate(
        facility_config(1186),
        "https://recwell.example.test",
        FIXED_NOW,
    )

    assert candidate.ok is True
    assert candidate.source == "direct_html"
    assert candidate.source_modified_gmt is None
    assert candidate.resolved_url == (
        "https://recwell.example.test/locations/nick/"
    )
    assert wordpress_calls == []


def test_unsafe_direct_redirect_is_invalidated_before_wordpress_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fixture_text: Callable[[str], str],
) -> None:
    marker_url = "https://user:private-marker@attacker.invalid/nick/"
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_direct_html",
        lambda _url: (
            fixture_text("facility_hours/nick-direct.html"),
            marker_url,
        ),
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_wp_json_html",
        lambda site_base, slug: (
            fixture_text("facility_hours/nick-direct.html"),
            "2026-09-01T10:55:00",
            f"{site_base}/locations/{slug}/",
        ),
    )

    candidate = facility_hours_fetch.collect_facility_candidate(
        facility_config(1186),
        "https://recwell.example.test",
        FIXED_NOW,
    )

    assert candidate.ok is True
    assert candidate.source == "wp_json"
    assert candidate.source_modified_gmt == "2026-09-01T10:55:00"
    assert candidate.resolved_url == (
        "https://recwell.example.test/locations/nick/"
    )
    assert "private-marker" not in repr(candidate)
    assert "attacker.invalid" not in repr(candidate)


def test_unsafe_wordpress_link_is_dropped_while_valid_schedule_is_kept(
    monkeypatch: pytest.MonkeyPatch,
    fixture_text: Callable[[str], str],
) -> None:
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_direct_html",
        lambda _url: (_ for _ in ()).throw(
            facility_hours_fetch.ScheduleFetchError("upstream_timeout")
        ),
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_wp_json_html",
        lambda site_base, slug: (
            fixture_text("facility_hours/nick-direct.html"),
            "2026-09-01T10:55:00",
            "https://recwell.example.test.attacker.invalid/private-marker",
        ),
    )

    candidate = facility_hours_fetch.collect_facility_candidate(
        facility_config(1186),
        "https://recwell.example.test",
        FIXED_NOW,
    )

    assert candidate.ok is True
    assert candidate.source == "wp_json"
    assert candidate.resolved_url is None
    assert "private-marker" not in repr(candidate)
    assert "attacker.invalid" not in repr(candidate)


def test_collection_sanitizes_unexpected_failures_and_returns_final_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "private-unexpected-provider-marker"

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(marker)

    monkeypatch.setattr(facility_hours_fetch, "fetch_direct_html", unexpected)
    monkeypatch.setattr(facility_hours_fetch, "fetch_wp_json_html", unexpected)

    candidate = facility_hours_fetch.collect_facility_candidate(
        facility_config(1186),
        "https://recwell.example.test",
        FIXED_NOW,
    )

    assert candidate.ok is False
    assert candidate.error_category == "schema_invalid"
    assert candidate.source is None
    assert candidate.sections == ()
    assert marker not in repr(candidate)


def test_collection_rejects_invalid_configuration_before_requesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "private-config-marker"
    request_calls: list[object] = []

    def record_request(value: object) -> object:
        request_calls.append(value)
        raise AssertionError("request seam must not be reached")

    monkeypatch.setattr(facility_hours_fetch, "fetch_direct_html", record_request)
    invalid = facility_config(1186)
    invalid["url"] = f"https://user:{marker}@recwell.example.test/nick/"

    with pytest.raises(ValueError) as captured:
        facility_hours_fetch.collect_facility_candidate(
            invalid,
            "https://recwell.example.test",
            FIXED_NOW,
        )

    assert str(captured.value) == "invalid facility configuration"
    assert marker not in str(captured.value)
    assert request_calls == []


def test_schedule_payload_validation_returns_a_normalized_deep_copy() -> None:
    payload = valid_schedule_payload()

    normalized = facility_hours_fetch.validate_schedule_payload(
        payload,
        now=FIXED_NOW,
    )

    assert normalized == payload
    assert normalized is not payload
    assert normalized["facilities"] is not payload["facilities"]
    assert normalized["facilities"][0] is not payload["facilities"][0]
    assert normalized["facilities"][0]["sections"] is not (
        payload["facilities"][0]["sections"]
    )
    normalized["facilities"][0]["sections"][0]["rows"][0]["hours"] = (
        "Closed"
    )
    assert payload["facilities"][0]["sections"][0]["rows"][0]["hours"] == (
        "6:00 am - 10:00 pm"
    )


@pytest.mark.parametrize("status", ["stale", "error"])
def test_schedule_payload_validation_accepts_truthful_nonfresh_states(
    status: str,
) -> None:
    payload = valid_schedule_payload()
    record = payload["facilities"][0]
    if status == "stale":
        mark_facility_stale(record)
    else:
        mark_facility_error(record)
    payload["okCount"] = 1

    normalized = facility_hours_fetch.validate_schedule_payload(
        payload,
        now=FIXED_NOW,
    )

    assert normalized["facilities"][0]["status"] == status
    assert normalized["okCount"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_top",
        "missing_total",
        "generated_offset",
        "generated_future",
        "source_http",
        "source_query",
        "source_path",
        "missing_facility",
        "duplicate_facility",
        "reversed_facilities",
        "bool_facility_id",
        "wrong_name",
        "wrong_slug",
        "wrong_public_path",
        "unsafe_resolved",
        "bool_ok_count",
        "wrong_ok_count",
        "bool_total_count",
        "wrong_total_count",
        "unknown_facility_field",
        "unknown_section_field",
        "tuple_rows",
        "ok_marked_stale",
        "ok_with_error",
        "stale_marked_fresh",
        "stale_without_error",
        "error_with_source",
        "error_with_sections",
        "error_with_timestamp",
        "direct_modified_gmt",
        "wp_bad_modified_gmt",
        "future_updated",
        "future_source_fetched",
    ],
)
def test_schedule_payload_validation_rejects_every_untrusted_dimension(
    mutation: str,
) -> None:
    payload = valid_schedule_payload()
    facilities = payload["facilities"]
    nick = facilities[0]
    if mutation == "unknown_top":
        payload["privateMarker"] = "private-marker"
    elif mutation == "missing_total":
        payload.pop("totalCount")
    elif mutation == "generated_offset":
        payload["generatedAt"] = "2026-09-01T07:00:00-05:00"
    elif mutation == "generated_future":
        payload["generatedAt"] = "2026-09-01T12:00:01Z"
    elif mutation == "source_http":
        payload["sourceSite"] = "http://recwell.example.test"
    elif mutation == "source_query":
        payload["sourceSite"] = "https://recwell.example.test?private=marker"
    elif mutation == "source_path":
        payload["sourceSite"] = "https://recwell.example.test/private-marker"
    elif mutation == "missing_facility":
        payload["facilities"] = facilities[:1]
    elif mutation == "duplicate_facility":
        payload["facilities"] = [facilities[0], facilities[0]]
    elif mutation == "reversed_facilities":
        payload["facilities"] = list(reversed(facilities))
    elif mutation == "bool_facility_id":
        nick["facilityId"] = True
    elif mutation == "wrong_name":
        nick["facilityName"] = "private-marker"
    elif mutation == "wrong_slug":
        nick["slug"] = "bakke"
    elif mutation == "wrong_public_path":
        nick["url"] = "https://recwell.example.test/locations/bakke/"
    elif mutation == "unsafe_resolved":
        nick["resolvedUrl"] = "https://attacker.invalid/private-marker"
    elif mutation == "bool_ok_count":
        payload["okCount"] = True
    elif mutation == "wrong_ok_count":
        payload["okCount"] = 1
    elif mutation == "bool_total_count":
        payload["totalCount"] = True
    elif mutation == "wrong_total_count":
        payload["totalCount"] = 3
    elif mutation == "unknown_facility_field":
        nick["privateMarker"] = "private-marker"
    elif mutation == "unknown_section_field":
        nick["sections"][0]["privateMarker"] = "private-marker"
    elif mutation == "tuple_rows":
        nick["sections"][0]["rows"] = tuple(nick["sections"][0]["rows"])
    elif mutation == "ok_marked_stale":
        nick["stale"] = True
    elif mutation == "ok_with_error":
        nick["error"] = "private-marker"
    elif mutation == "stale_marked_fresh":
        mark_facility_stale(nick)
        nick["stale"] = False
        payload["okCount"] = 1
    elif mutation == "stale_without_error":
        mark_facility_stale(nick)
        nick["error"] = None
        payload["okCount"] = 1
    elif mutation == "error_with_source":
        mark_facility_error(nick)
        nick["source"] = "direct_html"
        payload["okCount"] = 1
    elif mutation == "error_with_sections":
        original_sections = nick["sections"]
        mark_facility_error(nick)
        nick["sections"] = original_sections
        payload["okCount"] = 1
    elif mutation == "error_with_timestamp":
        mark_facility_error(nick)
        nick["sourceFetchedAt"] = "2026-09-01T11:00:00Z"
        payload["okCount"] = 1
    elif mutation == "direct_modified_gmt":
        nick["sourceModifiedGmt"] = "2026-09-01T10:55:00"
    elif mutation == "wp_bad_modified_gmt":
        nick["source"] = "wp_json"
        nick["sourceModifiedGmt"] = "private-marker"
    elif mutation == "future_updated":
        nick["updatedAt"] = "2026-09-01T12:00:01Z"
    elif mutation == "future_source_fetched":
        nick["sourceFetchedAt"] = "2026-09-01T12:00:01Z"

    with pytest.raises(ValueError) as captured:
        facility_hours_fetch.validate_schedule_payload(
            payload,
            now=FIXED_NOW,
        )

    rendered = "".join(traceback.format_exception(captured.value))
    assert str(captured.value) == "invalid schedule payload"
    assert "private-marker" not in rendered
    assert "attacker.invalid" not in rendered


def test_build_combined_payload_updates_facilities_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = valid_schedule_payload()

    def collect(
        facility: dict[str, object],
        site_base: str,
        fetched_at: datetime,
    ) -> facility_hours_fetch.FacilityCandidate:
        assert site_base == "https://recwell.example.test"
        if facility["facilityId"] == 1186:
            return candidate_for(
                1186,
                source=None,
                error_category="anti_bot",
                fetched_at=fetched_at,
            )
        return candidate_for(
            1656,
            source="direct_html",
            error_category=None,
            title="Bakke fresh rows",
            fetched_at=fetched_at,
        )

    monkeypatch.setattr(facility_hours_fetch, "collect_facility_candidate", collect)

    result = facility_hours_fetch.build_combined_payload(
        [facility_config(1186), facility_config(1656)],
        previous,
        "https://recwell.example.test",
        FIXED_NOW,
    )

    nick, bakke = result["facilities"]
    assert (nick["status"], nick["stale"], nick["sections"]) == (
        "stale",
        True,
        previous["facilities"][0]["sections"],
    )
    assert (bakke["status"], bakke["stale"]) == ("ok", False)
    assert bakke["sections"][0]["title"] == "Bakke fresh rows"
    assert (result["okCount"], result["totalCount"]) == (1, 2)


def test_one_invalid_previous_row_discards_the_entire_previous_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = valid_schedule_payload()
    previous["facilities"][1]["privateMarker"] = "private-marker"
    monkeypatch.setattr(
        facility_hours_fetch,
        "collect_facility_candidate",
        lambda facility, site_base, fetched_at: candidate_for(
            facility["facilityId"],
            source=None,
            error_category="upstream_http",
            fetched_at=fetched_at,
        ),
    )

    result = facility_hours_fetch.build_combined_payload(
        [facility_config(1186), facility_config(1656)],
        previous,
        "https://recwell.example.test",
        FIXED_NOW,
    )

    assert [row["status"] for row in result["facilities"]] == ["error", "error"]
    assert [row["sections"] for row in result["facilities"]] == [[], []]
    assert "private-marker" not in str(result)


def test_one_previous_row_mismatching_current_config_discards_all_previous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = valid_schedule_payload()
    current = [facility_config(1186), facility_config(1656)]
    current[0]["url"] = "https://recwell.example.test/alternate/nick/"

    def failed_candidate(
        facility: dict[str, object],
        _site_base: str,
        fetched_at: datetime,
    ) -> facility_hours_fetch.FacilityCandidate:
        facility_id = facility["facilityId"]
        identity = {1186: ("Nick", "nick"), 1656: ("Bakke", "bakke")}
        facility_name, slug = identity[facility_id]
        return facility_hours_fetch.FacilityCandidate(
            facility_id=facility_id,
            facility_name=facility_name,
            slug=slug,
            public_url=str(facility["url"]),
            source=None,
            resolved_url=None,
            source_modified_gmt=None,
            sections=(),
            fetched_at=fetched_at,
            error_category="upstream_http",
        )

    monkeypatch.setattr(
        facility_hours_fetch,
        "collect_facility_candidate",
        failed_candidate,
    )

    result = facility_hours_fetch.build_combined_payload(
        current,
        previous,
        "https://recwell.example.test",
        FIXED_NOW,
    )

    assert [row["status"] for row in result["facilities"]] == ["error", "error"]
    assert [row["sections"] for row in result["facilities"]] == [[], []]


def test_build_combined_payload_validates_its_own_merged_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        facility_hours_fetch,
        "collect_facility_candidate",
        lambda facility, site_base, fetched_at: candidate_for(
            facility["facilityId"],
            source="direct_html",
            error_category=None,
            fetched_at=fetched_at,
        ),
    )
    original_merge = facility_hours_fetch.merge_facility_candidate

    def malformed_merge(*args: object, **kwargs: object) -> dict[str, object]:
        record = original_merge(*args, **kwargs)
        record["privateMarker"] = "private-marker"
        return record

    monkeypatch.setattr(
        facility_hours_fetch,
        "merge_facility_candidate",
        malformed_merge,
    )

    with pytest.raises(ValueError, match="invalid schedule payload"):
        facility_hours_fetch.build_combined_payload(
            [facility_config(1186), facility_config(1656)],
            None,
            "https://recwell.example.test",
            FIXED_NOW,
        )


def test_invalid_payload_is_rejected_before_creating_output_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "missing" / "facility_hours.json"

    with pytest.raises(ValueError, match="invalid schedule payload"):
        facility_hours_fetch.atomic_write_json(
            str(target),
            {"generatedAt": "2026-09-01T12:00:00Z"},
        )

    assert not target.parent.exists()


def test_atomic_write_replaces_with_validated_fsynced_same_directory_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "facility_hours.json"
    target.write_bytes(b"old artifact\n")
    original_fsync = facility_hours_fetch.os.fsync
    original_replace = facility_hours_fetch.os.replace
    fsync_calls: list[int] = []

    def fsync_file(fd: int) -> None:
        fsync_calls.append(fd)
        original_fsync(fd)

    def replace_same_directory(source: str, destination: str) -> None:
        assert Path(source).parent == target.parent
        assert Path(destination) == target
        original_replace(source, destination)

    monkeypatch.setattr(facility_hours_fetch.os, "fsync", fsync_file)
    monkeypatch.setattr(
        facility_hours_fetch.os,
        "replace",
        replace_same_directory,
    )

    facility_hours_fetch.atomic_write_json(
        str(target),
        valid_schedule_payload(),
    )

    assert json.loads(target.read_text(encoding="utf-8")) == (
        valid_schedule_payload()
    )
    assert fsync_calls
    assert list(tmp_path.glob(".facility-hours-*.json")) == []


@pytest.mark.parametrize(
    "failure_point",
    ["serialize", "write", "flush", "fsync", "close", "replace"],
)
def test_atomic_write_failure_preserves_target_and_cleans_temporary_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str,
) -> None:
    marker = "private-atomic-marker"
    target = tmp_path / "facility_hours.json"
    original_bytes = b"old artifact stays byte exact\n"
    target.write_bytes(original_bytes)
    original_fdopen = facility_hours_fetch.os.fdopen

    class FailingHandle:
        def __init__(self, handle: object) -> None:
            self.handle = handle

        def __enter__(self) -> "FailingHandle":
            return self

        def __exit__(self, *_args: object) -> None:
            self.handle.close()
            if failure_point == "close":
                raise OSError(marker)

        def write(self, value: str) -> int:
            if failure_point == "write":
                raise OSError(marker)
            return self.handle.write(value)

        def flush(self) -> None:
            if failure_point == "flush":
                raise OSError(marker)
            self.handle.flush()

        def fileno(self) -> int:
            return self.handle.fileno()

    if failure_point == "serialize":
        monkeypatch.setattr(
            facility_hours_fetch.json,
            "dumps",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError(marker)),
        )
    elif failure_point in {"write", "flush", "close"}:
        monkeypatch.setattr(
            facility_hours_fetch.os,
            "fdopen",
            lambda fd, *args, **kwargs: FailingHandle(
                original_fdopen(fd, *args, **kwargs)
            ),
        )
    elif failure_point == "fsync":
        monkeypatch.setattr(
            facility_hours_fetch.os,
            "fsync",
            lambda _fd: (_ for _ in ()).throw(OSError(marker)),
        )
    elif failure_point == "replace":
        monkeypatch.setattr(
            facility_hours_fetch.os,
            "replace",
            lambda _source, _target: (_ for _ in ()).throw(OSError(marker)),
        )

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.atomic_write_json(
            str(target),
            valid_schedule_payload(),
        )

    rendered = "".join(traceback.format_exception(captured.value))
    assert captured.value.category == "io_error"
    assert captured.value.__cause__ is None
    assert marker not in rendered
    assert target.read_bytes() == original_bytes
    assert list(tmp_path.glob(".facility-hours-*.json")) == []


def test_atomic_write_cleans_temporary_file_on_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "facility_hours.json"
    original_bytes = b"old artifact\n"
    target.write_bytes(original_bytes)
    monkeypatch.setattr(
        facility_hours_fetch.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(KeyboardInterrupt()),
    )

    with pytest.raises(KeyboardInterrupt):
        facility_hours_fetch.atomic_write_json(
            str(target),
            valid_schedule_payload(),
        )

    assert target.read_bytes() == original_bytes
    assert list(tmp_path.glob(".facility-hours-*.json")) == []


def test_write_json_compatibility_symbol_uses_validated_atomic_publication(
    tmp_path: Path,
) -> None:
    target = tmp_path / "facility_hours.json"
    target.write_bytes(b"old artifact\n")

    with pytest.raises(ValueError, match="invalid schedule payload"):
        facility_hours_fetch.write_json(
            str(target),
            {"generatedAt": "2026-09-01T12:00:00Z"},
        )

    assert target.read_bytes() == b"old artifact\n"


def test_previous_schedule_load_is_all_or_nothing_and_sanitized(
    tmp_path: Path,
) -> None:
    target = tmp_path / "facility_hours.json"
    target.write_text(json.dumps(valid_schedule_payload()), encoding="utf-8")
    loaded = facility_hours_fetch.load_previous_schedule(
        str(target),
        now=FIXED_NOW,
    )
    assert loaded == valid_schedule_payload()
    assert loaded is not valid_schedule_payload()

    invalid = valid_schedule_payload()
    invalid["facilities"][1]["privateMarker"] = "private-marker"
    target.write_text(json.dumps(invalid), encoding="utf-8")
    assert facility_hours_fetch.load_previous_schedule(
        str(target),
        now=FIXED_NOW,
    ) is None

    target.write_text("{private-json-marker", encoding="utf-8")
    assert facility_hours_fetch.load_previous_schedule(
        str(target),
        now=FIXED_NOW,
    ) is None
    assert facility_hours_fetch.load_previous_schedule(
        str(tmp_path / "missing.json"),
        now=FIXED_NOW,
    ) is None

    duplicate = json.dumps(valid_schedule_payload())
    duplicate = duplicate[:-1] + ', "okCount": 2}'
    target.write_text(duplicate, encoding="utf-8")
    assert facility_hours_fetch.load_previous_schedule(
        str(target),
        now=FIXED_NOW,
    ) is None


def test_atomic_failure_does_not_create_an_absent_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "facility_hours.json"
    monkeypatch.setattr(
        facility_hours_fetch.os,
        "replace",
        lambda _source, _target: (_ for _ in ()).throw(OSError("private-marker")),
    )

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.atomic_write_json(
            str(target),
            valid_schedule_payload(),
        )

    assert captured.value.category == "io_error"
    assert not target.exists()
    assert list(tmp_path.glob(".facility-hours-*.json")) == []


def test_main_validates_environment_before_loading_or_collecting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        facility_hours_fetch,
        "load_project_dotenv",
        lambda: events.append("dotenv"),
    )

    def reject_environment(*_args: object, **_kwargs: object) -> None:
        events.append("validate")
        raise RuntimeError("environment rejected")

    monkeypatch.setattr(
        facility_hours_fetch,
        "validate_production_environment",
        reject_environment,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "load_previous_schedule",
        lambda *_args, **_kwargs: events.append("load"),
        raising=False,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "collect_facility_candidate",
        lambda *_args, **_kwargs: events.append("collect"),
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "write_json",
        lambda *_args, **_kwargs: events.append("write"),
    )

    with pytest.raises(RuntimeError, match="environment rejected"):
        facility_hours_fetch.main()

    assert events == ["dotenv", "validate"]


def test_fetch_cli_publishes_valid_two_facility_payload_and_preserves_phase_five_predicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixed_now: datetime,
    fixture_text: Callable[[str], str],
) -> None:
    output_path = tmp_path / "facility_hours.json"

    def fixture_backed_direct_fetch(url: str) -> tuple[str, str]:
        fixture = (
            "facility_hours/nick-direct.html"
            if "/nick" in url
            else "facility_hours/bakke-direct.html"
        )
        return fixture_text(fixture), url

    monkeypatch.setattr(facility_hours_fetch, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_direct_html",
        fixture_backed_direct_fetch,
    )

    assert facility_hours_fetch.main_for_output(str(output_path)) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [row["facilityId"] for row in payload["facilities"]] == [1186, 1656]
    assert all(
        row["status"] == "ok" and row["stale"] is False
        for row in payload["facilities"]
    )
    assert facility_schedule.official_facility_is_open(
        payload,
        1186,
        chicago_datetime(2026, 8, 31, 12, 0),
    ) is True


def test_main_failure_output_contains_only_safe_category(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    marker = "private-cli-provider-marker"
    target = tmp_path / "private-output-marker.json"
    monkeypatch.setattr(sys, "argv", ["facility_hours_fetch.py", "--output", str(target)])
    monkeypatch.setattr(facility_hours_fetch, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(
        facility_hours_fetch,
        "validate_production_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "load_previous_schedule",
        lambda *_args, **_kwargs: None,
        raising=False,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "build_combined_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "collect_facility_hours",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(marker)),
    )

    assert facility_hours_fetch.main() == 1
    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert "schema_invalid" in rendered
    assert marker not in rendered
    assert str(target) not in rendered


def test_main_publishes_only_sanitized_status_counts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    target = tmp_path / "private-output-marker.json"
    payload = valid_schedule_payload()
    mark_facility_stale(payload["facilities"][0])
    payload["okCount"] = 1
    monkeypatch.setattr(sys, "argv", ["facility_hours_fetch.py", "--output", str(target)])
    monkeypatch.setattr(facility_hours_fetch, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(
        facility_hours_fetch,
        "validate_production_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "build_combined_payload",
        lambda *_args, **_kwargs: payload,
    )

    assert facility_hours_fetch.main() == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        "facility_hours_fetch: published ok=1 stale=1 error=0 total=2\n"
    )
    assert json.loads(target.read_text(encoding="utf-8")) == payload


def test_main_atomic_failure_reports_only_io_error_category(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    marker = "private-write-marker"
    target = tmp_path / "private-output-marker.json"
    monkeypatch.setattr(sys, "argv", ["facility_hours_fetch.py", "--output", str(target)])
    monkeypatch.setattr(facility_hours_fetch, "load_project_dotenv", lambda: None)
    monkeypatch.setattr(
        facility_hours_fetch,
        "validate_production_environment",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "load_previous_schedule",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        facility_hours_fetch,
        "build_combined_payload",
        lambda *_args, **_kwargs: valid_schedule_payload(),
    )

    def write_failure(*_args: object, **_kwargs: object) -> None:
        try:
            raise OSError(marker)
        except OSError:
            raise facility_hours_fetch.ScheduleFetchError("io_error") from None

    monkeypatch.setattr(facility_hours_fetch, "atomic_write_json", write_failure)

    assert facility_hours_fetch.main() == 1
    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert rendered == "facility_hours_fetch: failed category=io_error\n"
    assert marker not in rendered
    assert str(target) not in rendered


def test_parse_utc_timestamp_accepts_only_aware_iso_values() -> None:
    parsed = facility_schedule.parse_utc_timestamp("2026-08-31T07:00:00-05:00")
    assert parsed == datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    assert parsed is not None
    assert parsed.tzinfo is timezone.utc
    assert facility_schedule.parse_utc_timestamp("2026-08-31T12:00:00Z") == (
        datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
    )
    assert facility_schedule.parse_utc_timestamp("") is None
    assert facility_schedule.parse_utc_timestamp("2026-08-31") is None
    assert facility_schedule.parse_utc_timestamp("2026-08-31T12:00:00") is None
    assert facility_schedule.parse_utc_timestamp("not-a-timestamp") is None
    assert facility_schedule.parse_utc_timestamp(None) is None


def test_iso_utc_normalizes_aware_values_and_rejects_naive_values() -> None:
    west = timezone(-timedelta(hours=5))
    assert facility_hours_fetch.iso_utc(
        datetime(2026, 8, 31, 7, 0, tzinfo=west)
    ) == "2026-08-31T12:00:00Z"
    with pytest.raises(ValueError, match="timezone-aware"):
        facility_hours_fetch.iso_utc(datetime(2026, 8, 31, 12, 0))


def test_forecast_job_reexports_the_shared_schedule_grammar() -> None:
    import forecast_job

    assert forecast_job.parse_schedule_date_range is (
        facility_schedule.parse_schedule_date_range
    )
    assert forecast_job.parse_schedule_weekday_set is (
        facility_schedule.parse_schedule_weekday_set
    )
    assert forecast_job.parse_schedule_hours_window is (
        facility_schedule.parse_schedule_hours_window
    )


def test_forecast_cached_open_state_contract_remains_compatible() -> None:
    import forecast_job

    at = datetime(2026, 8, 31, 12, 0, tzinfo=ZoneInfo("America/Chicago"))
    for hours, expected in (
        ("6:00 am - 10:00 pm", True),
        ("Closed", False),
    ):
        sections = [
            {
                "title": "Building Hours",
                "rows": [{"label": "Mon-Fri", "hours": hours}],
            }
        ]

        assert facility_schedule.get_facility_schedule_open_state(
            sections,
            at,
        ) is expected
        assert forecast_job.get_facility_schedule_open_state(
            sections=sections,
            ts=at,
            date_range_cache={},
            weekday_cache={},
            hours_window_cache={},
        ) is expected

    open_sections = [
        {
            "title": "Building Hours",
            "rows": [
                {"label": "Mon-Fri", "hours": "6:00 am - 10:00 pm"}
            ],
        }
    ]
    assert forecast_job.get_facility_schedule_boundary_state(
        sections=open_sections,
        ts=at.replace(hour=6),
        date_range_cache={},
        weekday_cache={},
        hours_window_cache={},
    ) == (True, False)
    assert forecast_job.get_facility_schedule_window_for_timestamp(
        sections=open_sections,
        ts=at,
        date_range_cache={},
        weekday_cache={},
        hours_window_cache={},
    ) == (360, 1320, False)


def test_runtime_requirements_pin_the_optional_html_parser_once() -> None:
    pins = [
        line.strip()
        for line in (ROOT / "server" / "requirements.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip().lower().startswith(("beautifulsoup4", "bs4"))
    ]
    assert pins == ["beautifulsoup4==4.12.3"]


@pytest.mark.parametrize(
    ("fixture", "expected_label", "expected_hours", "expected_notice"),
    [
        (
            "facility_hours/nick-direct.html",
            "Aug 31 - Sep 4",
            "6:00 am - 10:00 pm",
            "Maintenance closure: the building closes at 6:00 pm on Sep 6.",
        ),
        (
            "facility_hours/bakke-direct.html",
            "Weekdays",
            "5:30 am - 11:00 pm",
            "No scheduled maintenance closures at this time.",
        ),
        (
            "facility_hours/structure-changed.html",
            "Saturday",
            "8:00 am - 8:00 pm",
            None,
        ),
    ],
)
def test_direct_html_fixtures_preserve_schedule_rows_and_bounded_notices(
    fixture_text: Callable[[str], str],
    fixture: str,
    expected_label: str,
    expected_hours: str,
    expected_notice: str | None,
) -> None:
    sections = facility_hours_fetch.parse_schedule_sections(fixture_text(fixture))
    rows = [row for section in sections for row in section["rows"]]
    notes = [section.get("note") for section in sections if section.get("note")]

    assert {"label": expected_label, "hours": expected_hours} in rows
    if expected_notice is not None:
        assert expected_notice in notes


@pytest.mark.parametrize(
    (
        "fixture",
        "expected_label",
        "expected_hours",
        "expected_url",
        "expected_modified_gmt",
    ),
    [
        (
            "facility_hours/nick-wp.json",
            "Aug 31 - Sep 4",
            "6:00 am - 10:00 pm",
            "https://recwell.example.test/locations/nick/",
            "2026-08-31T12:00:00",
        ),
        (
            "facility_hours/bakke-wp.json",
            "Weekdays",
            "5:30 am - 11:00 pm",
            "https://recwell.example.test/locations/bakke/",
            "2026-08-31T12:05:00",
        ),
    ],
)
def test_wordpress_fixture_preserves_source_modified_time_and_schedule_html(
    fixture_json: Callable[[str], object],
    fixture: str,
    expected_label: str,
    expected_hours: str,
    expected_url: str,
    expected_modified_gmt: str,
) -> None:
    html, source_modified_gmt, resolved_url = (
        facility_hours_fetch.parse_wp_page_payload(fixture_json(fixture))
    )
    rows = [
        row
        for section in facility_hours_fetch.parse_schedule_sections(html)
        for row in section["rows"]
    ]

    assert {"label": expected_label, "hours": expected_hours} in rows
    assert source_modified_gmt == expected_modified_gmt
    assert resolved_url == expected_url


def test_antibot_fixture_is_rejected_with_only_the_safe_category(
    fixture_text: Callable[[str], str],
) -> None:
    html = fixture_text("facility_hours/anti-bot.html")

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.parse_direct_response(
            html,
            "https://recwell.example.test/locations/nick/",
        )

    assert captured.value.category == "anti_bot"
    assert str(captured.value) == "anti_bot"
    assert "Checking your browser" not in str(captured.value)


def test_valid_html_without_a_schedule_uses_parse_empty_not_raw_content() -> None:
    marker = "private-upstream-body-marker"

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.parse_direct_response(
            f"<html><body><p>{marker}</p></body></html>",
            "https://recwell.example.test/locations/nick/",
        )

    assert captured.value.category == "parse_empty"
    assert str(captured.value) == "parse_empty"
    assert marker not in str(captured.value)


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        [],
        [None],
        [{}],
        [{"content": None}],
        [{"content": {"rendered": ""}}],
        [{"content": {"rendered": 123}}],
        [{"content": {"rendered": "<p>Hours</p>"}, "modified_gmt": {}}],
        [{"content": {"rendered": "<p>Hours</p>"}, "link": 123}],
    ],
)
def test_malformed_wordpress_payloads_use_only_wp_payload_invalid(
    payload: object,
) -> None:
    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.parse_wp_page_payload(payload)

    assert captured.value.category == "wp_payload_invalid"
    assert str(captured.value) == "wp_payload_invalid"


def test_wordpress_source_modified_string_is_preserved_byte_for_byte() -> None:
    source_modified_gmt = " 2026-08-31T12:00:00 "
    _html, returned_modified_gmt, _link = (
        facility_hours_fetch.parse_wp_page_payload(
            [
                {
                    "content": {
                        "rendered": (
                            "<h2>Building Hours</h2><table>"
                            "<tr><td>Weekdays</td><td>6:00 am - 10:00 pm</td></tr>"
                            "</table>"
                        )
                    },
                    "modified_gmt": source_modified_gmt,
                }
            ]
        )
    )

    assert returned_modified_gmt == source_modified_gmt


def test_wordpress_optional_metadata_can_be_absent() -> None:
    html = (
        "<h2>Building Hours</h2><table>"
        "<tr><td>Weekdays</td><td>6:00 am - 10:00 pm</td></tr>"
        "</table>"
    )

    assert facility_hours_fetch.parse_wp_page_payload(
        [{"content": {"rendered": html}}]
    ) == (html, None, None)


def test_schedule_fetch_error_rejects_arbitrary_categories_without_echoing_them() -> None:
    marker = "private-arbitrary-error-marker"

    with pytest.raises(ValueError) as captured:
        facility_hours_fetch.ScheduleFetchError(marker)

    rendered = "".join(traceback.format_exception(captured.value))
    assert marker not in str(captured.value)
    assert marker not in captured.value.args
    assert marker not in rendered


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        (facility_hours_fetch.requests.Timeout("private-timeout"), "upstream_timeout"),
        (facility_hours_fetch.requests.HTTPError("private-http"), "upstream_http"),
    ],
)
def test_direct_fetch_classifies_request_failures_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_category: str,
) -> None:
    def fail_request(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(facility_hours_fetch.requests, "get", fail_request)

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.fetch_direct_html(
            "https://recwell.example.test/locations/nick/"
        )

    assert captured.value.category == expected_category
    assert str(captured.value) == expected_category
    assert captured.value.__cause__ is None
    assert "private-" not in "".join(
        traceback.format_exception(captured.value)
    )
    assert "private-" not in str(captured.value)


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        (facility_hours_fetch.requests.Timeout("private-timeout"), "upstream_timeout"),
        (facility_hours_fetch.requests.HTTPError("private-http"), "upstream_http"),
    ],
)
def test_wordpress_fetch_classifies_request_failures_without_exception_text(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    expected_category: str,
) -> None:
    def fail_request(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(facility_hours_fetch.requests, "get", fail_request)

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.fetch_wp_json_html(
            "https://recwell.example.test",
            "nick",
        )

    assert captured.value.category == expected_category
    assert str(captured.value) == expected_category
    assert captured.value.__cause__ is None
    assert "private-" not in "".join(
        traceback.format_exception(captured.value)
    )
    assert "private-" not in str(captured.value)


def test_wordpress_json_decode_failure_uses_wp_payload_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidJsonResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            raise facility_hours_fetch.requests.JSONDecodeError(
                "private-json-body-marker",
                "private-json-body-marker",
                0,
            )

    monkeypatch.setattr(
        facility_hours_fetch.requests,
        "get",
        lambda *_args, **_kwargs: InvalidJsonResponse(),
    )

    with pytest.raises(facility_hours_fetch.ScheduleFetchError) as captured:
        facility_hours_fetch.fetch_wp_json_html(
            "https://recwell.example.test",
            "nick",
        )

    assert captured.value.category == "wp_payload_invalid"
    assert str(captured.value) == "wp_payload_invalid"
    assert captured.value.__cause__ is None
    assert "private-json-body-marker" not in "".join(
        traceback.format_exception(captured.value)
    )


def test_collection_classifies_valid_wordpress_html_without_schedule_as_parse_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "private-valid-html-without-schedule"

    def direct_failure(_url: str) -> tuple[str, str]:
        raise facility_hours_fetch.ScheduleFetchError("upstream_http")

    monkeypatch.setattr(facility_hours_fetch, "fetch_direct_html", direct_failure)
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_wp_json_html",
        lambda site_base, slug: (
            f"<p>{marker}</p>",
            "2026-08-31T12:00:00",
            f"{site_base}/locations/{slug}/",
        ),
    )

    result = facility_hours_fetch.collect_facility_hours(
        {
            "facilityId": 1186,
            "facilityName": "Nick",
            "slug": "nick",
            "url": "https://recwell.example.test/locations/nick/",
        },
        "https://recwell.example.test",
    )

    assert result["status"] == "error"
    assert result["error"] == (
        "direct_html: upstream_http; wp_json: parse_empty"
    )
    assert marker not in str(result["error"])


def test_collection_maps_unexpected_parser_failures_to_schema_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "private-unexpected-parser-marker"

    def unexpected(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError(marker)

    monkeypatch.setattr(facility_hours_fetch, "fetch_direct_html", unexpected)
    monkeypatch.setattr(facility_hours_fetch, "fetch_wp_json_html", unexpected)

    result = facility_hours_fetch.collect_facility_hours(
        {
            "facilityId": 1186,
            "facilityName": "Nick",
            "slug": "nick",
            "url": "https://recwell.example.test/locations/nick/",
        },
        "https://recwell.example.test",
    )

    assert result["error"] == (
        "direct_html: schema_invalid; wp_json: schema_invalid"
    )
    assert marker not in str(result["error"])


def test_collection_classifies_direct_challenge_without_persisting_html(
    monkeypatch: pytest.MonkeyPatch,
    fixture_text: Callable[[str], str],
) -> None:
    challenge_html = fixture_text("facility_hours/anti-bot.html")

    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_direct_html",
        lambda _url: (
            challenge_html,
            "https://recwell.example.test/locations/nick/",
        ),
    )

    def wordpress_failure(*_args: object, **_kwargs: object) -> object:
        raise facility_hours_fetch.ScheduleFetchError("wp_payload_invalid")

    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_wp_json_html",
        wordpress_failure,
    )

    result = facility_hours_fetch.collect_facility_hours(
        {
            "facilityId": 1186,
            "facilityName": "Nick",
            "slug": "nick",
            "url": "https://recwell.example.test/locations/nick/",
        },
        "https://recwell.example.test",
    )

    assert result["error"] == (
        "direct_html: anti_bot; wp_json: wp_payload_invalid"
    )
    assert "Checking your browser" not in str(result)
    assert "challenge-platform" not in str(result)


def test_wordpress_collection_preserves_source_modified_gmt(
    monkeypatch: pytest.MonkeyPatch,
    fixture_json: Callable[[str], object],
) -> None:
    wp_result = facility_hours_fetch.parse_wp_page_payload(
        fixture_json("facility_hours/nick-wp.json")
    )

    def direct_failure(_url: str) -> tuple[str, str]:
        raise facility_hours_fetch.ScheduleFetchError("upstream_http")

    monkeypatch.setattr(facility_hours_fetch, "fetch_direct_html", direct_failure)
    monkeypatch.setattr(
        facility_hours_fetch,
        "fetch_wp_json_html",
        lambda site_base, slug: wp_result,
    )

    result = facility_hours_fetch.collect_facility_hours(
        {
            "facilityId": 1186,
            "facilityName": "Nick",
            "slug": "nick",
            "url": "https://recwell.example.test/locations/nick/",
        },
        "https://recwell.example.test",
    )

    assert result["status"] == "ok"
    assert result["source"] == "wp_json"
    assert result["sourceModifiedGmt"] == "2026-08-31T12:00:00"


def test_changed_structure_has_exact_bs4_and_regex_fallback_parity(
    fixture_text: Callable[[str], str],
) -> None:
    if facility_hours_fetch.BeautifulSoup is None:
        pytest.skip("Beautiful Soup is not installed in this test interpreter")
    html = fixture_text("facility_hours/structure-changed.html")
    expected = [
        {
            "title": "Hours of Operation",
            "rows": [
                {"label": "Saturday", "hours": "8:00 am - 8:00 pm"},
                {"label": "Sunday", "hours": "Closed"},
            ],
            "note": None,
        }
    ]

    assert facility_hours_fetch.parse_hours_sections_with_bs4(html) == expected
    assert facility_hours_fetch.parse_hours_sections_with_regex(html) == expected


@pytest.mark.parametrize(
    "fixture",
    [
        "facility_hours/nick-direct.html",
        "facility_hours/bakke-direct.html",
        "facility_hours/nick-wp.json",
        "facility_hours/bakke-wp.json",
        "facility_hours/structure-changed.html",
    ],
)
def test_saved_sources_have_exact_bs4_and_regex_fallback_parity(
    fixture: str,
    fixture_text: Callable[[str], str],
    fixture_json: Callable[[str], object],
) -> None:
    if facility_hours_fetch.BeautifulSoup is None:
        pytest.skip("Beautiful Soup is not installed in this test interpreter")
    if fixture.endswith(".json"):
        payload = fixture_json(fixture)
        assert isinstance(payload, list) and payload
        page = payload[0]
        assert isinstance(page, dict)
        content = page.get("content")
        assert isinstance(content, dict)
        html = content.get("rendered")
        assert isinstance(html, str)
    else:
        html = fixture_text(fixture)

    assert facility_hours_fetch.parse_hours_sections_with_bs4(html) == (
        facility_hours_fetch.parse_hours_sections_with_regex(html)
    )


def test_legacy_parser_name_delegates_to_the_authoritative_fallback(
    monkeypatch: pytest.MonkeyPatch,
    fixture_text: Callable[[str], str],
) -> None:
    html = fixture_text("facility_hours/structure-changed.html")
    monkeypatch.setattr(facility_hours_fetch, "BeautifulSoup", None)

    assert facility_hours_fetch.parse_hours_sections(html) == (
        facility_hours_fetch.parse_schedule_sections(html)
    )


def test_regex_parser_imports_and_parses_when_beautiful_soup_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional parser must not disable the dependency-free fallback."""

    original_import = builtins.__import__

    def import_without_bs4(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "bs4" or name.startswith("bs4."):
            raise ModuleNotFoundError("No module named 'bs4'", name="bs4")
        return original_import(name, globals, locals, fromlist, level)

    previous_module = sys.modules.pop("facility_hours_fetch", None)
    monkeypatch.setattr(builtins, "__import__", import_without_bs4)
    try:
        try:
            facility_hours_fetch = importlib.import_module("facility_hours_fetch")
        except ModuleNotFoundError as exc:
            pytest.fail(f"optional Beautiful Soup import escaped: {exc.name}")

        assert facility_hours_fetch.BeautifulSoup is None
        html = (
            "<table><tr><th>Dates</th><th>Hours</th></tr>"
            "<tr><td>Aug 31 - Sep 4</td><td>6:00 am - 10:00 pm</td></tr>"
            "</table>"
        )
        sections = facility_hours_fetch.parse_hours_sections(html)
        assert sections[0]["rows"][0] == {
            "label": "Aug 31 - Sep 4",
            "hours": "6:00 am - 10:00 pm",
        }
    finally:
        sys.modules.pop("facility_hours_fetch", None)
        if previous_module is not None:
            sys.modules["facility_hours_fetch"] = previous_module
