# RecLive Facility-Hours Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reliably collect, validate, preserve, serve, and health-report official Nick and Bakke schedules while providing a fail-closed open-now predicate for push alerts.

**Architecture:** `facility_hours_fetch.py` will collect each facility into an in-memory candidate and merge it with the last valid published payload only after strict combined-payload validation. A small dependency-light `facility_schedule.py` module will own the existing schedule grammar and expose the fail-closed predicate used by `forecast_api.py`; `forecast_job.py` will consume the same grammar through compatibility imports so forecast behavior does not silently diverge. The published JSON remains the durable schedule artifact and is atomically replaced only after all two-facility rows have valid safe metadata.

**Tech Stack:** Python 3, Requests, Beautiful Soup 4, FastAPI, Pydantic v2, pytest, React 19, TypeScript, Vitest, React Testing Library.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md` (Phase 7), with every Phase 7 requirement from the original brief reproduced below.

## Global Constraints

- Implement Phase 7 only; preserve the existing product identity, facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, forecasting behavior, push behavior, and executable compatibility entry points.
- The master plan's phase-level commit policy is authoritative: task commit snippets describe staging/review scope only; make the single Phase 7 source commit after Task 7.
- Pin `beautifulsoup4` in `server/requirements.txt` and retain the exact runtime fallback `try: from bs4 import BeautifulSoup` / `except ImportError: BeautifulSoup = None`; the fallback parser must work when Beautiful Soup is unavailable.
- Saved fixtures live under `tests/fixtures/facility_hours/`; backend tests live under `tests/backend/`; frontend tests are colocated with the affected TypeScript source.
- A facility candidate is accepted only when it has one supported facility ID, a nonempty name/slug, a recognized source (`direct_html` or `wp_json`), and at least one schema-valid schedule section containing nonempty title plus valid `{label, hours}` rows or a bounded notice.
- Preserve existing top-level `generatedAt`, `sourceSite`, `facilities`, `okCount`, and `totalCount`, and existing facility `facilityId`, `facilityName`, `slug`, `url`, `resolvedUrl`, `status`, `source`, `sections`, `error`, and `updatedAt` fields where consumers rely on them.
- Add `sourceFetchedAt`, `lastSuccessfulAt`, `stale`, and `errorCategory`. `status` is `ok`, `stale`, or `error`; errors are one of `anti_bot`, `upstream_timeout`, `upstream_http`, `wp_payload_invalid`, `parse_empty`, `schema_invalid`, or `io_error`.
- Never persist or return raw exception strings, request headers, response bodies, credential-bearing URLs, or private redirect URLs. `error` is either `null` or a fixed safe UI message; `errorCategory` is the only diagnostic detail in the published artifact.
- A failed facility reuses its previous valid sections, source metadata, and `lastSuccessfulAt`, sets `status` to `stale`, `stale` to `true`, and records a safe category. A fresh facility updates independently. Empty, malformed, or untrusted candidates never replace a prior valid facility.
- Validate the complete two-facility artifact before writing a same-directory temporary file, flushing it, and replacing the target with `os.replace`. Do not publish a partially assembled payload.
- `official_facility_is_open(payload: Mapping[str, Any], facility_id: int, at: datetime) -> bool` must return `True` only for a fresh `status == "ok"` facility with a matching official building-hours rule that is open at the Chicago-local timestamp. It returns `False` for missing, stale, error, closed, maintenance, unparsable, and unmatched data.
- Schedule health is sanitized and includes only `state` (`healthy`, `stale`, or `unavailable`), `ageSeconds`, and the two supported facility status values. It uses canonical `SCHEDULE_STALE_AFTER_SECONDS=21600` (six hours) unless explicitly configured, while accepting `SCHEDULE_MAX_AGE_SECONDS` only as a legacy runtime fallback.
- Do not merge, deploy, contact upstream providers while testing, alter production credentials, or commit source changes while planning.

---

## File Structure

- Create: `server/facility_schedule.py` — dependency-light date/weekday/time parsing, facility-wide section selection, and the Phase 5 `official_facility_is_open` predicate.
- Modify: `server/forecast_job.py` — import the extracted schedule helpers while retaining its existing public helper names and forecasting semantics.
- Modify: `server/facility_hours_fetch.py` — optional Beautiful Soup import, fixture-testable collection, category sanitization, previous-artifact merge, combined schema validation, and atomic publish.
- Modify: `server/forecast_api.py` — validate schedule artifacts, expose fresh/stale metadata through existing schedule endpoints, use the fail-closed predicate, and add sanitized schedule health to `/health`.
- Modify: `server/requirements.txt` — add the pinned Beautiful Soup runtime dependency.
- Modify: `.env.example` — add only `SCHEDULE_STALE_AFTER_SECONDS=21600` and no provider secrets.
- Create: `tests/fixtures/facility_hours/nick-direct.html` — Nick direct-page fixture with a building-hours date range and closure notice.
- Create: `tests/fixtures/facility_hours/bakke-direct.html` — Bakke direct-page fixture with weekday rows and a maintenance notice.
- Create: `tests/fixtures/facility_hours/nick-wp.json` — Nick WordPress API response fixture containing rendered schedule HTML.
- Create: `tests/fixtures/facility_hours/bakke-wp.json` — Bakke WordPress API response fixture containing rendered schedule HTML.
- Create: `tests/fixtures/facility_hours/structure-changed.html` — valid schedule content with changed heading/table markup.
- Create: `tests/fixtures/facility_hours/anti-bot.html` — challenge page fixture with no schedule rows.
- Create: `tests/backend/test_facility_hours.py` — parser, fallback, merge, atomic publication, predicate, API, and health tests.
- Modify: `src/lib/types/facilitySchedule.ts` — represent safe freshness and stale metadata.
- Modify: `src/lib/api/schemas.ts` — tighten Phase 6's transitional schedule schema to the final Phase 7 metadata invariants.
- Modify: `src/lib/api/facilityScheduleParser.ts` — parse the extended safe schedule response.
- Create: `src/lib/api/facilityScheduleParser.test.ts` — reject malformed freshness metadata and accept a safe stale response.
- Modify: `src/facilities/FacilityHoursBlock.tsx` — visibly label preserved stale schedules without changing the schedule-table design.
- Create: `src/facilities/FacilityHoursBlock.test.tsx` — verify fresh and stale copy.

### Task 1: Pin Beautiful Soup and extract a dependency-light official-hours predicate

**Files:**
- Create: `server/facility_schedule.py`
- Modify: `server/forecast_job.py:733-1424`
- Modify: `server/facility_hours_fetch.py:1-18, 307-312`
- Modify: `server/requirements.txt`
- Create: `tests/backend/test_facility_hours.py`

**Interfaces:**
- Consumes: existing schedule labels/rows generated by `facility_hours_fetch.py` and the existing schedule grammar in `forecast_job.py`.
- Produces: `parse_schedule_date_range(value: str, fallback_year: int) -> Optional[tuple[date, date, int]]`, `parse_schedule_weekday_set(label: str) -> Optional[set[int]]`, `parse_schedule_hours_window(value: str) -> Optional[tuple[int, int, bool]]`, `get_facility_schedule_open_state(sections: list[dict[str, object]], at: datetime) -> Optional[bool]`, and `official_facility_is_open(payload: Mapping[str, Any], facility_id: int, at: datetime, *, stale_after_seconds: int = 21600) -> bool`.

- [ ] **Step 1: Write the failing extraction and fallback tests**

```python
from datetime import datetime, timezone
from zoneinfo import ZoneInfo


FIXED_NOW = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)


def iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def chicago_datetime(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/Chicago"))


def facility_record(
    facility_id: int, now: datetime, *, status: str = "ok", stale: bool = False,
    rows: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    slug = "nick" if facility_id == 1186 else "bakke"
    name = "Nick" if facility_id == 1186 else "Bakke"
    sections = [{"title": "Building Hours", "rows": rows or [{"label": "Mon-Fri", "hours": "6:00 am - 10:00 pm"}], "note": None}]
    return {
        "facilityId": facility_id, "facilityName": name, "slug": slug,
        "url": f"https://recwell.example.test/{slug}/",
        "resolvedUrl": f"https://recwell.example.test/{slug}/",
        "status": status, "source": "direct_html", "sections": sections,
        "error": None if status == "ok" else "Official hours could not be refreshed.",
        "updatedAt": iso_utc(now), "sourceFetchedAt": iso_utc(now),
        "lastSuccessfulAt": iso_utc(now), "stale": stale,
        "errorCategory": None if status == "ok" else "anti_bot",
    }


def schedule_payload(
    *, status: str = "ok", stale: bool = False,
    rows: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    facilities = [
        facility_record(1186, FIXED_NOW, status=status, stale=stale, rows=rows),
        facility_record(1656, FIXED_NOW),
    ]
    return {
        "generatedAt": iso_utc(FIXED_NOW),
        "sourceSite": "https://recwell.example.test",
        "facilities": facilities,
        "okCount": sum(row["status"] == "ok" for row in facilities),
        "totalCount": 2,
    }


def test_official_predicate_accepts_fresh_open_weekday_and_rejects_stale_payload():
    payload = schedule_payload(status="ok", stale=False, rows=[{"label": "Mon-Fri", "hours": "6:00 am - 10:00 pm"}])
    monday_noon = chicago_datetime(2026, 8, 31, 12, 0)
    assert official_facility_is_open(payload, 1186, monday_noon) is True
    payload["facilities"][0]["stale"] = True
    assert official_facility_is_open(payload, 1186, monday_noon) is False


def test_official_predicate_rejects_old_or_future_ok_artifacts():
    payload = schedule_payload(status="ok", stale=False)
    monday_noon = chicago_datetime(2026, 8, 31, 12, 0)
    payload["generatedAt"] = "2026-08-31T10:59:59Z"
    payload["facilities"][0]["lastSuccessfulAt"] = "2026-08-31T10:59:59Z"
    assert official_facility_is_open(payload, 1186, monday_noon, stale_after_seconds=21600) is False
    payload["generatedAt"] = "2026-08-31T17:00:01Z"
    payload["facilities"][0]["lastSuccessfulAt"] = "2026-08-31T17:00:01Z"
    assert official_facility_is_open(payload, 1186, monday_noon, stale_after_seconds=21600) is False


def test_regex_parser_remains_available_when_beautiful_soup_is_missing(monkeypatch):
    monkeypatch.setattr(facility_hours_fetch, "BeautifulSoup", None)
    html = "<table><tr><th>Dates</th><th>Hours</th></tr><tr><td>Aug 31 - Sep 4</td><td>6:00 am - 10:00 pm</td></tr></table>"
    sections = facility_hours_fetch.parse_hours_sections(html)
    assert sections[0]["rows"][0] == {"label": "Aug 31 - Sep 4", "hours": "6:00 am - 10:00 pm"}
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'official_predicate or regex_parser' -q`

Expected: FAIL because Beautiful Soup imports unconditionally, no shared predicate exists, and `forecast_job.py` owns the only backend schedule grammar.

- [ ] **Step 3: Add the optional import and shared predicate without importing XGBoost**

```python
# server/facility_hours_fetch.py
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
```

```python
# server/facility_schedule.py
from datetime import timezone


def official_facility_is_open(
    payload: Mapping[str, Any], facility_id: int, at: datetime, *, stale_after_seconds: int = 21_600,
) -> bool:
    if at.tzinfo is None or stale_after_seconds <= 0:
        return False
    facility = next((row for row in payload.get("facilities", []) if isinstance(row, dict) and row.get("facilityId") == facility_id), None)
    if not isinstance(facility, dict) or facility.get("status") != "ok" or facility.get("stale") is not False:
        return False
    now_utc = at.astimezone(timezone.utc)
    for raw_timestamp in (payload.get("generatedAt"), facility.get("lastSuccessfulAt")):
        observed_at = parse_utc_timestamp(raw_timestamp)
        if observed_at is None:
            return False
        age_seconds = (now_utc - observed_at).total_seconds()
        if age_seconds < 0 or age_seconds > stale_after_seconds:
            return False
    sections = facility.get("sections")
    if not isinstance(sections, list):
        return False
    return get_facility_schedule_open_state(sections, at.astimezone(CHICAGO_TZ)) is True
```

Move the existing pure schedule parsing functions from `server/forecast_job.py` into `server/facility_schedule.py`, then import them back under their existing names in `forecast_job.py`. Add a strict `parse_utc_timestamp` helper that accepts only offset-aware ISO strings and returns UTC or `None`; the predicate uses it exactly as above. Do not import `forecast_job.py` from the fetcher or API. Add the timezone-aware `iso_utc(value)` implementation shown in the test helper to `server/facility_hours_fetch.py` and use it for every published schedule timestamp. Pin `beautifulsoup4==4.12.3` in `server/requirements.txt`. Keep `parse_hours_sections` selecting Beautiful Soup when present and the existing regex parser when `BeautifulSoup is None`.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'official_predicate or regex_parser' -q`

Expected: PASS; the predicate fails closed for stale data and the saved direct-page fixture parses without Beautiful Soup.

- [ ] **Step 5: Commit the parser-contract slice**

```bash
git add server/facility_schedule.py server/forecast_job.py server/facility_hours_fetch.py server/requirements.txt tests/backend/test_facility_hours.py
git commit -m "feat(schedules): add fail-closed official hours predicate"
```

### Task 2: Add saved direct and WordPress parsing fixtures with safe source classification

**Files:**
- Create: `tests/fixtures/facility_hours/nick-direct.html`
- Create: `tests/fixtures/facility_hours/bakke-direct.html`
- Create: `tests/fixtures/facility_hours/nick-wp.json`
- Create: `tests/fixtures/facility_hours/bakke-wp.json`
- Create: `tests/fixtures/facility_hours/structure-changed.html`
- Create: `tests/fixtures/facility_hours/anti-bot.html`
- Modify: `server/facility_hours_fetch.py:102-367`
- Modify: `tests/backend/test_facility_hours.py`

**Interfaces:**
- Consumes: `parse_hours_sections(html: str) -> list[dict[str, Any]]` from Task 1 and the fixture loader in `tests/backend/test_facility_hours.py`.
- Produces: `parse_schedule_sections(html: str) -> list[dict[str, Any]]`, `looks_like_bot_challenge(html: str) -> bool`, `parse_wp_page_payload(payload: Any) -> tuple[str, Optional[str], Optional[str]]`, and `ScheduleFetchError(category: str)`.

- [ ] **Step 1: Write the failing fixture-driven parser tests**

```python
import json
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.fixture
def fixture_text():
    def load(relative_path: str) -> str:
        return (FIXTURE_ROOT / relative_path).read_text(encoding="utf-8")
    return load


@pytest.fixture
def fixture_json():
    def load(relative_path: str) -> object:
        return json.loads((FIXTURE_ROOT / relative_path).read_text(encoding="utf-8"))
    return load


@pytest.mark.parametrize(("fixture", "expected_label", "expected_hours"), [
    ("facility_hours/nick-direct.html", "Aug 31 - Sep 4", "6:00 am - 10:00 pm"),
    ("facility_hours/bakke-direct.html", "Weekdays", "5:30 am - 11:00 pm"),
    ("facility_hours/structure-changed.html", "Saturday", "8:00 am - 8:00 pm"),
])
def test_direct_html_fixtures_preserve_schedule_rows(fixture_text, fixture, expected_label, expected_hours):
    sections = parse_schedule_sections(fixture_text(fixture))
    rows = [row for section in sections for row in section["rows"]]
    assert {"label": expected_label, "hours": expected_hours} in rows


@pytest.mark.parametrize(("fixture", "facility_id"), [("facility_hours/nick-wp.json", 1186), ("facility_hours/bakke-wp.json", 1656)])
def test_wordpress_fixture_yields_rendered_schedule_html(fixture_json, fixture, facility_id):
    html, modified_at, resolved_url = parse_wp_page_payload(fixture_json(fixture))
    assert facility_id in (1186, 1656)
    assert parse_schedule_sections(html)
    assert modified_at == "2026-08-31T12:00:00"
    assert resolved_url is not None


def test_antibot_fixture_is_rejected_with_a_category_not_html(fixture_text):
    with pytest.raises(ScheduleFetchError, match="anti_bot"):
        parse_direct_response(fixture_text("facility_hours/anti-bot.html"), "https://recwell.example.test/nick/")
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'fixtures_preserve or wordpress_fixture or antibot_fixture' -q`

Expected: FAIL because there are no saved fixtures, no WordPress payload parser seam, and direct-response errors use arbitrary exception strings.

- [ ] **Step 3: Create representative fixtures and category-only parser seams**

```html
<!-- tests/fixtures/facility_hours/nick-direct.html -->
<h2>Nick Building Hours</h2>
<table><tr><th>Dates</th><th>Hours</th></tr><tr><td>Aug 31 - Sep 4</td><td>6:00 am - 10:00 pm</td></tr><tr><td>Sep 5</td><td>Closed</td></tr></table>
<p>Maintenance closure: the building closes at 6:00 pm on Sep 6.</p>
```

```python
class ScheduleFetchError(RuntimeError):
    def __init__(self, category: str) -> None:
        super().__init__(category)
        self.category = category

def parse_direct_response(html: str, resolved_url: str) -> tuple[list[dict[str, Any]], str]:
    if looks_like_bot_challenge(html):
        raise ScheduleFetchError("anti_bot")
    sections = parse_schedule_sections(html)
    if not sections:
        raise ScheduleFetchError("parse_empty")
    return sections, resolved_url
```

Give the Bakke fixture a weekday row and a maintenance notice. Give the Nick WordPress fixture a `content.rendered`, `modified_gmt`, and `link` value containing no query credentials; give the Bakke WordPress fixture the equivalent values with different safe fixture text. The structure-change fixture must use a non-table schedule container that the parser recognizes by scanning paired date/hours text. The anti-bot fixture must include `Checking your browser` and no facility schedule. Map timeouts, HTTP failures, malformed WordPress payloads, empty parses, and file errors to the fixed category list in Global Constraints.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'fixtures_preserve or wordpress_fixture or antibot_fixture' -q`

Expected: PASS; Nick/Bakke direct and WordPress fixtures parse, changed structure remains recognized, and the anti-bot fixture yields only `anti_bot`.

- [ ] **Step 5: Commit the fixture/parser slice**

```bash
git add tests/fixtures/facility_hours server/facility_hours_fetch.py tests/backend/test_facility_hours.py
git commit -m "test(schedules): cover official hours source formats"
```

### Task 3: Build safe per-facility candidates and preserve valid data on partial failure

**Files:**
- Modify: `server/facility_hours_fetch.py:370-415`
- Modify: `tests/backend/test_facility_hours.py`

**Interfaces:**
- Consumes: `ScheduleFetchError`, direct/WordPress parsing seams from Task 2, and a previous schema-valid facility record.
- Produces: `collect_facility_candidate(facility: Mapping[str, Any], site_base: str, fetched_at: datetime) -> FacilityCandidate`, `merge_facility_candidate(candidate: FacilityCandidate, previous: Optional[Mapping[str, Any]], generated_at: datetime) -> dict[str, Any]`, and `safe_error_message(category: str) -> str`.

- [ ] **Step 1: Write the failing independent-facility merge test**

```python
def test_one_failed_facility_keeps_last_known_good_sections_while_other_updates(monkeypatch, previous_payload, fixed_now):
    monkeypatch.setattr(facility_hours_fetch, "collect_facility_candidate", lambda facility, site_base, fetched_at: (
        failed_candidate(1186, "anti_bot") if facility["facilityId"] == 1186 else fresh_candidate(1656, "Bakke fresh rows", fixed_now)
    ))
    payload = build_combined_payload(build_facilities(), previous_payload, "https://recwell.example.test", fixed_now)
    nick, bakke = payload["facilities"]
    assert (nick["status"], nick["stale"], nick["sections"], nick["lastSuccessfulAt"]) == ("stale", True, previous_payload["facilities"][0]["sections"], previous_payload["facilities"][0]["lastSuccessfulAt"])
    assert (bakke["status"], bakke["stale"], bakke["sections"][0]["title"]) == ("ok", False, "Bakke fresh rows")
    assert "Checking your browser" not in nick["error"]
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/backend/test_facility_hours.py::test_one_failed_facility_keeps_last_known_good_sections_while_other_updates -q`

Expected: FAIL because `collect_facility_hours` returns an error row with empty sections and `main` overwrites the full artifact with that row.

- [ ] **Step 3: Implement candidate/merge state with sanitized metadata**

```python
@dataclass(frozen=True)
class FacilityCandidate:
    facility_id: int
    facility_name: str
    slug: str
    public_url: str
    source: str | None
    resolved_url: str | None
    sections: tuple[dict[str, object], ...]
    fetched_at: datetime
    error_category: str | None

    @property
    def ok(self) -> bool:
        return self.error_category is None and self.source in {"direct_html", "wp_json"} and bool(self.sections)

    def as_public_record(self) -> dict[str, object]:
        if not self.ok:
            raise ValueError("failed schedule candidate has no public record")
        return {
            "facilityId": self.facility_id, "facilityName": self.facility_name,
            "slug": self.slug, "url": self.public_url,
            "resolvedUrl": self.resolved_url, "source": self.source,
            "sections": list(self.sections),
        }


def merge_facility_candidate(candidate: FacilityCandidate, previous: Optional[Mapping[str, Any]], generated_at: datetime) -> dict[str, Any]:
    if candidate.ok:
        return {**candidate.as_public_record(), "status": "ok", "stale": False, "error": None, "errorCategory": None, "sourceFetchedAt": iso_utc(candidate.fetched_at), "lastSuccessfulAt": iso_utc(candidate.fetched_at), "updatedAt": iso_utc(generated_at)}
    if previous_is_valid_facility(previous, candidate.facility_id):
        return {**copy_public_schedule(previous), "status": "stale", "stale": True, "error": safe_error_message(candidate.error_category), "errorCategory": candidate.error_category, "updatedAt": iso_utc(generated_at)}
    return {"facilityId": candidate.facility_id, "facilityName": candidate.facility_name, "slug": candidate.slug, "url": candidate.public_url, "resolvedUrl": None, "status": "error", "source": None, "sections": [], "error": safe_error_message(candidate.error_category), "errorCategory": candidate.error_category, "sourceFetchedAt": None, "lastSuccessfulAt": None, "stale": False, "updatedAt": iso_utc(generated_at)}
```

In `tests/backend/test_facility_hours.py`, define `fixed_now` as a fixture returning `FIXED_NOW`, `previous_payload` as a deep copy of `schedule_payload()`, `failed_candidate(id, category)` as a `FacilityCandidate` with no source/sections and the fixed safe category, and `fresh_candidate(id, title, now)` as a direct-HTML candidate with one `{title, rows: [{label: "Mon-Fri", hours: "6:00 am - 10:00 pm"}], note: None}` section. These are test builders, not production shortcuts.

`FacilityCandidate` never stores a raw exception after classification. Fresh direct parsing wins; WordPress parsing is attempted only after direct parsing produces a safe category. A failure copies only a fully valid prior record, including its source timestamps and sections. Set `resolvedUrl` only from an HTTPS URL on the configured RecWell origin; otherwise set it to `None`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `python -m pytest tests/backend/test_facility_hours.py::test_one_failed_facility_keeps_last_known_good_sections_while_other_updates -q`

Expected: PASS; Nick is visibly stale with prior validated data, Bakke is fresh, and no upstream HTML or exception phrase reaches the artifact.

- [ ] **Step 5: Commit the per-facility resilience slice**

```bash
git add server/facility_hours_fetch.py tests/backend/test_facility_hours.py
git commit -m "feat(schedules): preserve valid hours on partial failures"
```

### Task 4: Validate the complete artifact and atomically publish it

**Files:**
- Modify: `server/facility_hours_fetch.py:437-490`
- Modify: `tests/backend/test_facility_hours.py`

**Interfaces:**
- Consumes: `merge_facility_candidate` from Task 3 and fixed supported facilities from `DEFAULT_FACILITIES`.
- Produces: `validate_schedule_payload(payload: Mapping[str, Any]) -> dict[str, Any]`, `build_combined_payload(facilities: Sequence[Mapping[str, Any]], previous_payload: Optional[Mapping[str, Any]], site_base: str, generated_at: datetime) -> dict[str, Any]`, and `atomic_write_json(path: str, payload: Mapping[str, Any]) -> None`.

- [ ] **Step 1: Write the failing validation and atomic-write tests**

```python
def test_invalid_candidate_does_not_replace_the_existing_artifact(tmp_path, previous_payload):
    target = tmp_path / "facility_hours.json"
    target.write_text(json.dumps(previous_payload), encoding="utf-8")
    malformed = {"generatedAt": "2026-08-31T12:00:00Z", "facilities": [{"facilityId": 1186}]}
    with pytest.raises(ValueError, match="schedule payload"):
        atomic_write_json(str(target), malformed)
    assert json.loads(target.read_text(encoding="utf-8")) == previous_payload


def test_complete_payload_requires_each_supported_facility_once(fixed_now):
    payload = {"generatedAt": iso_utc(fixed_now), "sourceSite": "https://recwell.example.test", "facilities": [facility_record(1186, fixed_now)]}
    with pytest.raises(ValueError, match="1656"):
        validate_schedule_payload(payload)
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'invalid_candidate or requires_each_supported' -q`

Expected: FAIL because `atomic_write_json` and `validate_schedule_payload` do not yet exist; the existing writer has no complete-artifact guard.

- [ ] **Step 3: Implement full schema validation and same-directory replacement**

```python
def atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    normalized = validate_schedule_payload(payload)
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary_path = tempfile.mkstemp(prefix=".facility-hours-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
```

Require exactly `[1186, 1656]` once each, a valid ISO-8601 UTC `generatedAt`, `sourceSite` without query/fragment text, `okCount` equal to the count of `status == "ok"`, `totalCount == 2`, valid sections for `ok`/`stale`, and null source/last-success timestamps only for `error`. Load the previous target before collection; if its schema is invalid, treat it as unavailable rather than copying any of its fields. Replace `write_json` with `atomic_write_json` and log only the final facility status counts, never source URLs, exceptions, or row content.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'invalid_candidate or requires_each_supported' -q`

Expected: PASS; malformed output leaves the previous artifact byte-for-byte intact and every publishable artifact has both supported facility records.

- [ ] **Step 5: Commit the atomic-publication slice**

```bash
git add server/facility_hours_fetch.py tests/backend/test_facility_hours.py
git commit -m "feat(schedules): validate and atomically publish hours"
```

### Task 5: Serve schedule freshness and preserve Phase 5’s fail-closed API contract

**Files:**
- Modify: `server/forecast_api.py:102-120, 227-280, 488-505, 822-855`
- Modify: `.env.example`
- Modify: `tests/backend/test_facility_hours.py`

**Interfaces:**
- Consumes: `validate_schedule_payload`, `official_facility_is_open`, and the published metadata from Tasks 1-4.
- Produces: `schedule_health(payload: Mapping[str, Any], now: datetime) -> dict[str, Any]`, extended `GET /health`, unchanged `GET /api/facility-hours`, unchanged `GET /api/facility-hours/facilities/{facility_id}`, and an imported `official_facility_is_open` with the Task 1 signature and configured stale threshold.

- [ ] **Step 1: Write the failing API/health compatibility tests**

```python
@pytest.fixture
def schedule_test_client(monkeypatch, tmp_path):
    def build(payload: Mapping[str, Any], now: datetime) -> TestClient:
        schedule_path = tmp_path / "facility_hours.json"
        schedule_path.write_text(json.dumps(payload), encoding="utf-8")
        monkeypatch.setattr(forecast_api, "FACILITY_HOURS_JSON_PATH", str(schedule_path))
        monkeypatch.setattr(forecast_api, "now_utc", lambda: now)
        monkeypatch.setattr(forecast_api, "load_forecast", lambda: {"generatedAt": iso_utc(now), "facilities": [], "modelInfo": {"status": "fixture"}})
        monkeypatch.setattr(forecast_api, "evaluator_enabled", lambda: False)
        return TestClient(forecast_api.app)
    return build


def test_schedule_api_exposes_safe_freshness_metadata(schedule_test_client):
    response = schedule_test_client(schedule_payload(), FIXED_NOW).get("/api/facility-hours/facilities/1186")
    assert response.status_code == 200
    assert response.json()["stale"] is False
    assert response.json()["lastSuccessfulAt"] == "2026-08-31T12:00:00Z"
    assert "exception" not in response.text.lower()


def test_health_marks_old_or_stale_schedule_without_paths(schedule_test_client):
    now = FIXED_NOW + timedelta(seconds=21601)
    response = schedule_test_client(schedule_payload(status="stale", stale=True), now).get("/health")
    assert response.json()["schedule"] == {"state": "stale", "ageSeconds": 21601, "facilities": {"1186": "stale", "1656": "ok"}}
    assert "facility_hours.json" not in response.text


def test_phase_five_predicate_rejects_maintenance_and_unmatched_schedule_rows():
    at = chicago_datetime(2026, 8, 31, 12, 0)
    maintenance = schedule_payload(rows=[{"label": "Maintenance", "hours": "Closed"}])
    area_only = schedule_payload(rows=[{"label": "Pool", "hours": "6:00 am - 10:00 pm"}])
    assert official_facility_is_open(maintenance, 1186, at) is False
    assert official_facility_is_open(area_only, 1186, at) is False
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'schedule_api or health_marks or phase_five_predicate' -q`

Expected: FAIL because the API returns raw current fields only, `/health` has no schedule state, and no shared Phase 5 predicate is imported by the API.

- [ ] **Step 3: Add safe metadata validation and health state**

```python
def schedule_health(payload: Mapping[str, Any], now: datetime) -> dict[str, Any]:
    generated_at = parse_utc_timestamp(payload.get("generatedAt"))
    age_seconds = None if generated_at is None else max(0, int((now - generated_at).total_seconds()))
    statuses = {str(row["facilityId"]): str(row["status"]) for row in payload["facilities"]}
    state = "healthy" if age_seconds is not None and age_seconds <= SCHEDULE_STALE_AFTER_SECONDS and all(status == "ok" for status in statuses.values()) else "stale"
    if age_seconds is None or all(status == "error" for status in statuses.values()):
        state = "unavailable"
    return {"state": state, "ageSeconds": age_seconds, "facilities": statuses}
```

Set `SCHEDULE_STALE_AFTER_SECONDS = int_with_default("SCHEDULE_STALE_AFTER_SECONDS", int_with_default("SCHEDULE_MAX_AGE_SECONDS", 21600))`, add only `SCHEDULE_STALE_AFTER_SECONDS=21600` to `.env.example`, and add a `now_utc()` clock seam. Pass that positive configured value to `official_facility_is_open(..., stale_after_seconds=SCHEDULE_STALE_AFTER_SECONDS)` from the Phase 5 evaluator, so an old-but-still-marked-`ok` artifact fails closed. Make `load_facility_hours()` call `validate_schedule_payload` before returning the artifact and return a stable `503` schedule-unavailable category for a missing, unreadable, or invalid artifact. Add transitional top-level `schedule` evidence to the existing `/health` response without emitting paths, upstream URLs, or exception details. Phase 10 must consume this exact evidence in `components.schedules` and remove the transitional key in the same tested health-router replacement. Leave the existing schedule route paths and core response fields unchanged while forwarding the new safe fields. Import `official_facility_is_open` into `forecast_api.py` rather than recreating date/time grammar there.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_facility_hours.py -k 'schedule_api or health_marks or phase_five_predicate' -q`

Expected: PASS; callers receive safe freshness fields, health marks old/partial data stale, and Phase 5 has a predictable fail-closed predicate.

- [ ] **Step 5: Commit the API/health slice**

```bash
git add server/forecast_api.py .env.example tests/backend/test_facility_hours.py
git commit -m "feat(schedules): report freshness and safe health"
```

### Task 6: Parse and present stale schedule metadata in the existing UI

**Files:**
- Modify: `src/lib/types/facilitySchedule.ts`
- Modify: `src/lib/api/schemas.ts`
- Modify: `src/lib/api/facilityScheduleParser.ts`
- Create: `src/lib/api/facilityScheduleParser.test.ts`
- Modify: `src/facilities/FacilityHoursBlock.tsx`
- Create: `src/facilities/FacilityHoursBlock.test.tsx`

**Interfaces:**
- Consumes: API facility response fields from Task 5.
- Produces: `FacilityHoursFacilityPayload.sourceFetchedAt`, `FacilityHoursFacilityPayload.lastSuccessfulAt`, `FacilityHoursFacilityPayload.stale`, `FacilityHoursFacilityPayload.errorCategory`, and a visible stale-data notice that preserves the official-hours table.

- [ ] **Step 1: Write the failing frontend parser and display tests**

```ts
import {HttpResponse, http} from "msw";
import {server} from "../../test/msw/server";

it("retains a server-marked stale schedule with safe metadata", async () => {
    server.use(http.get("*/api/facility-hours/facilities/1186", () => HttpResponse.json({
        generatedAt: "2026-08-31T12:00:00Z", sourceSite: "https://recwell.example.test",
        facilityId: 1186, facilityName: "Nick", slug: "nick",
        url: "https://recwell.example.test/nick/", resolvedUrl: "https://recwell.example.test/nick/",
        status: "stale", source: "direct_html", stale: true,
        sourceFetchedAt: "2026-08-30T12:00:00Z", lastSuccessfulAt: "2026-08-30T12:00:00Z",
        error: "Official hours could not be refreshed.", errorCategory: "anti_bot",
        updatedAt: "2026-08-31T12:00:00Z",
        sections: [{title: "Building Hours", rows: [{label: "Mon", hours: "6:00 am - 10:00 pm"}], note: null}],
    })));
    await expect(fetchFacilityHours(1186)).resolves.toMatchObject({stale: true, errorCategory: "anti_bot"});
});
```

```tsx
const staleSchedule: FacilityHoursFacilityPayload = {
    generatedAt: "2026-08-31T12:00:00Z",
    sourceSite: "https://recwell.example.test",
    facilityId: 1186,
    facilityName: "Nick",
    slug: "nick",
    url: "https://recwell.example.test/nick/",
    resolvedUrl: "https://recwell.example.test/nick/",
    status: "stale",
    source: "direct_html",
    sections: [{title: "Building Hours", rows: [{label: "Mon", hours: "6:00 am - 10:00 pm"}], note: null}],
    sourceFetchedAt: "2026-08-30T12:00:00Z",
    lastSuccessfulAt: "2026-08-30T12:00:00Z",
    stale: true,
    error: "Official hours could not be refreshed.",
    errorCategory: "anti_bot",
    updatedAt: "2026-08-31T12:00:00Z",
};

it("labels preserved schedule data as stale without hiding its rows", () => {
    render(<FacilityHoursBlock facilityName="Nick" isLoading={false} error={null} schedule={staleSchedule} />);
    expect(screen.getByText(/official hours may be out of date/i)).toBeVisible();
    expect(screen.getByText("6:00 am - 10:00 pm")).toBeVisible();
});
```

- [ ] **Step 2: Run the focused frontend tests to verify they fail**

Run: `npm run test:run -- src/lib/api/facilityScheduleParser.test.ts src/facilities/FacilityHoursBlock.test.tsx`

Expected: FAIL because the current TypeScript payload ignores source freshness/stale fields and the UI does not distinguish preserved stale hours from current official hours.

- [ ] **Step 3: Add safe client types, parsing, and minimal stale copy**

```ts
export interface FacilityHoursFacilityPayload {
    generatedAt: string | null;
    sourceSite: string | null;
    sourceFetchedAt: string | null;
    lastSuccessfulAt: string | null;
    stale: boolean;
    errorCategory: "anti_bot" | "upstream_timeout" | "upstream_http" | "wp_payload_invalid" | "parse_empty" | "schema_invalid" | "io_error" | null;
    facilityId: FacilityId;
    facilityName: string;
    slug: string;
    url: string;
    resolvedUrl: string | null;
    status: "ok" | "stale";
    source: "direct_html" | "wp_json";
    sections: FacilityHoursSection[];
    error: string | null;
    updatedAt: string | null;
}
```

```tsx
{schedule?.stale && (
    <Typography variant="body2" color="warning.main">
        Official hours may be out of date. Showing the last verified schedule.
    </Typography>
)}
```

Parse valid ISO strings as strings without rendering `errorCategory` to visitors. Treat non-boolean `stale`, invalid facility IDs, and malformed `sections` as invalid response data. Preserve the existing accordion, rows, notices, loading behavior, and no-redesign constraint.

Tighten `facilityScheduleSchema` so `sourceFetchedAt`, `lastSuccessfulAt`, `stale`, and `errorCategory` are no longer optional. Add a `superRefine` rule: `ok` requires `stale === false`, nonempty sections, and non-null source/last-success times; `stale` requires `stale === true`, nonempty preserved sections, and a non-null `errorCategory`; `error` is rejected by the successful single-facility parser. Keep `generatedAt`, `sourceSite`, URL, source, error, and update timestamps schema-validated and retain `.strict()`.

- [ ] **Step 4: Run the focused frontend tests to verify they pass**

Run: `npm run test:run -- src/lib/api/facilityScheduleParser.test.ts src/facilities/FacilityHoursBlock.test.tsx && npm run build`

Expected: PASS; the typed parser accepts only safe fields and a stale schedule remains visible with an honest freshness notice.

- [ ] **Step 5: Commit the frontend trust slice**

```bash
git add src/lib/types/facilitySchedule.ts src/lib/api/schemas.ts src/lib/api/facilityScheduleParser.ts src/lib/api/facilityScheduleParser.test.ts src/facilities/FacilityHoursBlock.tsx src/facilities/FacilityHoursBlock.test.tsx
git commit -m "feat(schedules): label stale official hours"
```

### Task 7: Verify the full Phase 7 artifact lifecycle and compatibility entry points

**Files:**
- Modify: `tests/backend/test_facility_hours.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: fetcher CLI `python server/facility_hours_fetch.py --output <path>`, atomic output contract, `official_facility_is_open`, API health response, and frontend stale-display contract.
- Produces: a documented schedule-fetch command and a verified Phase 7 test suite that proves parsing, partial failure, atomic retention, freshness, and Phase 5 compatibility.

- [ ] **Step 1: Write the final end-to-end backend regression test**

```python
def test_fetch_cli_publishes_valid_two_facility_payload_and_preserves_phase_five_predicate(tmp_path, monkeypatch, fixed_now, fixture_text):
    output_path = tmp_path / "facility_hours.json"
    def fixture_backed_direct_fetch(url: str) -> tuple[str, str]:
        fixture = "facility_hours/nick-direct.html" if "/nick" in url else "facility_hours/bakke-direct.html"
        return fixture_text(fixture), url

    monkeypatch.setattr(facility_hours_fetch, "now_utc", lambda: fixed_now)
    monkeypatch.setattr(facility_hours_fetch, "fetch_direct_html", fixture_backed_direct_fetch)
    assert facility_hours_fetch.main_for_output(str(output_path)) == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [row["facilityId"] for row in payload["facilities"]] == [1186, 1656]
    assert all(row["status"] == "ok" and row["stale"] is False for row in payload["facilities"])
    assert official_facility_is_open(payload, 1186, chicago_datetime(2026, 8, 31, 12, 0)) is True
```

- [ ] **Step 2: Run the final regression test to verify it fails**

Run: `python -m pytest tests/backend/test_facility_hours.py::test_fetch_cli_publishes_valid_two_facility_payload_and_preserves_phase_five_predicate -q`

Expected: FAIL because the current CLI does not expose a testable `main_for_output`, does not require a complete validated payload, and does not provide the Phase 5 predicate.

- [ ] **Step 3: Close the CLI/documentation contract with the smallest code**

```python
def main_for_output(output_path: str) -> int:
    generated_at = now_utc()
    previous = load_previous_payload(output_path)
    payload = build_combined_payload(build_facilities(), previous, env_with_default("RECWELL_SITE_BASE", DEFAULT_SITE_BASE), generated_at)
    atomic_write_json(output_path, payload)
    return 0 if payload["okCount"] == payload["totalCount"] else 1
```

Add this README command under backend operations:

```bash
python server/facility_hours_fetch.py --output server/facility_hours.json
```

Document that the command writes a fully validated two-facility artifact atomically; a one-facility scrape failure retains only an earlier valid schedule for that facility, marks it stale, and exits nonzero while still preserving the valid fresh facility. Document canonical `SCHEDULE_STALE_AFTER_SECONDS=21600`, the legacy `SCHEDULE_MAX_AGE_SECONDS` runtime fallback, and that errors never expose upstream request data.

- [ ] **Step 4: Run the Phase 7 verification commands**

Run: `python -m pytest tests/backend/test_facility_hours.py -q && npm run test:run -- src/lib/api/facilityScheduleParser.test.ts src/facilities/FacilityHoursBlock.test.tsx && npm run lint && npm run build && git diff --check`

Expected: PASS for all executed checks. If the MySQL-backed Phase 1 test environment is unavailable, it is not required for this file-based Phase 7 suite; record it as unexecuted rather than inferring a result.

- [ ] **Step 5: Commit the verified Phase 7 completion**

```bash
git add server/facility_schedule.py server/forecast_job.py server/facility_hours_fetch.py server/forecast_api.py server/requirements.txt .env.example tests/fixtures/facility_hours tests/backend/test_facility_hours.py src/lib/types/facilitySchedule.ts src/lib/api/schemas.ts src/lib/api/facilityScheduleParser.ts src/lib/api/facilityScheduleParser.test.ts src/facilities/FacilityHoursBlock.tsx src/facilities/FacilityHoursBlock.test.tsx README.md
git commit -m "feat(schedules): harden official hours ingestion"
```

## Self-Review

**Spec coverage:** Task 1 pins Beautiful Soup and keeps a genuine fallback while defining the Phase 5 fail-closed predicate. Task 2 creates saved Nick/Bakke direct and WordPress fixtures plus structure-change, date-range, weekday, closure, maintenance, and anti-bot coverage. Tasks 3-4 implement independent facility updates, last-known-good preservation, stale/category metadata, complete validation, and atomic publication. Task 5 adds API freshness and sanitized schedule health. Task 6 preserves visitor-facing schedule rows while truthfully labelling stale output. Task 7 verifies the CLI and documents safe operation.

**Placeholder scan:** This plan contains no unresolved-work markers, deferred implementation wording, generic error instructions, or cross-task shorthand. Each task lists exact files and interfaces, a concrete failing test, a RED command/result, minimal implementation code, a GREEN command/result, and a commit.

**Type consistency:** The only fresh schedule status is `ok`; preservation uses `stale`; no prior data uses `error`. `sourceFetchedAt`, `lastSuccessfulAt`, `stale`, and `errorCategory` retain identical spellings in artifact validation, API output, frontend parsing, and presentation. `official_facility_is_open(payload, facility_id, at)` has the same boolean, fail-closed signature required by the Phase 5 evaluator.
