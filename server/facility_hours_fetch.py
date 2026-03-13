import argparse
import html as html_lib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

DEFAULT_OUTPUT_FILE = "facility_hours.json"
DEFAULT_SITE_BASE = "https://recwell.wisc.edu"
DEFAULT_FACILITIES = [
    {
        "facilityId": 1186,
        "slug": "nick",
        "facilityName": "Nick",
        "url": "https://recwell.wisc.edu/locations/nick/",
    },
    {
        "facilityId": 1656,
        "slug": "bakke",
        "facilityName": "Bakke",
        "url": "https://recwell.wisc.edu/locations/bakke/",
    },
]

DAY_HINT_RE = re.compile(
    r"\b("
    r"mon(day)?|tue(s|sday)?|wed(nesday)?|thu(r|rs|rsday)?|fri(day)?|"
    r"sat(urday)?|sun(day)?|daily|weekdays?|weekends?"
    r")\b",
    re.IGNORECASE,
)
DATE_LABEL_RE = re.compile(
    r"\b("
    r"date|dates|"
    r"jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|"
    r"aug(ust)?|sep(t|tember)?|oct(ober)?|nov(ember)?|dec(ember)?|"
    r"\d{1,2}/\d{1,2}(/\d{2,4})?|"
    r"\d{4}-\d{2}-\d{2}"
    r")\b",
    re.IGNORECASE,
)
HOURS_HINT_RE = re.compile(
    r"(am|pm|closed|24\s*hours?|noon|midnight|\d{1,2}(:\d{2})?\s*(am|pm)?)",
    re.IGNORECASE,
)
NOTICE_HOURS_RE = re.compile(
    r"check back later for [^.]+? hours\.",
    re.IGNORECASE,
)
NOTICE_MAINTENANCE_RE = re.compile(
    r"no scheduled maintenance closures at this time\.?",
    re.IGNORECASE,
)
TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.IGNORECASE | re.DOTALL)
ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
CELL_RE = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.IGNORECASE | re.DOTALL)
HEADING_RE = re.compile(r"<h([2-5])[^>]*>(.*?)</h\1>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")


def env_with_default(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    return value if value else default


def resolve_path(raw: str) -> str:
    if os.path.isabs(raw):
        return raw
    # Always resolve relative paths from the script directory.
    # If caller passes "server/foo.json" from repo root by mistake,
    # normalize it to script-local "foo.json".
    normalized = raw.replace("\\", "/")
    if normalized.startswith("server/"):
        normalized = normalized.split("/", 1)[1]
    return os.path.abspath(os.path.join(SCRIPT_DIR, normalized))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def is_header_row(left: str, right: str) -> bool:
    left_norm = clean_text(left).lower()
    right_norm = clean_text(right).lower()
    if left_norm in {"day", "days", "date", "dates"} and "hour" in right_norm:
        return True
    if left_norm == "hours" and right_norm in {"", "time"}:
        return True
    return False


def looks_like_hours_row(label: str, hours_value: str) -> bool:
    label_norm = clean_text(label)
    hours_norm = clean_text(hours_value)
    if not label_norm or not hours_norm:
        return False
    has_label_hint = bool(DAY_HINT_RE.search(label_norm) or DATE_LABEL_RE.search(label_norm))
    has_hours_hint = bool(HOURS_HINT_RE.search(hours_norm))
    return has_label_hint and has_hours_hint


def strip_html(value: str) -> str:
    with_breaks = re.sub(r"<br\s*/?>", " ", value, flags=re.IGNORECASE)
    without_tags = TAG_RE.sub(" ", with_breaks)
    return clean_text(html_lib.unescape(without_tags))


def extract_rows_from_table(table: Any) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for tr in table.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in tr.find_all(["th", "td"])]
        if len(cells) < 2:
            continue

        left = cells[0]
        right = cells[1]
        if is_header_row(left, right):
            continue
        if not left or not right:
            continue

        rows.append({"label": left, "hours": right})
    return rows


def find_table_heading(table: Any) -> str:
    preferred_tags = {"h2", "h3", "h4", "h5"}
    fallback: Optional[str] = None
    for prev in table.find_all_previous(limit=16):
        if not getattr(prev, "name", None):
            continue
        text = clean_text(prev.get_text(" ", strip=True))
        if not text:
            continue
        if len(text) > 140:
            continue
        tag_name = str(prev.name).lower()

        if tag_name in preferred_tags:
            if any(keyword in text.lower() for keyword in ("hour", "schedule", "building")):
                return text
            if fallback is None:
                fallback = text
    return fallback or "Hours"


def dedupe_sections(sections: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output: List[Dict[str, Any]] = []
    for section in sections:
        key = (
            clean_text(str(section.get("title", ""))).lower(),
            json.dumps(section.get("rows", []), sort_keys=True, ensure_ascii=False),
            clean_text(str(section.get("note", ""))).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(section)
    return output


def parse_hours_sections_with_bs4(html: str) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    sections: List[Dict[str, Any]] = []

    for table in soup.find_all("table"):
        rows = extract_rows_from_table(table)
        if len(rows) < 1:
            continue

        matching_rows = [row for row in rows if looks_like_hours_row(row["label"], row["hours"])]
        if len(matching_rows) < 1:
            continue

        sections.append(
            {
                "title": find_table_heading(table),
                "rows": matching_rows,
            }
        )

    for match in NOTICE_HOURS_RE.finditer(soup.get_text("\n", strip=True)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Seasonal Notice",
                "rows": [],
                "note": notice,
            }
        )

    for match in NOTICE_MAINTENANCE_RE.finditer(soup.get_text("\n", strip=True)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Maintenance Closures",
                "rows": [],
                "note": notice,
            }
        )

    return dedupe_sections(sections)


def find_heading_before_table(html: str, table_start: int) -> str:
    fallback: Optional[str] = None
    for match in HEADING_RE.finditer(html):
        if match.start() >= table_start:
            break

        text = strip_html(match.group(2))
        if not text:
            continue
        if len(text) > 140:
            continue
        fallback = text
        if any(keyword in text.lower() for keyword in ("hour", "schedule", "building")):
            return text
    return fallback or "Hours"


def parse_hours_sections_with_regex(html: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    for table_match in TABLE_RE.finditer(html):
        table_html = table_match.group(1)
        rows: List[Dict[str, str]] = []

        for row_match in ROW_RE.finditer(table_html):
            cells_raw = [strip_html(cell) for cell in CELL_RE.findall(row_match.group(1))]
            if len(cells_raw) < 2:
                continue

            left = cells_raw[0]
            right = cells_raw[1]
            if is_header_row(left, right):
                continue
            if not left or not right:
                continue
            rows.append({"label": left, "hours": right})

        if len(rows) < 1:
            continue

        matching_rows = [row for row in rows if looks_like_hours_row(row["label"], row["hours"])]
        if len(matching_rows) < 1:
            continue

        sections.append(
            {
                "title": find_heading_before_table(html, table_match.start()),
                "rows": matching_rows,
            }
        )

    for match in NOTICE_HOURS_RE.finditer(strip_html(html)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Seasonal Notice",
                "rows": [],
                "note": notice,
            }
        )

    for match in NOTICE_MAINTENANCE_RE.finditer(strip_html(html)):
        notice = clean_text(match.group(0))
        if not notice:
            continue
        sections.append(
            {
                "title": "Maintenance Closures",
                "rows": [],
                "note": notice,
            }
        )

    return dedupe_sections(sections)


def parse_hours_sections(html: str) -> List[Dict[str, Any]]:
    if BeautifulSoup is not None:
        return parse_hours_sections_with_bs4(html)
    return parse_hours_sections_with_regex(html)


def looks_like_bot_challenge(html: str) -> bool:
    lowered = html.lower()
    markers = (
        "please enable javascript",
        "checking your browser",
        "challenge-platform",
        "just a moment...",
        "cf-chl-",
    )
    return any(marker in lowered for marker in markers)


def fetch_direct_html(url: str) -> Tuple[str, str]:
    response = requests.get(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=25,
    )
    response.raise_for_status()
    text = response.text
    if looks_like_bot_challenge(text):
        raise RuntimeError("blocked by anti-bot challenge on direct page fetch")
    return text, response.url


def fetch_wp_json_html(site_base: str, slug: str) -> Tuple[str, Optional[str], Optional[str]]:
    endpoint = site_base.rstrip("/") + "/wp-json/wp/v2/pages"
    response = requests.get(
        endpoint,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        params={
            "slug": slug,
            "_fields": "slug,link,title.rendered,content.rendered,modified_gmt",
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or not payload:
        raise RuntimeError("wp-json returned no pages for slug")

    page = payload[0] if isinstance(payload[0], dict) else {}
    content = page.get("content", {}) if isinstance(page.get("content"), dict) else {}
    rendered = content.get("rendered")
    if not isinstance(rendered, str) or not clean_text(rendered):
        raise RuntimeError("wp-json page has no rendered HTML content")

    modified_gmt = page.get("modified_gmt")
    link = page.get("link")
    modified_text = str(modified_gmt).strip() if modified_gmt else None
    link_text = str(link).strip() if link else None
    return rendered, modified_text, link_text


def collect_facility_hours(facility: Dict[str, Any], site_base: str) -> Dict[str, Any]:
    url = str(facility.get("url", "")).strip()
    slug = str(facility.get("slug", "")).strip()

    output: Dict[str, Any] = {
        "facilityId": int(facility.get("facilityId")),
        "facilityName": str(facility.get("facilityName", "")).strip(),
        "slug": slug,
        "url": url,
        "status": "error",
        "source": None,
        "sections": [],
        "error": None,
        "updatedAt": now_iso(),
    }
    errors: List[str] = []

    try:
        html, resolved_url = fetch_direct_html(url)
        sections = parse_hours_sections(html)
        if sections:
            output["status"] = "ok"
            output["source"] = "direct_html"
            output["resolvedUrl"] = resolved_url
            output["sections"] = sections
            return output
        errors.append("direct_html: parsed 0 schedule sections")
    except Exception as exc:
        errors.append("direct_html: " + str(exc))

    try:
        html, modified_gmt, resolved_url = fetch_wp_json_html(site_base=site_base, slug=slug)
        sections = parse_hours_sections(html)
        if sections:
            output["status"] = "ok"
            output["source"] = "wp_json"
            output["resolvedUrl"] = resolved_url or url
            output["sourceModifiedGmt"] = modified_gmt
            output["sections"] = sections
            return output
        errors.append("wp_json: parsed 0 schedule sections")
    except Exception as exc:
        errors.append("wp_json: " + str(exc))

    output["error"] = "; ".join(errors) if errors else "unknown parse error"
    return output


def build_facilities() -> List[Dict[str, Any]]:
    nick_url = env_with_default("RECWELL_NICK_URL", DEFAULT_FACILITIES[0]["url"])
    bakke_url = env_with_default("RECWELL_BAKKE_URL", DEFAULT_FACILITIES[1]["url"])
    return [
        {
            "facilityId": DEFAULT_FACILITIES[0]["facilityId"],
            "facilityName": DEFAULT_FACILITIES[0]["facilityName"],
            "slug": DEFAULT_FACILITIES[0]["slug"],
            "url": nick_url,
        },
        {
            "facilityId": DEFAULT_FACILITIES[1]["facilityId"],
            "facilityName": DEFAULT_FACILITIES[1]["facilityName"],
            "slug": DEFAULT_FACILITIES[1]["slug"],
            "url": bakke_url,
        },
    ]


def write_json(path: str, payload: Dict[str, Any]) -> None:
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def main() -> int:
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
    site_base = env_with_default("RECWELL_SITE_BASE", DEFAULT_SITE_BASE)
    facilities = build_facilities()

    facility_payloads = [collect_facility_hours(facility=row, site_base=site_base) for row in facilities]
    success_count = sum(1 for row in facility_payloads if row.get("status") == "ok")

    payload = {
        "generatedAt": now_iso(),
        "sourceSite": site_base,
        "facilities": facility_payloads,
        "okCount": success_count,
        "totalCount": len(facility_payloads),
    }
    write_json(output_path, payload)

    print(
        "facility_hours_fetch:"
        f" wrote {output_path} | ok={success_count}/{len(facility_payloads)}"
    )
    for row in facility_payloads:
        status = str(row.get("status"))
        name = str(row.get("facilityName"))
        source = str(row.get("source"))
        sections_count = len(row.get("sections", []))
        error = str(row.get("error") or "")
        if status == "ok":
            print(f"  - {name}: ok via {source} ({sections_count} sections)")
        else:
            print(f"  - {name}: error ({error})")

    return 0 if success_count == len(facility_payloads) else 1


if __name__ == "__main__":
    sys.exit(main())
