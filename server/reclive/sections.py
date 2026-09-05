from __future__ import annotations

import sys as _import_sys
import json
from typing import Any, Dict, List, Optional, Tuple
from server.forecast_shared import normalize_section_key
from server.facility_capacities import load_facility_capacities
from server.reclive.runtime import current_runtime


def load_facility_sections() -> Tuple[Dict[int, str], Dict[int, Dict[str, List[int]]]]:
    try:
        with open(
            current_runtime().settings.facility_section_config_path,
            "r",
            encoding="utf-8",
        ) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read facility section config at {current_runtime().settings.facility_section_config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Facility section config is not valid JSON: {current_runtime().settings.facility_section_config_path}"
        ) from exc
    facilities_raw = payload.get("facilities")
    if not isinstance(facilities_raw, dict):
        raise RuntimeError(
            "Facility section config must contain an object at key 'facilities'"
        )
    facility_names: Dict[int, str] = {}
    section_ids: Dict[int, Dict[str, List[int]]] = {}
    for facility_id_text, facility_payload in facilities_raw.items():
        if not isinstance(facility_payload, dict):
            continue
        try:
            facility_id = int(facility_id_text)
        except (TypeError, ValueError):
            continue
        short_name = str(facility_payload.get("shortName", "")).strip()
        if not short_name:
            short_name = str(facility_payload.get("facilityName", "")).strip()
        if not short_name:
            short_name = f"Facility {facility_id}"
        facility_names[facility_id] = short_name
        sections_raw = facility_payload.get("sections")
        if not isinstance(sections_raw, list):
            sections_raw = []
        by_section: Dict[str, List[int]] = {}
        overall: List[int] = []
        overall_seen = set()
        for section in sections_raw:
            if not isinstance(section, dict):
                continue
            key = normalize_section_key(str(section.get("key", "")))
            ids_raw = section.get("ids")
            if not key or not isinstance(ids_raw, list):
                continue
            location_ids: List[int] = []
            for location_id_raw in ids_raw:
                try:
                    location_id = int(location_id_raw)
                except (TypeError, ValueError):
                    continue
                location_ids.append(location_id)
                if location_id not in overall_seen:
                    overall_seen.add(location_id)
                    overall.append(location_id)
            if location_ids:
                by_section[key] = location_ids
        by_section["overall"] = overall
        section_ids[facility_id] = by_section
    if not facility_names or not section_ids:
        raise RuntimeError("Facility section config did not produce any facilities")
    return (facility_names, section_ids)


def load_runtime_facility_configuration() -> None:
    runtime = current_runtime()
    capacities = (
        load_facility_capacities()
        if runtime.legacy
        else load_facility_capacities(runtime.settings.capacity_config_path)
    )
    facility_names, section_ids = load_facility_sections()
    current_runtime().capacities = capacities
    current_runtime().facility_names = facility_names
    current_runtime().section_ids = section_ids
    current_runtime().configuration_loaded = True


def ensure_runtime_facility_configuration() -> None:
    if current_runtime().configuration_loaded:
        return
    with current_runtime().configuration_lock:
        if current_runtime().configuration_loaded:
            return
        load_runtime_facility_configuration()


def category_location_ids_for_forecast(
    facility_id: int, category: Dict[str, Any]
) -> List[int]:
    section_map = current_runtime().section_ids.get(facility_id, {})
    key_raw = _str_or_none(category.get("key"))
    title_raw = _str_or_none(category.get("title"))
    candidates: List[str] = []
    if key_raw:
        candidates.append(normalize_section_key(key_raw))
        candidates.append(normalize_section_key(key_raw.replace("_", " ")))
    if title_raw:
        candidates.append(normalize_section_key(title_raw))
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ids = section_map.get(candidate)
        if not isinstance(ids, list):
            continue
        output: List[int] = []
        for raw_id in ids:
            try:
                output.append(int(raw_id))
            except (TypeError, ValueError):
                continue
        if output:
            return output
    return []


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _str_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def canonical_section_key(value: str) -> str:
    key = normalize_section_key(value)
    if key in {
        "overall",
        "entire facility",
        "facility",
        "all",
        "all sections",
        "whole gym",
    }:
        return "overall"
    return key


def location_ids_for_section(
    facility_id: int, section_key: str, *, section_ids=None
) -> List[int]:
    maps = current_runtime().section_ids if section_ids is None else section_ids
    section_map = maps.get(facility_id, {})
    normalized_key = canonical_section_key(section_key)
    if normalized_key == "overall":
        output: List[int] = []
        seen = set()
        for ids in section_map.values():
            if not isinstance(ids, list):
                continue
            for raw_id in ids:
                try:
                    location_id = int(raw_id)
                except (TypeError, ValueError):
                    continue
                if location_id in seen:
                    continue
                seen.add(location_id)
                output.append(location_id)
        return output
    ids = section_map.get(normalized_key)
    if not isinstance(ids, list):
        return []
    output = []
    for raw_id in ids:
        try:
            output.append(int(raw_id))
        except (TypeError, ValueError):
            continue
    return output


_import_sys.modules.setdefault("server.reclive.sections", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.sections", _import_sys.modules[__name__])
