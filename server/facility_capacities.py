import json
import os
from typing import Dict


SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
DEFAULT_CAPACITY_PATH = os.path.join(PROJECT_ROOT, "shared", "facility_capacities.json")


def _resolve_path(raw_path: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path

    project_candidate = os.path.abspath(os.path.join(PROJECT_ROOT, raw_path))
    if os.path.exists(project_candidate):
        return project_candidate

    return os.path.abspath(os.path.join(SCRIPT_DIR, raw_path))


def load_facility_capacities() -> Dict[int, int]:
    raw_path = os.getenv("FACILITY_CAPACITIES_JSON_PATH", DEFAULT_CAPACITY_PATH)
    path = _resolve_path(raw_path)

    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise RuntimeError(f"Failed to read facility capacities at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Facility capacities file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Facility capacities payload must be a JSON object")

    capacities: Dict[int, int] = {}
    for location_id_raw, capacity_raw in payload.items():
        try:
            location_id = int(location_id_raw)
            capacity = int(capacity_raw)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid facility capacity entry: {location_id_raw}") from exc

        if location_id <= 0 or capacity < 0:
            raise RuntimeError(f"Invalid facility capacity entry: {location_id_raw}")
        capacities[location_id] = capacity

    if not capacities:
        raise RuntimeError("Facility capacities payload is empty")

    return capacities
