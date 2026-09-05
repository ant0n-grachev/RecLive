"""Executable compatibility entry point for live-count ingestion."""

import sys

if not __package__:
    import reclive  # noqa: F401 - initialize the canonical parent for direct scripts.

from server.env_loader import load_project_dotenv
from server.reclive.settings import Settings
from server.reclive.ingestion import (
    LIVE_COUNTS_URL as LIVE_COUNTS_URL,
    require_env as require_env,
    require_int_env as require_int_env,
    utc_now as utc_now,
    db_connect as db_connect,
    fetch_live as fetch_live,
    load_facility_capacities as load_facility_capacities,
    failed_result as failed_result,
    finish_ingestion_result as finish_ingestion_result,
    run_ingestion as run_ingestion,
    run_configured_ingestion as run_configured_ingestion,
)


def main() -> int:
    try:
        load_project_dotenv()
        result = run_configured_ingestion(Settings.for_commands())
    except Exception:
        finish_ingestion_result(failed_result("validation"), None, utc_now, print)
        return 1

    return 0 if result.status == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())

sys.modules.setdefault("server.gym_fetch", sys.modules[__name__])
sys.modules.setdefault("gym_fetch", sys.modules[__name__])
setattr(sys.modules["server"], "gym_fetch", sys.modules[__name__])
