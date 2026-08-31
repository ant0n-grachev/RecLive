from __future__ import annotations

import argparse
import sys
from pathlib import Path

from reclive.migrations import MigrationError, MigrationSettings, run_migrations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply ordered RecLive MySQL migrations"
    )
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=Path(__file__).with_name("migrations"),
    )
    args = parser.parse_args()
    try:
        applied = run_migrations(
            MigrationSettings.from_environment(), args.migrations_dir
        )
    except MigrationError as exc:
        print(f"migration failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("migration failed: unexpected migration error", file=sys.stderr)
        return 1
    print(f"migration complete: {len(applied)} applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
