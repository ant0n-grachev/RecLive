from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SAFE_ERROR = "push_rate_limit_prune_failed"
DELETE_EXPIRED_WINDOWS_SQL = "DELETE FROM push_rate_limits WHERE updated_at < %s"


class PushRateLimitPruneError(RuntimeError):
    """A fixed, non-sensitive rate-limit maintenance failure."""

    def __init__(self) -> None:
        super().__init__(SAFE_ERROR)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_runtime_dependencies() -> tuple[Callable[..., Any], object]:
    dependencies: tuple[Callable[..., Any], object] | None = None
    inserted_path: str | None = None
    failed = False
    try:
        runtime_path = str(Path(__file__).resolve().parent)
        if runtime_path not in sys.path:
            inserted_path = runtime_path
            sys.path.insert(0, inserted_path)
        from forecast_api import (  # type: ignore[import-not-found]
            PUSH_WRITE_RATE_WINDOW_SECONDS,
            open_db_connection,
        )
        dependencies = open_db_connection, PUSH_WRITE_RATE_WINDOW_SECONDS
    except BaseException:
        failed = True
    finally:
        if inserted_path is not None:
            try:
                for index, entry in enumerate(sys.path):
                    if entry is inserted_path:
                        del sys.path[index]
                        break
            except BaseException:
                failed = True
    if failed or dependencies is None:
        raise PushRateLimitPruneError() from None
    return dependencies


def _safe_rollback(connection: Any) -> None:
    try:
        connection.rollback()
    except BaseException:
        pass


def prune_push_rate_limits(now: datetime | None = None) -> int:
    connection: Any | None = None
    removed: int | None = None
    failure: PushRateLimitPruneError | None = None
    committed = False

    try:
        sampled_now = _utc_now() if now is None else now
        if not isinstance(sampled_now, datetime):
            raise PushRateLimitPruneError()
        if sampled_now.tzinfo is None or sampled_now.utcoffset() != timedelta(0):
            raise PushRateLimitPruneError()

        open_connection, window_seconds = _load_runtime_dependencies()
        if type(window_seconds) is not int or window_seconds <= 0:
            raise PushRateLimitPruneError()

        cutoff = sampled_now - timedelta(seconds=window_seconds * 2)
        mysql_cutoff = cutoff.replace(tzinfo=None)
        connection = open_connection(autocommit=False)
        try:
            with connection.cursor() as cursor:
                cursor.execute(DELETE_EXPIRED_WINDOWS_SQL, (mysql_cutoff,))
                candidate_rowcount = cursor.rowcount
            if type(candidate_rowcount) is not int or candidate_rowcount < 0:
                raise PushRateLimitPruneError()
            removed = candidate_rowcount
            connection.commit()
            committed = True
        except BaseException:
            if not committed:
                _safe_rollback(connection)
            raise
    except BaseException:
        failure = PushRateLimitPruneError()
    finally:
        if connection is not None:
            try:
                connection.close()
            except BaseException:
                if failure is None:
                    failure = PushRateLimitPruneError()

    if failure is not None:
        raise failure from None
    if type(removed) is not int:
        raise PushRateLimitPruneError() from None
    return removed


def main() -> int:
    try:
        removed = prune_push_rate_limits()
        print(f"pruned_push_rate_limit_windows={removed}")
    except BaseException:
        try:
            print(SAFE_ERROR, file=sys.stderr)
        except BaseException:
            pass
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
