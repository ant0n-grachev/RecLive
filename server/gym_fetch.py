import os
import sys
import requests
import pymysql
from datetime import datetime
import pytz
from env_loader import load_project_dotenv

SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
load_project_dotenv()

def require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required env var: {name}")

    normalized = value.strip()
    if not normalized:
        raise RuntimeError(f"Missing required env var: {name}")
    return normalized


def require_int_env(name: str) -> int:
    raw = require_env(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid integer for env var {name}: {raw}") from exc


LIVE_COUNTS_URL = require_env("LIVE_COUNTS_URL")
TZ_NAME = "America/Chicago"
TZ = pytz.timezone(TZ_NAME)

# location_id -> max_capacity
MAX_CAP = {
    # Nick
    5761: 140, 5764: 230, 5760: 150, 7089: 24, 5762: 100,
    5758: 200, 7090: 48, 5766: 24, 5753: 6, 5754: 6, 5763: 100,
    # Bakke
    8718: 30, 8717: 130, 8720: 24, 8714: 24, 8716: 116, 10550: 200,
    8705: 65, 8708: 27, 8712: 12, 8700: 246, 8698: 48, 8701: 39,
    8699: 75, 8696: 46, 8694: 100, 8695: 18,
}

LATEST_SQL = """
SELECT h.location_id, h.last_updated
FROM location_history h
JOIN (
    SELECT location_id, MAX(id) AS max_id
    FROM location_history
    GROUP BY location_id
) x ON x.location_id = h.location_id AND x.max_id = h.id;
"""

INSERT_SQL = """
INSERT INTO location_history
(
    location_id,
    is_closed,
    current_capacity,
    max_capacity,
    last_updated,
    fetched_at
)
VALUES (%s, %s, %s, %s, %s, %s);
"""


def chicago_now_str(milliseconds: bool = False) -> str:
    pattern = "%Y-%m-%d %H:%M:%S.%f" if milliseconds else "%Y-%m-%d %H:%M:%S"
    value = datetime.now(TZ).strftime(pattern)
    return value[:-3] if milliseconds else value

def db_connect():
    host = require_env("GYM_DB_HOST")
    port = require_int_env("GYM_DB_PORT")
    user = require_env("GYM_DB_USER")
    password = require_env("GYM_DB_PASSWORD")
    database = require_env("GYM_DB_NAME")

    return pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        autocommit=True,
        charset="utf8mb4",
        connect_timeout=10,
        read_timeout=20,
        write_timeout=20,
    )

def fetch_live():
    r = requests.get(LIVE_COUNTS_URL, timeout=20)
    r.raise_for_status()
    payload = r.json()
    if not isinstance(payload, list):
        raise RuntimeError("Live feed payload is not a list")
    return payload

def get_latest_map(conn):
    with conn.cursor() as cur:
        cur.execute(LATEST_SQL)
        rows = cur.fetchall()
    # normalize None -> "" so comparisons are consistent
    return {int(loc_id): (last_upd or "") for (loc_id, last_upd) in rows}

def insert_if_changed(conn, live):
    latest = get_latest_map(conn)
    fetched_at = chicago_now_str(milliseconds=True)

    to_insert = []
    skipped = 0

    for f in live:
        if not isinstance(f, dict):
            continue

        loc_id = f.get("LocationId")
        if loc_id is None:
            continue

        try:
            loc_id = int(loc_id)
        except (TypeError, ValueError):
            continue

        last_updated_raw = f.get("LastUpdatedDateAndTime")
        last_updated = str(last_updated_raw).strip() if last_updated_raw is not None else ""

        if latest.get(loc_id, "") == last_updated:
            skipped += 1
            continue

        to_insert.append((
            loc_id,
            f.get("IsClosed"),
            f.get("LastCount"),
            MAX_CAP.get(loc_id),
            last_updated,
            fetched_at,
        ))

    if to_insert:
        with conn.cursor() as cur:
            cur.executemany(INSERT_SQL, to_insert)

    return len(to_insert), skipped

def main():
    conn = db_connect()
    try:
        live = fetch_live()
        inserted, skipped = insert_if_changed(conn, live)
        print(
            f"{chicago_now_str()} OK: fetched {len(live)} | inserted {inserted} | skipped {skipped}"
        )
        return 0
    except Exception as e:
        print(chicago_now_str(), "ERROR:", e)
        return 1
    finally:
        conn.close()

if __name__ == "__main__":
    sys.exit(main())
