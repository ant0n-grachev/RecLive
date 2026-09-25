"""Connection construction without connection or environment I/O at import."""

import sys as _import_sys

from typing import Any
import pymysql

from .settings import DatabaseSettings


def safe_sql_identifier(value: str, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"{name} must not be empty")
    if not (text[0].isalpha() or text[0] == "_"):
        raise RuntimeError(f"{name} must start with a letter or underscore")
    for ch in text:
        if not (ch.isalnum() or ch == "_"):
            raise RuntimeError(f"{name} contains invalid characters")
    return text


def open_db_connection(
    database: DatabaseSettings | None = None, *, autocommit: bool = True
) -> Any:
    if database is None:
        from .settings import require_env

        database = DatabaseSettings(
            host=require_env("GYM_DB_HOST"),
            port=require_env("GYM_DB_PORT"),
            user=require_env("GYM_DB_USER"),
            password=require_env("GYM_DB_PASSWORD"),
            name=require_env("GYM_DB_NAME"),
        )
    try:
        port = int(database.port)
    except (TypeError, ValueError, OverflowError):
        raise RuntimeError("Invalid integer for env var GYM_DB_PORT") from None
    return pymysql.connect(
        host=database.host,
        port=port,
        user=database.user,
        password=database.password,
        database=database.name,
        autocommit=autocommit,
        charset=database.charset,
        connect_timeout=database.connect_timeout,
        read_timeout=database.read_timeout,
        write_timeout=database.write_timeout,
    )


_import_sys.modules.setdefault("server.reclive.db", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.db", _import_sys.modules[__name__])
