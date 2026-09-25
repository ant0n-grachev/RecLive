"""Dialect detection and exact frozen-migration execution adaptations."""
from __future__ import annotations

import re
import sys
from enum import Enum
from types import MappingProxyType


# Exact frozen-hook statements only: retain the coordinated cutover's lock.
MARIADB_HOOK_REWRITES = MappingProxyType({
    f"ALTER TABLE `{table}` MODIFY endpoint_hash BINARY(32) NOT NULL, "
    "ALGORITHM=INPLACE, LOCK=EXCLUSIVE":
    f"ALTER TABLE `{table}` MODIFY endpoint_hash BINARY(32) NOT NULL, "
    "ALGORITHM=COPY, LOCK=EXCLUSIVE"
    for table in ("push_rules", "_reclive_push_rules_cutover")
})


class UnsupportedDatabaseError(RuntimeError):
    """The server is outside the database versions supported by migrations."""


class DatabaseDialect(Enum):
    MYSQL8 = "mysql8"
    MARIADB1011 = "mariadb10.11"

    @property
    def collation(self) -> str:
        return (
            "utf8mb4_unicode_ci"
            if self is DatabaseDialect.MARIADB1011
            else "utf8mb4_0900_ai_ci"
        )


def detect_database_dialect(connection) -> DatabaseDialect:
    with connection.cursor() as cursor:
        cursor.execute("SELECT VERSION(), @@version_comment")
        identity = cursor.fetchone()
    if (
        isinstance(identity, (tuple, list))
        and len(identity) == 2
        and all(isinstance(value, str) for value in identity)
    ):
        version, comment = identity
        if re.fullmatch(
            r"(?:5\.5\.5-)?10\.11\.\d+-MariaDB(?:-[A-Za-z0-9.+~_-]+)?", version
        ):
            return DatabaseDialect.MARIADB1011
        if (
            re.fullmatch(r"8\.(?:0|4)\.\d+(?:-commercial)?", version)
            and comment in {
                "MySQL Community Server - GPL",
                "MySQL Enterprise Server - Commercial",
                "Homebrew",
            }
        ):
            return DatabaseDialect.MYSQL8
    # Server identity may contain private host/build details; never echo it.
    raise UnsupportedDatabaseError(
        "Unsupported database; expected MySQL 8.0/8.4 or MariaDB 10.11"
    )


class _MariaDBHookCursor:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        self._cursor.__enter__()
        return self

    def __exit__(self, *args):
        return self._cursor.__exit__(*args)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def execute(self, statement, params=None):
        return self._cursor.execute(MARIADB_HOOK_REWRITES.get(statement, statement), params)


class _MariaDBHookConnection:
    def __init__(self, connection) -> None:
        self._connection = connection

    def __getattr__(self, name):
        return getattr(self._connection, name)

    def cursor(self):
        return _MariaDBHookCursor(self._connection.cursor())


def migration_hook_connection(connection, dialect: DatabaseDialect):
    if dialect is DatabaseDialect.MARIADB1011:
        return _MariaDBHookConnection(connection)
    return connection


sys.modules.setdefault("server.reclive.database_dialect", sys.modules[__name__])
sys.modules.setdefault("reclive.database_dialect", sys.modules[__name__])
