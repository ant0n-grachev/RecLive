"""Immutable API configuration; importing this module performs no initialization."""

import sys as _import_sys
from dataclasses import dataclass, field
import os
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
from typing import Dict, List, Optional, Sequence
from fastapi import HTTPException
from server.reclive.push_identity import endpoint_hash

SERVER_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatabaseSettings:
    host: str | None = None
    port: str | int | None = None
    user: str | None = None
    password: str | None = field(default=None, repr=False)
    name: str | None = None
    timezone: str = "UTC"
    charset: str = "utf8mb4"
    connect_timeout: int = 10
    read_timeout: int = 20
    write_timeout: int = 20


@dataclass(frozen=True)
class PushSettings:
    table: str = "push_rules"
    body_max_bytes: int = 16 * 1024
    default_rule_ttl_seconds: int = 86400
    max_rule_ttl_seconds: int = 604800
    max_active_rules_per_endpoint: int = 10
    write_rate_limit: int = 20
    write_rate_window_seconds: int = 600
    evaluator_interval_seconds: int = 180
    evaluator_lock_name: str = "reclive_push_eval"
    evaluator_enabled: bool = True
    admin_routes_enabled: bool = False
    vapid_public_key: str = ""
    vapid_private_key: str = field(default="", repr=False)
    vapid_subject: str = ""
    admin_token: str = field(default="", repr=False)


@dataclass(frozen=True)
class Settings:
    forecast_json_path: str = str(SERVER_ROOT / "forecast.json")
    facility_hours_json_path: str = str(SERVER_ROOT / "facility_hours.json")
    facility_section_config_path: str = str(SERVER_ROOT / "facility_sections.json")
    capacity_config_path: str = str(
        SERVER_ROOT.parent / "shared/facility_capacities.json"
    )
    actual_hour_min_coverage: float = 0.75
    schedule_stale_after_seconds: int = 21600
    cors_origins: tuple[str, ...] = ("*",)
    host: str = "0.0.0.0"
    port: int = 8000
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    push: PushSettings = field(default_factory=PushSettings)
    environment: str = "development"
    environment_values: Mapping[str, str] = field(default_factory=dict, repr=False)
    capacities: Mapping[int, int] | None = None
    facility_names: Mapping[int, str] | None = None
    section_ids: Mapping[int, Mapping[str, tuple[int, ...]]] | None = None

    def __post_init__(self):
        object.__setattr__(self, "cors_origins", tuple(self.cors_origins))
        object.__setattr__(
            self, "environment_values", MappingProxyType(dict(self.environment_values))
        )
        for name in ("capacities", "facility_names"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        if self.section_ids is not None:
            object.__setattr__(
                self,
                "section_ids",
                MappingProxyType(
                    {
                        facility: MappingProxyType(
                            {key: tuple(ids) for key, ids in sections.items()}
                        )
                        for facility, sections in self.section_ids.items()
                    }
                ),
            )

    @classmethod
    def from_environment(cls, *, legacy=False):
        return build_settings_from_environment(os.environ, legacy=legacy)

    @classmethod
    def for_test(cls, *, forecast_json_path: str | None = None):
        return cls(
            forecast_json_path=forecast_json_path or str(SERVER_ROOT / "forecast.json"),
            environment="test",
            environment_values={"APP_ENV": "test"},
            database=DatabaseSettings(
                host="127.0.0.1",
                port=3306,
                user="reclive",
                password="reclive-ci-password",
                name="reclive_test",
            ),
            push=PushSettings(evaluator_enabled=False),
            capacities={},
            facility_names={},
            section_ids={},
        )


def build_settings_from_environment(
    environment: Mapping[str, str], *, legacy=False
) -> Settings:
    values = dict(environment)

    def read(name, default=None):
        raw = values.get(name)
        return str(raw).strip() if raw is not None and str(raw).strip() else default

    def integer(name, default):
        raw = read(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            raise RuntimeError(f"Invalid integer for env var {name}") from None

    def boolean(name, default):
        raw = read(name)
        if raw is None:
            return default
        if raw.lower() in {"1", "true", "yes", "on"}:
            return True
        if raw.lower() in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(f"Invalid boolean for env var {name}") from None

    def path(name, default):
        raw = read(name, str(default))
        return os.path.abspath(os.path.join(SERVER_ROOT, raw))

    forecast_path = path("FORECAST_JSON_PATH", SERVER_ROOT / "forecast.json")
    host = read("FORECAST_API_HOST", "0.0.0.0")
    port = integer("FORECAST_API_PORT", 8000)
    db_timezone = read("GYM_DB_TIMEZONE", "UTC")
    try:
        coverage = float(read("ACTUAL_HOUR_MIN_COVERAGE", "0.75"))
    except ValueError:
        raise RuntimeError(
            "Invalid number for env var ACTUAL_HOUR_MIN_COVERAGE"
        ) from None
    section_path = path(
        "FACILITY_SECTION_CONFIG_PATH", SERVER_ROOT / "facility_sections.json"
    )
    schedule_path = path(
        "FACILITY_HOURS_JSON_PATH", SERVER_ROOT / "facility_hours.json"
    )
    stale_name = (
        "SCHEDULE_STALE_AFTER_SECONDS"
        if read("SCHEDULE_STALE_AFTER_SECONDS") is not None
        else "SCHEDULE_MAX_AGE_SECONDS"
    )
    try:
        stale = int(read(stale_name, "21600"))
    except ValueError:
        raise RuntimeError(
            f"Invalid positive integer for env var {stale_name}"
        ) from None
    if stale <= 0:
        raise RuntimeError(f"Invalid positive integer for env var {stale_name}")
    push = PushSettings(
        table=read("PUSH_RULES_TABLE", "push_rules"),
        default_rule_ttl_seconds=integer("PUSH_RULE_DEFAULT_TTL_SECONDS", 86400),
        max_rule_ttl_seconds=integer("PUSH_RULE_MAX_TTL_SECONDS", 604800),
        max_active_rules_per_endpoint=integer("PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT", 10),
        write_rate_limit=integer("PUSH_WRITE_RATE_LIMIT", 20),
        write_rate_window_seconds=integer("PUSH_WRITE_RATE_WINDOW_SECONDS", 600),
        evaluator_interval_seconds=integer("PUSH_EVALUATOR_INTERVAL_SECONDS", 180),
        evaluator_lock_name=read("PUSH_EVALUATOR_DB_LOCK_NAME", "reclive_push_eval"),
        vapid_public_key=read("PUSH_VAPID_PUBLIC_KEY", ""),
        vapid_private_key=read("PUSH_VAPID_PRIVATE_KEY", ""),
        vapid_subject=read("PUSH_VAPID_SUBJECT", ""),
        admin_token=read("PUSH_ADMIN_TOKEN", ""),
        evaluator_enabled=True if legacy else boolean("PUSH_EVALUATOR_ENABLED", True),
        admin_routes_enabled=False
        if legacy
        else boolean("PUSH_ADMIN_ROUTES_ENABLED", False),
    )
    raw_capacity = read(
        "FACILITY_CAPACITIES_JSON_PATH",
        str(SERVER_ROOT.parent / "shared/facility_capacities.json"),
    )
    capacity_path = raw_capacity
    if not os.path.isabs(raw_capacity):
        project_candidate = str(SERVER_ROOT.parent / raw_capacity)
        capacity_path = (
            project_candidate
            if os.path.exists(project_candidate)
            else str(SERVER_ROOT / raw_capacity)
        )
    origins = tuple(
        (
            item.strip()
            for item in read("FORECAST_API_ALLOW_ORIGINS", "*").split(",")
            if item.strip()
        )
    ) or ("*",)
    return Settings(
        forecast_json_path=forecast_path,
        host=host,
        port=port,
        facility_hours_json_path=schedule_path,
        facility_section_config_path=section_path,
        capacity_config_path=capacity_path,
        actual_hour_min_coverage=coverage,
        schedule_stale_after_seconds=stale,
        cors_origins=origins,
        database=DatabaseSettings(
            host=read("GYM_DB_HOST"),
            port=read("GYM_DB_PORT"),
            user=read("GYM_DB_USER"),
            password=read("GYM_DB_PASSWORD"),
            name=read("GYM_DB_NAME"),
            timezone=db_timezone,
        ),
        push=push,
        environment=read("APP_ENV", "development").lower(),
        environment_values=values,
    )


APP_ENVIRONMENTS = frozenset({"development", "test", "production"})
PUSH_ADMIN_TOKEN_MIN_BYTES = 32
PUSH_ADMIN_TOKEN_MAX_BYTES = 512


def current_runtime():
    from server.reclive.runtime import current_runtime as get_runtime

    return get_runtime()


def _read_env(name: str, aliases: Sequence[str] = ()) -> Optional[str]:
    try:
        runtime = current_runtime()
    except RuntimeError:
        runtime = None
    values = (
        os.environ
        if runtime is None or runtime.legacy
        else runtime.settings.environment_values
    )
    for key in (name, *aliases):
        value = values.get(key)
        if value is None:
            continue
        normalized = value.strip()
        if normalized:
            return normalized
    return None


def require_env(name: str, aliases: Sequence[str] = ()) -> str:
    value = _read_env(name, aliases=aliases)
    if value is None:
        alias_text = f" (aliases: {', '.join(aliases)})" if aliases else ""
        raise RuntimeError(f"Missing required env var: {name}{alias_text}")
    return value


def env_with_default(name: str, default: str, aliases: Sequence[str] = ()) -> str:
    value = _read_env(name, aliases=aliases)
    if value is None:
        return default
    return value


def int_with_default(name: str, default: int, aliases: Sequence[str] = ()) -> int:
    raw = _read_env(name, aliases=aliases)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Invalid integer for env var {name}") from None


def positive_int_with_legacy_alias(name: str, legacy_name: str, default: int) -> int:
    selected_name = name
    raw = _read_env(name)
    if raw is None:
        selected_name = legacy_name
        raw = _read_env(legacy_name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            raise RuntimeError(
                f"Invalid positive integer for env var {selected_name}"
            ) from None
    if value <= 0:
        raise RuntimeError(f"Invalid positive integer for env var {selected_name}")
    return value


def bool_with_default(name: str, default: bool, aliases: Sequence[str] = ()) -> bool:
    raw = _read_env(name, aliases=aliases)
    if raw is None:
        return default
    value = raw.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"Invalid boolean for env var {name}") from None


def path_with_default(name: str, default: str, aliases: Sequence[str] = ()) -> str:
    raw = env_with_default(name, default, aliases=aliases)
    return resolve_path(raw)


def resolve_path(raw: str) -> str:
    if os.path.isabs(raw):
        return raw
    script_candidate = os.path.abspath(os.path.join(str(SERVER_ROOT), raw))
    return script_candidate


def app_environment() -> str:
    runtime = current_runtime()
    value = (
        env_with_default("APP_ENV", "development").lower()
        if runtime.legacy
        else runtime.settings.environment
    )
    if value not in APP_ENVIRONMENTS:
        raise RuntimeError("APP_ENV must be development, test, or production")
    return value


def push_admin_routes_enabled() -> bool:
    runtime = current_runtime()
    return (
        bool_with_default("PUSH_ADMIN_ROUTES_ENABLED", False)
        if runtime.legacy
        else runtime.settings.push.admin_routes_enabled
    )


def _validated_admin_token_bytes() -> bytes:
    try:
        token = current_runtime().settings.push.admin_token.encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(
            "PUSH_ADMIN_TOKEN must contain only ASCII characters"
        ) from None
    if len(token) < PUSH_ADMIN_TOKEN_MIN_BYTES:
        raise RuntimeError(
            "PUSH_ADMIN_TOKEN must be at least 32 bytes when admin routes are enabled"
        )
    if len(token) > PUSH_ADMIN_TOKEN_MAX_BYTES:
        raise RuntimeError(
            "PUSH_ADMIN_TOKEN must not exceed 512 bytes when admin routes are enabled"
        )
    return token


def validate_push_configuration() -> None:
    if current_runtime().settings.push.default_rule_ttl_seconds <= 0:
        raise RuntimeError("Push default rule TTL must be positive")
    if current_runtime().settings.push.max_rule_ttl_seconds <= 0:
        raise RuntimeError("Push maximum rule TTL must be positive")
    if current_runtime().settings.push.max_active_rules_per_endpoint <= 0:
        raise RuntimeError("Push active-rule maximum must be positive")
    if current_runtime().settings.push.write_rate_limit <= 0:
        raise RuntimeError("Push write rate limit must be positive")
    if current_runtime().settings.push.write_rate_window_seconds <= 0:
        raise RuntimeError("Push rate-limit window must be positive")
    if (
        current_runtime().settings.push.default_rule_ttl_seconds
        > current_runtime().settings.push.max_rule_ttl_seconds
    ):
        raise RuntimeError("Push default rule TTL must not exceed the maximum rule TTL")
    if current_runtime().settings.push.max_rule_ttl_seconds > 604800:
        raise RuntimeError("Push maximum rule TTL must not exceed 604800 seconds")
    if current_runtime().settings.push.max_active_rules_per_endpoint > 10:
        raise RuntimeError("Push active-rule maximum must not exceed 10")
    if current_runtime().settings.push.write_rate_limit > 20:
        raise RuntimeError("Push write rate limit must not exceed 20")
    environment = app_environment()
    if environment == "production":
        endpoint_hash("https://push.reclive.app/startup-check")
    if push_admin_routes_enabled():
        _validated_admin_token_bytes()


def parse_allowed_origins() -> List[str]:
    raw = env_with_default("FORECAST_API_ALLOW_ORIGINS", "*")
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    if not parsed:
        return ["*"]
    return parsed


def evaluator_enabled() -> bool:
    runtime = current_runtime()
    enabled = (
        bool_with_default("PUSH_EVALUATOR_ENABLED", True)
        if runtime.legacy
        else runtime.settings.push.evaluator_enabled
    )
    return enabled and push_vapid_configured()


def push_vapid_configured() -> bool:
    return bool(
        current_runtime().settings.push.vapid_public_key
        and current_runtime().settings.push.vapid_private_key
        and current_runtime().settings.push.vapid_subject
    )


def push_admin_configured() -> bool:
    try:
        _validated_admin_token_bytes()
    except RuntimeError:
        return False
    return True


def get_vapid_public_key() -> str:
    if not current_runtime().settings.push.vapid_public_key:
        raise HTTPException(
            status_code=503, detail="Push VAPID public key is not configured"
        )
    return current_runtime().settings.push.vapid_public_key


def get_vapid_private_key() -> str:
    if not current_runtime().settings.push.vapid_private_key:
        raise HTTPException(
            status_code=503, detail="Push VAPID private key is not configured"
        )
    return current_runtime().settings.push.vapid_private_key


def get_vapid_claims() -> Dict[str, str]:
    if not current_runtime().settings.push.vapid_subject:
        raise HTTPException(
            status_code=503, detail="Push VAPID subject is not configured"
        )
    return {"sub": current_runtime().settings.push.vapid_subject}


_import_sys.modules.setdefault("server.reclive.settings", _import_sys.modules[__name__])
_import_sys.modules.setdefault("reclive.settings", _import_sys.modules[__name__])
