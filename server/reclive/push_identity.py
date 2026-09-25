from __future__ import annotations

import hmac
import ipaddress
import os
import re
from hashlib import sha256
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

HASH_KEY_NAME = "PUSH_ENDPOINT_HASH_KEY"
DNS_LABEL = re.compile(r"^[a-z0-9-]+$")
LEGACY_IPV4_COMPONENT = re.compile(r"(?:[0-9]+|0[xX][0-9a-fA-F]+)")
MIGRATION_KEY_ID_DOMAIN = b"reclive:push:migration-key-id:v1"
UNSAFE_DNS_SUFFIXES = (
    ".corp",
    ".example",
    ".example.com",
    ".example.net",
    ".example.org",
    ".home",
    ".home.arpa",
    ".internal",
    ".intranet",
    ".invalid",
    ".lan",
    ".local",
    ".localdomain",
    ".localhost",
    ".private",
    ".test",
)


class PushHashKeyConfigurationError(RuntimeError):
    """A fixed, operator-safe push identity configuration error."""


def configured_push_hash_key() -> bytes:
    key = os.environ.get(HASH_KEY_NAME, "").encode("utf-8")
    if len(key) < 32:
        raise PushHashKeyConfigurationError(
            f"{HASH_KEY_NAME} must be configured with at least 32 bytes"
        )
    return key


def migration_hash_key_identifier() -> bytes:
    return hmac.new(
        configured_push_hash_key(), MIGRATION_KEY_ID_DOMAIN, sha256
    ).digest()


def _is_global_unicast(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return bool(
        address.is_global
        and not address.is_loopback
        and not address.is_private
        and not address.is_link_local
        and not address.is_multicast
        and not address.is_unspecified
        and not address.is_reserved
    )


def normalize_push_endpoint(value: str) -> str:
    text = value.strip()
    if len(text.encode("utf-8")) > 2048:
        raise ValueError("push endpoint is too long")
    if "#" in text:
        raise ValueError("push endpoint must not contain a fragment")

    try:
        parsed = urlsplit(text)
    except ValueError as exc:
        raise ValueError("push endpoint has an invalid authority") from exc
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError("push endpoint must be HTTPS")

    host, port = _canonical_authority(parsed.netloc)
    authority = host if port in (None, 443) else f"{host}:{port}"
    normalized = urlunsplit(
        ("https", authority, parsed.path or "/", parsed.query, "")
    )
    if len(normalized.encode("utf-8")) > 2048:
        raise ValueError("push endpoint is too long")
    return normalized


def _canonical_authority(netloc: str) -> tuple[str, int | None]:
    if "@" in netloc:
        raise ValueError("push endpoint must not contain userinfo")

    port_text: str | None = None
    if netloc.startswith("["):
        closing_bracket = netloc.find("]")
        if closing_bracket < 0:
            raise ValueError("push endpoint has an invalid IP literal")
        raw_host = netloc[1:closing_bracket]
        suffix = netloc[closing_bracket + 1 :]
        if suffix:
            if not suffix.startswith(":") or len(suffix) == 1:
                raise ValueError("push endpoint has an invalid port")
            port_text = suffix[1:]
        if not raw_host or "%" in raw_host:
            raise ValueError("push endpoint has an invalid IP literal")
        try:
            address = ipaddress.ip_address(raw_host)
        except ValueError as exc:
            raise ValueError("push endpoint has an invalid IP literal") from exc
        if not isinstance(address, ipaddress.IPv6Address):
            raise ValueError("push endpoint has an invalid IP literal")
        if not _is_global_unicast(address):
            raise ValueError("push endpoint must use a global IP address")
        host = f"[{address.compressed}]"
    else:
        if "[" in netloc or "]" in netloc or netloc.count(":") > 1:
            raise ValueError("push endpoint has an invalid authority")
        if ":" in netloc:
            raw_host, port_text = netloc.rsplit(":", 1)
            if not port_text:
                raise ValueError("push endpoint has an invalid port")
        else:
            raw_host = netloc
        if not raw_host:
            raise ValueError("push endpoint has an invalid authority")
        try:
            address = ipaddress.ip_address(raw_host)
        except ValueError:
            host = _canonical_dns_name(raw_host)
        else:
            if not isinstance(address, ipaddress.IPv4Address):
                raise ValueError("IPv6 push endpoints must use brackets")
            if not _is_global_unicast(address):
                raise ValueError("push endpoint must use a global IP address")
            host = address.compressed

    port = None
    if port_text is not None:
        if not re.fullmatch(r"[0-9]+", port_text):
            raise ValueError("push endpoint has an invalid port")
        port = int(port_text)
        if not 1 <= port <= 65535:
            raise ValueError("push endpoint has an invalid port")
    return host, port


def _canonical_dns_name(value: str) -> str:
    host = value[:-1] if value.endswith(".") else value
    if not host or host.endswith("."):
        raise ValueError("push endpoint has an invalid DNS name")
    try:
        ascii_host = host.encode("idna").decode("ascii").lower()
        decoded = ascii_host.encode("ascii").decode("idna")
        ascii_host = decoded.encode("idna").decode("ascii").lower()
    except (UnicodeError, UnicodeDecodeError) as exc:
        raise ValueError("push endpoint has an invalid DNS name") from exc
    labels = ascii_host.split(".")
    if len(ascii_host.encode("ascii")) > 253 or any(
        not label
        or len(label.encode("ascii")) > 63
        or not DNS_LABEL.fullmatch(label)
        or label.startswith("-")
        or label.endswith("-")
        for label in labels
    ):
        raise ValueError("push endpoint has an invalid DNS name")
    if all(LEGACY_IPV4_COMPONENT.fullmatch(label) for label in labels):
        raise ValueError("push endpoint has an invalid IP address")
    if len(labels) < 2:
        raise ValueError("push endpoint must use a public DNS name")
    if ascii_host == "localhost" or ascii_host.endswith(".localhost"):
        raise ValueError("push endpoint must not use localhost")
    if any(
        ascii_host == suffix[1:] or ascii_host.endswith(suffix)
        for suffix in UNSAFE_DNS_SUFFIXES
    ):
        raise ValueError("push endpoint must not use a special-use DNS suffix")
    return ascii_host


def _configured_hmac(domain: str, value: str) -> bytes:
    message = f"{domain}\x00{value}".encode("utf-8")
    return hmac.new(configured_push_hash_key(), message, sha256).digest()


def endpoint_hash(endpoint: str) -> bytes:
    return _configured_hmac(
        "reclive:push:endpoint:v1", normalize_push_endpoint(endpoint)
    )


def cancelled_legacy_endpoint_hash(rule_id: int) -> bytes:
    if rule_id < 1:
        raise ValueError("invalid legacy push-rule identity")
    return _configured_hmac(
        "reclive:push:cancelled-legacy:v1", str(rule_id)
    )


def rate_limit_subject_hash(
    subject_kind: Literal["endpoint", "client"], subject: str
) -> bytes:
    if subject_kind == "endpoint":
        subject = normalize_push_endpoint(subject)
    elif subject_kind != "client" or not subject.strip():
        raise ValueError("invalid rate-limit subject")
    return _configured_hmac(
        f"reclive:push:rate-limit:{subject_kind}:v1", subject
    )
