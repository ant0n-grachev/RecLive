# RecLive Push-Alert Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make RecLive Web Push rules private, durable, manageable, rate-limited, expiring, and at-most-once across concurrent FastAPI workers.

**Architecture:** Keep Phase 5 in the existing FastAPI compatibility entry point, but give it explicit validation, persistence, and evaluator interfaces that can be tested with injected MySQL and Web Push seams. MySQL is the source of truth for rules and fixed-window counters; the browser supplies its current PushSubscription to prove ownership, while localStorage retains only non-authoritative form defaults. The evaluator takes the MySQL advisory lock before it reads any rule, schedule, ingestion, or snapshot state, then uses a conditional status transition to claim a qualifying rule before sending.

**Tech Stack:** Python 3, FastAPI, Pydantic v2, PyMySQL/MySQL 8.4, `pywebpush`, React 19, TypeScript, Material UI, Vitest, React Testing Library, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md` (Phase 5), with every Phase 5 requirement from the original brief reproduced below.

## Global Constraints

- Implement Phase 5 only; Phase 1 migrations and backend/frontend test foundations are prerequisites, not work to repeat here.
- The master plan's phase-level commit policy is authoritative: task commit snippets describe staging/review scope only; make the single Phase 5 source commit after Task 8.
- Preserve facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, existing product identity, forecasting, Web Push, installation behavior, and the executable `server/forecast_api.py` entry point.
- Public write endpoints reject a request body larger than 16 KiB before JSON validation. A non-integer, negative, or above-limit `Content-Length` is rejected safely before body consumption; unheadered bodies are accumulated from `request.stream()` only through the first over-limit chunk, never through FastAPI's cached whole-body helper.
- A subscription endpoint must be HTTPS and no longer than 2,048 characters; `keys.p256dh` must base64url-decode to a 65-byte uncompressed P-256 key beginning with `0x04`; `keys.auth` must base64url-decode to exactly 16 bytes.
- The shared Phase 1 endpoint normalizer rejects localhost/local-only names, single-label hosts, and every non-global or special-use IP literal. Immediately before each send, Phase 5 resolves the canonical hostname, rejects an empty, non-global, or mixed public/non-public answer set, pins one validated public address for the TLS connection while preserving the original hostname for SNI, certificate verification, and the `Host` header, and rejects every redirect. A resolve-then-unpinned `pywebpush` call is forbidden.
- Thresholds are integer percentages from 1 through 100 inclusive. The default TTL is 86,400 seconds (24 hours); the maximum accepted TTL is 604,800 seconds (seven days); one endpoint may have at most ten active rules.
- Use the Phase 1 shared `server/reclive/push_identity.py` `normalize_push_endpoint` and `endpoint_hash` helpers for endpoint validation, persisted identity, ownership comparison, and legacy backfill lookup; use `rate_limit_subject_hash` only for fixed-window subjects. Store HMAC-SHA-256 results in the Phase 1 `BINARY(32)` identity fields, never an indexed raw endpoint or address. Store the PushSubscription JSON needed by `pywebpush`; never return it to the browser and never log a raw endpoint, key, subscription body, client address, credential, VAPID key, or environment value.
- `PUSH_ENDPOINT_HASH_KEY` is a dedicated secret and production startup rejects it when missing or shorter than 32 bytes. Admin authorization uses `hmac.compare_digest`, and production startup rejects an enabled admin route with a missing or shorter-than-32-byte admin token.
- Public write rate limiting is durable MySQL fixed-window limiting: 20 requests per HMAC subject per 600-second window. Use the Phase 1 `BINARY(32)` rate-limit hash field with a normalized endpoint subject when one is valid; otherwise use an HMAC of the immediate request client address without storing the raw address.
- Rules are pending, claimed, sent, failed, invalid_subscription, cancelled, or expired. A claimed rule is never reclaimed; at-most-once delivery is preferred to duplicate delivery.
- Before dispatch, require an official schedule that says the facility is open, fresh successful ingestion within the existing centralized 600-second freshness threshold, fresh relevant snapshots, valid section membership, at least 0.80 coverage, an unexpired pending rule, and a percentage at or below its threshold.
- `404` and `410` from Web Push finalize the rule as `invalid_subscription`; other send failures finalize it as `failed`; successful delivery finalizes it as `sent`. Notification URLs are exactly `/nick` for `1186` and `/bakke` for `1656`.
- Do not merge, deploy, alter production credentials, contact providers, print historical credentials, or commit source changes while planning.

---

## File Structure

- Modify: `server/forecast_api.py` — retain the public FastAPI entry point; add bounded push-body parsing, cryptographic validation, endpoint identity, repository helpers, management routes, evaluator gates, and safe terminal-state changes.
- Consume: `server/migrations/0003_push_rule_lifecycle.sql` — the applied Phase 1 push-rule contract containing `BINARY(32)` `endpoint_hash`, generated nullable `active_identity`, `uq_push_rules_identity`, lifecycle timestamps, `finalized_at`, and the `cancelled` state. This plan must not edit this applied migration.
- Consume: `server/migrations/0004_rate_limits.sql` — the applied Phase 1 durable counter contract containing a `BINARY(32)` subject hash and its pruning index.
- Consume: `server/reclive/push_identity.py` — Phase 1's canonical `normalize_push_endpoint(value: str) -> str`, `endpoint_hash(endpoint: str) -> bytes`, and `rate_limit_subject_hash(subject_kind: Literal["endpoint", "client"], subject: str) -> bytes` functions. Both hash helpers load and validate `PUSH_ENDPOINT_HASH_KEY` internally; `endpoint_hash` canonicalizes the URL before HMAC and `rate_limit_subject_hash` domain-separates the fixed-window subject. This phase imports them rather than reimplementing HMAC, URL normalization, or key loading.
- Modify: `.env.example` — document only placeholder-valued Phase 5 settings and their exact defaults; do not add a real secret.
- Create: `server/prune_push_rate_limits.py` — delete expired fixed-window counters through the same validated database settings without printing rows or subjects.
- Modify: `README.md` — document the exact safe counter-pruning command and the alert TTL, maximum-rule, and endpoint-privacy behavior.
- Create: `tests/backend/conftest.py` — provide deterministic UTC time, a valid synthetic PushSubscription, and injected database/Web Push seams for Phase 5 tests.
- Create: `tests/backend/test_push_lifecycle.py` — cover request validation, rate limiting, migrations, idempotency, management ownership, evaluator gates, claims, and terminal states.
- Modify: `src/lib/api/pushNotifications.ts` — replace the exists-then-upsert client with typed subscribe/list/cancel APIs that transmit the current browser subscription only when required.
- Modify: `src/facilities/CrowdAlertSubscriptionCard.tsx` — fetch server-backed rules on open, render expiry and management controls, call the single idempotent subscribe endpoint, and retain localStorage only for form defaults.
- Create: `src/lib/api/pushNotifications.test.ts` — assert the changed wire contract and safe rule parsing.
- Create: `src/facilities/CrowdAlertSubscriptionCard.test.tsx` — assert expiry copy, server-backed rule listing, idempotent subscribe UI, cancel-one, and cancel-all behavior.

### Task 1: Validate the applied Phase 1 schema and add Phase 5 configuration/test seams

**Files:**
- Create: `tests/backend/conftest.py`
- Create: `tests/backend/test_push_lifecycle.py`
- Modify: `.env.example`
- Modify: `server/forecast_api.py:1-126, 858-926`

**Interfaces:**
- Consumes: already-applied Phase 1 migrations `0003_push_rule_lifecycle.sql` and `0004_rate_limits.sql`, the Phase 1 migration runner/MySQL 8.4 test database, Phase 2 `location_snapshot` and `ingestion_runs` contracts, and the Phase 3 centralized 600-second freshness setting.
- Produces: `PushRuleRecord`, `PushRuleResponse`, `PushSubscriptionInput`, `validate_push_subscription(subscription: Mapping[str, Any]) -> ValidatedSubscription`, `require_admin_token(x_reclive_admin_token: Optional[str]) -> None`, `push_test_client(monkeypatch) -> TestClient`, `push_repository` test fixture, and `db_rule_status(rule_id: int) -> str` test helper. `endpoint_hash` and `rate_limit_subject_hash` remain imported exclusively from `server.reclive.push_identity`.

- [ ] **Step 1: Write the failing migration/configuration tests**

```python
def test_push_rule_schema_has_hashed_identity_and_pending_index(migrated_connection):
    columns = column_names(migrated_connection, "push_rules")
    assert {"endpoint_hash", "active_identity", "expires_at", "status", "claimed_at", "sent_at", "finalized_at", "failure_code"} <= columns
    assert column_type(migrated_connection, "push_rules", "endpoint_hash") == "binary(32)"
    assert index_columns(migrated_connection, "push_rules", "uq_push_rules_identity") == (
        "endpoint_hash", "facility_id", "section_key", "threshold", "active_identity",
    )
    assert column_is_generated(migrated_connection, "push_rules", "active_identity")
    assert "idx_push_rules_pending" in index_names(migrated_connection, "push_rules")
    assert column_type(migrated_connection, "push_rate_limits", "subject_hash") == "binary(32)"


def test_production_rejects_short_endpoint_hash_key(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PUSH_ENDPOINT_HASH_KEY", "short")
    with pytest.raises(RuntimeError, match="PUSH_ENDPOINT_HASH_KEY"):
        api.validate_push_configuration()
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_push_lifecycle.py::test_push_rule_schema_has_hashed_identity_and_pending_index tests/backend/test_push_lifecycle.py::test_production_rejects_short_endpoint_hash_key -q`

Expected: FAIL because `validate_push_configuration` does not exist. The schema assertions document the already-applied Phase 1 contract and must not be satisfied by editing an applied migration.

- [ ] **Step 3: Add only the Phase 5 configuration and test seams**

```python
PUSH_BODY_MAX_BYTES = 16 * 1024
PUSH_DEFAULT_RULE_TTL_SECONDS = int_with_default("PUSH_RULE_DEFAULT_TTL_SECONDS", 86_400)
PUSH_MAX_RULE_TTL_SECONDS = int_with_default("PUSH_RULE_MAX_TTL_SECONDS", 604_800)
PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT = int_with_default("PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT", 10)
PUSH_WRITE_RATE_LIMIT = int_with_default("PUSH_WRITE_RATE_LIMIT", 20)
PUSH_WRITE_RATE_WINDOW_SECONDS = int_with_default("PUSH_WRITE_RATE_WINDOW_SECONDS", 600)

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def validate_push_configuration() -> None:
    if env_with_default("APP_ENV", "development") != "production":
        return
    endpoint_hash("https://push-identity.invalid/startup-check")
    if push_admin_routes_enabled() and len(PUSH_ADMIN_TOKEN.encode("utf-8")) < 32:
        raise RuntimeError("PUSH_ADMIN_TOKEN must be at least 32 bytes when admin routes are enabled")
```

```python
@pytest.fixture
def valid_subscription() -> dict[str, object]:
    return {"endpoint": "https://push.reclive-notify.net/subscription-a", "keys": {"p256dh": VALID_P256DH, "auth": VALID_AUTH}}

@pytest.fixture
def push_test_client(monkeypatch: pytest.MonkeyPatch, push_repository: FakePushRepository) -> TestClient:
    monkeypatch.setattr(api, "open_db_connection", push_repository.open_connection)
    monkeypatch.setattr(api, "now_utc", lambda: FIXED_NOW)
    return TestClient(api.app)
```

Validate at startup that the TTL, maximum-TTL, active-rule, rate-limit, and window settings are positive, that the default TTL does not exceed the seven-day maximum, and that the maximum TTL does not exceed 604,800. Add placeholder settings to `.env.example`: `PUSH_ENDPOINT_HASH_KEY=change_me`, `PUSH_RULE_DEFAULT_TTL_SECONDS=86400`, `PUSH_RULE_MAX_TTL_SECONDS=604800`, `PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT=10`, `PUSH_WRITE_RATE_LIMIT=20`, and `PUSH_WRITE_RATE_WINDOW_SECONDS=600`. Call `validate_push_configuration()` during FastAPI lifespan startup before an evaluator task can start. In `require_admin_token`, replace `!=` with `hmac.compare_digest(x_reclive_admin_token or "", PUSH_ADMIN_TOKEN)`.

- [ ] **Step 4: Run the migration/configuration tests to verify they pass**

Run: `python -m pytest tests/backend/test_push_lifecycle.py::test_push_rule_schema_has_hashed_identity_and_pending_index tests/backend/test_push_lifecycle.py::test_production_rejects_short_endpoint_hash_key -q`

Expected: PASS; the Phase 1 migration runner yields the required applied fields/indexes, and the Phase 5 production short-key check fails closed.

- [ ] **Step 5: Commit the schema/configuration slice**

```bash
git add .env.example server/forecast_api.py tests/backend/conftest.py tests/backend/test_push_lifecycle.py
git commit -m "feat(push): add secure lifecycle configuration"
```

### Task 2: Parse bounded public requests, validate subscriptions, and rate-limit writes

**Files:**
- Modify: `server/forecast_api.py:1-126, 1237-1269, 1490-1595`
- Modify: `tests/backend/test_push_lifecycle.py`

**Interfaces:**
- Consumes: `PUSH_BODY_MAX_BYTES`, `endpoint_hash`, and `PushSubscriptionInput` from Task 1.
- Produces: `parse_limited_push_body(request: Request, model_type: type[T]) -> T`, `validate_push_subscription(value: Mapping[str, Any]) -> ValidatedSubscription`, `rate_limit_public_push_write(request: Request, decoded: object | None) -> None`, and HTTP `413`, `422`, and `429` responses containing only stable error categories.

- [ ] **Step 1: Write the failing validation and rate-limit tests**

```python
@pytest.mark.parametrize("subscription", [
    {"endpoint": "http://push.reclive-notify.net/a", "keys": VALID_KEYS},
    {"endpoint": "https://push.reclive-notify.net/" + "x" * 2049, "keys": VALID_KEYS},
    {"endpoint": "https://push.reclive-notify.net/a", "keys": {"p256dh": "AA", "auth": "AA"}},
])
def test_subscribe_rejects_invalid_subscription_without_echoing_endpoint(push_test_client, subscription):
    response = push_test_client.post("/api/push/subscribe", json={
        "subscription": subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": 40,
    })
    assert response.status_code == 422
    assert subscription["endpoint"] not in response.text


def test_subscribe_rejects_16_kib_body_before_json_parsing(push_test_client):
    response = push_test_client.post("/api/push/subscribe", content=b"{" + b"x" * (16 * 1024))
    assert response.status_code == 413


@pytest.mark.parametrize("content_length", ["-1", "sixteen kib", str(16 * 1024 + 1)])
def test_subscribe_safely_rejects_invalid_or_excessive_content_length(push_test_client, content_length):
    response = push_test_client.post(
        "/api/push/subscribe",
        headers={"content-length": content_length},
        content=b"{}",
    )
    assert response.status_code in {413, 422}
    assert "content-length" not in response.text.lower()


def test_twenty_first_push_write_in_window_is_rate_limited(push_test_client, valid_subscription):
    for _ in range(20):
        assert push_test_client.post("/api/push/rules/cancel-all", json={"subscription": valid_subscription}).status_code == 200
    response = push_test_client.post("/api/push/rules/cancel-all", json={"subscription": valid_subscription})
    assert response.status_code == 429


@pytest.mark.parametrize("threshold", [0, 101, 1.5, "40"])
def test_subscribe_rejects_out_of_range_or_noninteger_threshold(push_test_client, valid_subscription, threshold):
    response = push_test_client.post("/api/push/subscribe", json={
        "subscription": valid_subscription, "facilityId": 1186,
        "sectionKey": "overall", "threshold": threshold,
    })
    assert response.status_code == 422
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -k 'invalid_subscription or 16_kib or rate_limited or threshold' -q`

Expected: FAIL because the routes accept an HTTP endpoint and malformed/oversized default-parsed bodies, and no durable counter limits writes.

- [ ] **Step 3: Implement bounded parsing, strict crypto checks, and atomic fixed-window counters**

```python
class StrictPushModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)


class PushKeysInput(StrictPushModel):
    p256dh: str = Field(min_length=1, max_length=512)
    auth: str = Field(min_length=1, max_length=512)


class PushSubscriptionInput(StrictPushModel):
    endpoint: str = Field(min_length=1, max_length=2048)
    keys: PushKeysInput


class PushRuleRequest(StrictPushModel):
    subscription: PushSubscriptionInput
    facility_id: Literal[1186, 1656] = Field(alias="facilityId")
    section_key: str = Field(alias="sectionKey", min_length=1, max_length=80, pattern=r"^[a-z0-9_-]+$")
    threshold: int = Field(ge=1, le=100)
    ttl_seconds: int | None = Field(default=None, alias="ttlSeconds", ge=1, le=PUSH_MAX_RULE_TTL_SECONDS)


class PushOwnershipRequest(StrictPushModel):
    subscription: PushSubscriptionInput


async def parse_limited_push_body(request: Request, model_type: type[T]) -> T:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_size = int(declared, 10)
        except ValueError as exc:
            rate_limit_public_push_write(request, None)
            raise HTTPException(422, "invalid_push_content_length") from exc
        if declared_size < 0:
            rate_limit_public_push_write(request, None)
            raise HTTPException(422, "invalid_push_content_length")
        if declared_size > PUSH_BODY_MAX_BYTES:
            rate_limit_public_push_write(request, None)
            raise HTTPException(413, "push_request_too_large")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > PUSH_BODY_MAX_BYTES:
            rate_limit_public_push_write(request, None)
            raise HTTPException(413, "push_request_too_large")
        chunks.append(chunk)
    raw = b"".join(chunks)
    try:
        decoded: object = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        rate_limit_public_push_write(request, None)
        raise HTTPException(422, "invalid_push_request") from exc
    rate_limit_public_push_write(request, decoded)
    try:
        return model_type.model_validate(decoded)
    except ValidationError as exc:
        raise HTTPException(422, "invalid_push_request") from exc

def validate_push_subscription(value: Mapping[str, Any]) -> ValidatedSubscription:
    endpoint_value = value.get("endpoint")
    keys = value.get("keys")
    if not isinstance(endpoint_value, str) or not isinstance(keys, Mapping):
        raise HTTPException(422, "invalid_push_subscription")
    p256dh_value, auth_value = keys.get("p256dh"), keys.get("auth")
    try:
        endpoint = normalize_push_endpoint(endpoint_value)
        p256dh = decode_base64url(p256dh_value)
        auth = decode_base64url(auth_value)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise HTTPException(422, "invalid_push_subscription") from exc
    if len(p256dh) != 65 or p256dh[0] != 4 or len(auth) != 16:
        raise HTTPException(422, "invalid_push_subscription")
    return ValidatedSubscription(endpoint=endpoint, subscription={"endpoint": endpoint, "keys": {"p256dh": p256dh_value, "auth": auth_value}})
```

Implement `decode_base64url(value: object) -> bytes` to accept only a nonempty string of at most 512 ASCII base64url characters with optional trailing padding, add the necessary `=` padding locally, and call `base64.b64decode(..., altchars=b"-_", validate=True)`. Reject whitespace, misplaced padding, non-ASCII input, and decoding errors without including the value in an exception response.

After `parse_limited_push_body` returns a strict model, call `validate_push_subscription(model.subscription.model_dump())` exactly once and pass its canonical `ValidatedSubscription` to repository functions; never hash or store the pre-normalized Pydantic field directly.

```python
with conn.cursor() as cur:
    cur.execute(
        "INSERT INTO push_rate_limits (subject_hash, window_started_at, request_count, updated_at) VALUES (%s, %s, 1, %s) ON DUPLICATE KEY UPDATE request_count = request_count + 1, updated_at = VALUES(updated_at)",
        (subject_hash, window_started_at, now),
    )
    cur.execute("SELECT request_count FROM push_rate_limits WHERE subject_hash = %s AND window_started_at = %s", (subject_hash, window_started_at))
    request_count = int(cur.fetchone()[0])
conn.commit()
if request_count > PUSH_WRITE_RATE_LIMIT:
    raise HTTPException(429, "push_write_rate_limited")
```

Import `normalize_push_endpoint`, `endpoint_hash`, and `rate_limit_subject_hash` from `server.reclive.push_identity`; do not create a local HMAC/URL-normalization or key-loading variant. `rate_limit_public_push_write(request, decoded)` inspects only `decoded.subscription.endpoint` when it is a string and normalization succeeds; it uses `rate_limit_subject_hash("endpoint", normalized_endpoint)` in that case, otherwise `rate_limit_subject_hash("client", request.client.host if request.client else "unavailable")`. Floor the UTC epoch to the configured 600-second window and run the exact counter transaction keyed by `(subject_hash, window_started_at)`, committing the increment before returning or raising 429. Invalid `Content-Length`, oversized streams, invalid JSON, schema failures, and invalid subscriptions therefore consume the client- or endpoint-scoped public-write budget exactly once. Never pass a domain-prefixed non-URL to `endpoint_hash`, and never persist the raw fallback address. Subscribe, cancel-one, and cancel-all use this parser/guard before their write transaction; listing is read-only and is not rate-limited.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -k 'invalid_subscription or 16_kib or rate_limited or threshold' -q`

Expected: PASS; invalid values receive safe errors, the 16 KiB limit is enforced before model parsing, and exactly 20 write requests fit a window.

- [ ] **Step 5: Commit the request-safety slice**

```bash
git add server/forecast_api.py tests/backend/test_push_lifecycle.py
git commit -m "feat(push): validate and rate limit public writes"
```

### Task 3: Persist idempotent rules and expose endpoint-owned management APIs

**Files:**
- Modify: `server/forecast_api.py:928-1179, 1490-1595`
- Modify: `tests/backend/test_push_lifecycle.py`

**Interfaces:**
- Consumes: `ValidatedSubscription`, `endpoint_hash`, body parsing, and rate limiting from Tasks 1-2.
- Produces: `POST /api/push/subscribe`, `POST /api/push/rules/list`, `DELETE /api/push/rules/{rule_id}`, and `POST /api/push/rules/cancel-all`; `PushRuleResponse = {id: int, facilityId: int, sectionKey: str, threshold: int, createdAt: str, expiresAt: str, status: Literal["pending"]}`; `db_select_rule_by_id(cursor: Any, rule_id: int) -> PushRuleRecord`; and `resolve_owned_rule(endpoint: str, rule_id: int) -> PushRuleRecord`.

- [ ] **Step 1: Write the failing idempotency and ownership tests**

```python
def test_duplicate_subscribe_returns_existing_safe_rule(push_test_client, valid_subscription):
    payload = {"subscription": valid_subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": 40}
    first = push_test_client.post("/api/push/subscribe", json=payload)
    second = push_test_client.post("/api/push/subscribe", json=payload)
    assert first.json()["created"] is True
    assert second.json() == {"status": "ok", "created": False, "rule": first.json()["rule"]}
    assert "endpoint" not in second.text and "p256dh" not in second.text


def test_rule_list_and_cancel_require_the_current_subscription(push_test_client, valid_subscription, other_subscription):
    created = push_test_client.post("/api/push/subscribe", json={
        "subscription": valid_subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": 40,
    }).json()["rule"]
    response = push_test_client.request("DELETE", f"/api/push/rules/{created['id']}", json={"subscription": other_subscription})
    assert response.status_code == 404
    listed = push_test_client.post("/api/push/rules/list", json={"subscription": valid_subscription})
    assert [rule["id"] for rule in listed.json()["rules"]] == [created["id"]]


def test_hmac_match_still_requires_the_full_subscription_endpoint(push_test_client, monkeypatch, valid_subscription, other_subscription):
    monkeypatch.setattr(api, "endpoint_hash", lambda endpoint: b"x" * 32)
    created = push_test_client.post("/api/push/subscribe", json={"subscription": valid_subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": 40}).json()["rule"]
    response = push_test_client.request("DELETE", f"/api/push/rules/{created['id']}", json={"subscription": other_subscription})
    assert response.status_code == 404


def test_cancel_transitions_owned_pending_rule_to_cancelled(push_test_client, valid_subscription):
    created = push_test_client.post("/api/push/subscribe", json={"subscription": valid_subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": 40}).json()["rule"]
    response = push_test_client.request("DELETE", f"/api/push/rules/{created['id']}", json={"subscription": valid_subscription})
    assert response.json() == {"status": "ok", "cancelled": 1}
    assert db_rule_status(created["id"]) == "cancelled"


def test_eleventh_active_rule_for_endpoint_is_rejected(push_test_client, valid_subscription):
    for threshold in range(1, 11):
        assert push_test_client.post("/api/push/subscribe", json={"subscription": valid_subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": threshold}).status_code == 200
    response = push_test_client.post("/api/push/subscribe", json={"subscription": valid_subscription, "facilityId": 1186, "sectionKey": "overall", "threshold": 11})
    assert response.status_code == 409


def test_concurrent_tenth_and_eleventh_distinct_thresholds_leave_exactly_ten_active_rules(mysql_push_repository, valid_subscription):
    for threshold in range(1, 10):
        mysql_push_repository.subscribe(valid_subscription, facility_id=1186, section_key="overall", threshold=threshold)
    responses = run_in_parallel(
        lambda: mysql_push_repository.subscribe(valid_subscription, facility_id=1186, section_key="overall", threshold=10),
        lambda: mysql_push_repository.subscribe(valid_subscription, facility_id=1186, section_key="overall", threshold=11),
    )
    assert sorted(response.status_code for response in responses) == [200, 409]
    assert mysql_push_repository.active_rule_count(valid_subscription) == 10
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -k 'duplicate_subscribe or current_subscription or full_subscription_endpoint or cancelled or eleventh_active or concurrent_tenth' -q`

Expected: FAIL because the current API splits exists/upsert, has no list/cancel-one endpoint, identifies rows by raw endpoint, has no terminal-row-compatible active identity, and has no race-safe ten-rule cap.

- [ ] **Step 3: Implement safe responses and endpoint-hash ownership checks**

```python
class PushRuleResponse(BaseModel):
    id: int
    facilityId: int
    sectionKey: str
    threshold: int
    createdAt: datetime
    expiresAt: datetime
    status: Literal["pending"]

def resolve_owned_rule(endpoint: str, rule_id: int) -> PushRuleRecord:
    supplied_endpoint = normalize_push_endpoint(endpoint)
    digest = endpoint_hash(supplied_endpoint)
    rule = db_select_rule_by_hash_and_id(digest, rule_id)
    if rule is None or not hmac.compare_digest(rule.endpoint_hash, digest):
        raise HTTPException(404, "push_rule_not_found")
    stored_endpoint = normalize_push_endpoint(subscription_endpoint(rule.subscription_json))
    if not hmac.compare_digest(stored_endpoint, supplied_endpoint):
        raise HTTPException(404, "push_rule_not_found")
    return rule

def db_subscribe_rule(subscription: ValidatedSubscription, facility_id: int, section_key: str, threshold: int, ttl_seconds: int | None) -> tuple[bool, PushRuleRecord]:
    now = now_utc()
    expires_at = now + timedelta(seconds=ttl_seconds or PUSH_DEFAULT_RULE_TTL_SECONDS)
    with open_db_connection(autocommit=False) as conn:
        digest = endpoint_hash(subscription.endpoint)
        lock_name = f"reclive:push:{digest.hex()[:48]}"
        locked = False
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT GET_LOCK(%s, 2)", (lock_name,))
                locked = cur.fetchone()[0] == 1
                if not locked:
                    raise HTTPException(503, "push_subscribe_busy")
                cur.execute("UPDATE push_rules SET status = 'expired', finalized_at = %s WHERE endpoint_hash = %s AND active_identity IS NOT NULL AND status = 'pending' AND expires_at <= %s", (now, digest, now))
                cur.execute("SELECT id FROM push_rules WHERE endpoint_hash = %s AND facility_id = %s AND section_key = %s AND threshold = %s AND active_identity IS NOT NULL FOR UPDATE", (digest, facility_id, section_key, threshold))
                existing = cur.fetchone()
                if existing is not None:
                    existing_rule = db_select_rule_by_id(cur, int(existing[0]))
                    stored_endpoint = normalize_push_endpoint(subscription_endpoint(existing_rule.subscription_json))
                    if not hmac.compare_digest(stored_endpoint, subscription.endpoint):
                        raise HTTPException(409, "push_identity_conflict")
                    if existing_rule.status != "pending":
                        raise HTTPException(409, "push_rule_in_progress")
                    conn.commit()
                    return False, existing_rule
                cur.execute("SELECT COUNT(*) FROM push_rules WHERE endpoint_hash = %s AND active_identity IS NOT NULL AND status IN ('pending', 'claimed') AND expires_at > %s", (digest, now))
                if int(cur.fetchone()[0]) >= PUSH_MAX_ACTIVE_RULES_PER_ENDPOINT:
                    raise HTTPException(409, "push_rule_limit_reached")
                cur.execute("INSERT INTO push_rules (endpoint_hash, subscription_json, facility_id, section_key, threshold, created_at, expires_at, status) VALUES (%s, %s, %s, %s, %s, %s, %s, 'pending')", (digest, json.dumps(subscription.subscription, separators=(",", ":")), facility_id, section_key, threshold, now, expires_at))
                rule = db_select_rule_by_id(cur, int(cur.lastrowid))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if locked:
                with conn.cursor() as cur:
                    cur.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
    return True, rule
```

`active_identity` is the Phase 1 generated nullable column: it is non-null only for the active `pending` and `claimed` lifecycle states and null for `sent`, `failed`, `invalid_subscription`, `cancelled`, and `expired`. Consequently `uq_push_rules_identity(endpoint_hash, facility_id, section_key, threshold, active_identity)` enforces one current rule while retaining terminal audit rows and permits the same alert to be created again later. Do not alter that migration or emulate this field in application code.

Replace `/api/push/rules/exists` and `/api/push/unsubscribe` with the four management routes above. The subscribe request accepts optional `ttlSeconds`, defaults to 86,400, and rejects values above 604,800. Import and call `normalize_push_endpoint` before every endpoint hash, full-endpoint comparison, listing, ownership, cancellation, and rate-limit operation; use `endpoint_hash(canonical_endpoint)` only for canonical URLs and `rate_limit_subject_hash(subject_kind, subject)` only for rate-limit subjects. The shared helpers, not Phase 5 code, load and validate the configured HMAC key. Normalize and canonicalize the section, require that it belongs to the selected facility, calculate `expires_at` in UTC, and do not extend an existing rule's expiry on an idempotent duplicate. List only unexpired `pending` rows with non-null `active_identity`; for every HMAC-matched candidate, normalize the endpoint embedded in `subscription_json` and reject/filter it unless `hmac.compare_digest(stored_endpoint, supplied_endpoint)` succeeds. Cancel-one and cancel-all perform that same full endpoint verification before transitioning owned pending rules to `cancelled` with `finalized_at`; neither deletes rows. The unique index is the final authority for duplicate races, and duplicate-key recovery reselects only an active current row in the same transaction.

The bounded MySQL advisory lock is required because a `SELECT COUNT(*) ... FOR UPDATE` aggregate does not serialize inserts for different thresholds. Derive its fixed-length name only from the HMAC digest (`reclive:push:` plus the first 48 hex characters), acquire it with `GET_LOCK` on the **same connection** before expiry, idempotency lookup, count, and insert, and release it in `finally` before the connection closes. It must never contain the raw endpoint. The concurrent test uses the real MySQL repository rather than the in-memory seam to prove the tenth/eleventh race leaves exactly ten active rules.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -k 'duplicate_subscribe or current_subscription or full_subscription_endpoint or cancelled or eleventh_active or concurrent_tenth' -q`

Expected: PASS; retries return the same safe rule, a different subscription cannot list or cancel it, and an endpoint cannot retain more than ten active rules.

- [ ] **Step 5: Commit the lifecycle-management slice**

```bash
git add server/forecast_api.py tests/backend/test_push_lifecycle.py
git commit -m "feat(push): add idempotent rule management"
```

### Task 4: Make evaluator dispatch lock-first, fresh, covered, and at-most-once

**Files:**
- Modify: `server/forecast_api.py:1218-1487, 1598-1640`
- Modify: `tests/backend/test_push_lifecycle.py`

**Interfaces:**
- Consumes: pending rule schema, `db_acquire_evaluator_lock() -> Optional[Any]`, Phase 2 snapshot/ingestion tables, current official-hours JSON, an injected DNS resolver, and an injected pinned-TLS Web Push transport.
- Produces: `evaluate_rules_once(now: datetime | None = None) -> EvaluatorResult`, `load_evaluator_candidates(conn: Any, now: datetime) -> list[PushRuleRecord]`, `official_facility_is_open(payload: Mapping[str, Any], facility_id: int, at: datetime) -> bool`, `compute_fresh_section_metrics(facility_id: int, section_key: str, snapshots: Mapping[int, SnapshotRow], now: datetime) -> SectionMetrics | None`, `claim_pending_rule(conn: Any, rule_id: int, now: datetime) -> bool`, `resolve_public_push_addresses(endpoint: str) -> tuple[IPAddress, ...]`, `send_notification_pinned(subscription: Mapping[str, Any], title: str, body: str, url: str) -> None`, and terminal `sent`, `failed`, `invalid_subscription`, and `expired` states.

- [ ] **Step 1: Write the failing evaluator-state tests**

```python
def test_two_evaluators_can_claim_and_send_a_rule_only_once(push_repository, monkeypatch, valid_rule):
    push_repository.insert(valid_rule)
    sent: list[int] = []
    monkeypatch.setattr(api, "send_notification", lambda **_: sent.append(1))
    first, second = run_in_parallel(api.evaluate_rules_once, api.evaluate_rules_once)
    assert sum(result["sent"] for result in (first, second)) == 1
    assert sent == [1]
    assert push_repository.rule(valid_rule.id).status == "sent"


@pytest.mark.parametrize("gate", ["closed", "stale_ingestion", "missing_section", "coverage_below_80", "expired"])
def test_evaluator_does_not_claim_or_send_when_a_required_gate_fails(push_repository, monkeypatch, valid_rule, gate):
    push_repository.set_gate(gate)
    monkeypatch.setattr(api, "send_notification", lambda **_: pytest.fail("must not send"))
    api.evaluate_rules_once()
    assert push_repository.rule(valid_rule.id).status == ("expired" if gate == "expired" else "pending")


def test_web_push_410_marks_rule_invalid_without_deleting_audit_state(push_repository, monkeypatch, valid_rule):
    monkeypatch.setattr(api, "send_notification", raise_web_push_status(410))
    api.evaluate_rules_once()
    rule = push_repository.rule(valid_rule.id)
    assert (rule.status, rule.failure_code, rule.finalized_at is not None) == ("invalid_subscription", "webpush_410", True)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1/push",
        "https://10.0.0.1/push",
        "https://[::1]/push",
        "https://push.local/push",
        "https://localhost/push",
    ],
)
def test_static_private_or_local_push_destination_is_rejected(valid_subscription, endpoint):
    valid_subscription["endpoint"] = endpoint
    with pytest.raises(ValueError):
        validate_push_subscription(valid_subscription)


@pytest.mark.parametrize(
    "answers",
    [
        ["10.0.0.8"],
        ["2606:4700:4700::1111", "fc00::8"],
        [],
    ],
)
def test_dispatch_rejects_private_mixed_or_empty_dns_answers(
    monkeypatch, valid_subscription, answers
):
    monkeypatch.setattr(api, "resolve_endpoint_host", lambda *_: answers)
    monkeypatch.setattr(
        api, "send_prepared_web_push",
        lambda **_: pytest.fail("unsafe destination must not reach transport"),
    )

    with pytest.raises(api.SafePushDispatchError):
        api.send_notification_pinned(
            valid_subscription, title="RecLive", body="Ready", url="/nick"
        )


def test_dispatch_pins_validated_address_and_preserves_tls_hostname(
    monkeypatch, valid_subscription
):
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        api,
        "resolve_endpoint_host",
        lambda *_: ["2606:4700:4700::1111", "8.8.8.8"],
    )
    monkeypatch.setattr(
        api, "send_prepared_web_push", lambda **kwargs: calls.append(kwargs)
    )

    api.send_notification_pinned(
        valid_subscription, title="RecLive", body="Ready", url="/nick"
    )

    assert calls == [{
        "connect_ip": "2606:4700:4700::1111",
        "tls_server_hostname": "push.reclive-notify.net",
        "host_header": "push.reclive-notify.net",
        "allow_redirects": False,
        "subscription": valid_subscription,
        "title": "RecLive",
        "body": "Ready",
        "url": "/nick",
    }]


def test_dispatch_rejects_redirect_without_resolving_or_following_new_origin(
    monkeypatch, valid_subscription
):
    monkeypatch.setattr(
        api, "resolve_endpoint_host", lambda *_: ["8.8.8.8"]
    )
    monkeypatch.setattr(
        api, "send_prepared_web_push", raise_web_push_status(302)
    )

    with pytest.raises(api.SafePushDispatchError):
        api.send_notification_pinned(
            valid_subscription, title="RecLive", body="Ready", url="/nick"
        )
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -k 'two_evaluators or required_gate or web_push_410 or static_private or dns_answers or pins_validated or rejects_redirect' -q`

Expected: FAIL because the current evaluator loads rules and live data before taking the advisory lock, has no freshness/coverage/schedule/expiry gates, deletes terminal rules, and uses no atomic claim.

- [ ] **Step 3: Implement the lock-first evaluator state machine**

```python
def claim_pending_rule(conn: Any, rule_id: int, now: datetime) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """UPDATE push_rules
               SET status = 'claimed', claimed_at = %s
               WHERE id = %s AND status = 'pending' AND expires_at > %s""",
            (now, rule_id, now),
        )
        return cur.rowcount == 1

def facility_notification_url(facility_id: int) -> str:
    if facility_id == 1186:
        return "/nick"
    if facility_id == 1656:
        return "/bakke"
    raise ValueError("unsupported facility")
```

Acquire `GET_LOCK` before loading anything else, return a safe `skippedLocked` result when unavailable, and release the same connection in `finally`. Under that lock, mark expired pending rows as `expired`, load only unexpired pending rules, load the official hours payload, latest successful ingestion time, and relevant `location_snapshot` rows, then evaluate each rule. `official_facility_is_open` must return `False` for a missing facility, non-`ok` official schedule, missing matching day row, `closed`/maintenance notice, unparseable hours, or a time outside the matching official interval; it therefore fails closed until a trustworthy schedule proves the facility open. `compute_fresh_section_metrics` must count only relevant fresh rows, calculate coverage against configured section capacity, and return no eligible metric below `0.80`.

Immediately before every outbound attempt, call `normalize_push_endpoint`, resolve the canonical hostname and port with the injected resolver, normalize every returned address with `ipaddress.ip_address`, and require a nonempty set containing only global unicast addresses. Reject a private-only, loopback, link-local, multicast, unspecified, reserved, or mixed public/non-public answer without opening a socket. Select the first address from a deterministically sorted validated set and pass that literal as `connect_ip` to the constrained transport. The transport must connect directly to that pinned literal, create TLS with the canonical endpoint hostname as `server_hostname`, retain normal certificate and hostname verification, send the canonical hostname in `Host`, and set `allow_redirects=False`. Prepare the encrypted Web Push request with the library, but do not let `pywebpush`, `requests`, or another client resolve the hostname again; a custom HTTPS connection/adapter or equivalently constrained outbound proxy is required. Treat every 3xx as `webpush_failed` without following `Location` or resolving another origin. Resolver and transport doubles are mandatory in tests; never contact a provider.

Call `claim_pending_rule` immediately before `send_notification_pinned`. After a successful send, finalize the claimed row as `sent` with `sent_at` and `finalized_at`. Map Web Push 404 and 410 to `invalid_subscription` and `failure_code` `webpush_404` or `webpush_410`; map every other send, DNS-validation, TLS, redirect, or transport error to `failed` with bounded non-sensitive code `webpush_failed`. Do not change `claimed` rows back to pending, do not delete a terminal row, and do not print exception text. Make the admin evaluate and dispatch routes use the same terminal-state helper and `hmac.compare_digest` authorization.

- [ ] **Step 4: Run the focused tests to verify they pass**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -k 'two_evaluators or required_gate or web_push_410 or static_private or dns_answers or pins_validated or rejects_redirect' -q`

Expected: PASS; only one worker sends, every failed eligibility gate prevents dispatch, expired rows are terminally expired, `410` becomes `invalid_subscription`, DNS answers are all-global, the TLS socket is pinned while SNI/certificate verification retain the hostname, and redirects fail closed without a raw endpoint leak or provider call.

- [ ] **Step 5: Commit the evaluator slice**

```bash
git add server/forecast_api.py tests/backend/test_push_lifecycle.py
git commit -m "feat(push): make evaluator lock-safe and at-most-once"
```

### Task 5: Add a safe rate-limit maintenance operation and document alert lifecycle limits

**Files:**
- Create: `server/prune_push_rate_limits.py`
- Modify: `README.md`
- Modify: `tests/backend/test_push_lifecycle.py`

**Interfaces:**
- Consumes: `open_db_connection`, `PUSH_WRITE_RATE_WINDOW_SECONDS`, and the `push_rate_limits.updated_at` index from Tasks 1-2.
- Produces: `prune_push_rate_limits(now: datetime | None = None) -> int` and the documented command `python server/prune_push_rate_limits.py`.

- [ ] **Step 1: Write the failing maintenance test**

```python
def test_prune_removes_only_expired_rate_limit_windows(push_repository, fixed_now):
    push_repository.insert_rate_limit("old-subject", fixed_now - timedelta(seconds=1201))
    push_repository.insert_rate_limit("current-subject", fixed_now - timedelta(seconds=599))
    assert prune_push_rate_limits(now=fixed_now) == 1
    assert push_repository.rate_limit_subjects() == {"current-subject"}
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `python -m pytest tests/backend/test_push_lifecycle.py::test_prune_removes_only_expired_rate_limit_windows -q`

Expected: FAIL because no pruning command or `prune_push_rate_limits` function exists.

- [ ] **Step 3: Implement the bounded maintenance command and documentation**

```python
from forecast_api import PUSH_WRITE_RATE_WINDOW_SECONDS, open_db_connection

def prune_push_rate_limits(now: datetime | None = None) -> int:
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(seconds=PUSH_WRITE_RATE_WINDOW_SECONDS * 2)
    with open_db_connection(autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM push_rate_limits WHERE updated_at < %s", (cutoff,))
            removed = int(cur.rowcount)
        conn.commit()
    return removed

if __name__ == "__main__":
    print(f"pruned_push_rate_limit_windows={prune_push_rate_limits()}")
```

Document that the command removes only counters older than two 600-second windows, prints only a removal count, runs with the normal private database settings, and must never be run with environment dumping. Document the 24-hour default, seven-day maximum, ten-active-rule limit, server-backed management behavior, and that RecLive never logs or returns raw subscription endpoints.

- [ ] **Step 4: Run the focused maintenance test to verify it passes**

Run: `python -m pytest tests/backend/test_push_lifecycle.py::test_prune_removes_only_expired_rate_limit_windows -q`

Expected: PASS; only expired counters are removed and no raw subject value is printed.

- [ ] **Step 5: Commit the maintenance/documentation slice**

```bash
git add server/prune_push_rate_limits.py README.md tests/backend/test_push_lifecycle.py
git commit -m "docs(push): document rate limit maintenance"
```

### Task 6: Replace the frontend exists/upsert flow with typed server-backed rule APIs

**Files:**
- Modify: `src/lib/api/pushNotifications.ts`
- Create: `src/lib/api/pushNotifications.test.ts`

**Interfaces:**
- Consumes: backend routes from Task 3 and existing `ensurePushSubscription()` / `getExistingPushSubscription()`.
- Produces: `PushRule`, `SubscribePushRulePayload`, `subscribePushRule(payload) -> Promise<{created: boolean; rule: PushRule}>`, `listPushRules(subscription) -> Promise<PushRule[]>`, `cancelPushRule(id, subscription) -> Promise<void>`, and `cancelAllPushRules(subscription) -> Promise<number>`.

- [ ] **Step 1: Write the failing frontend API tests**

```ts
import {HttpResponse, http} from "msw";
import {server} from "../../test/msw/server";

const validSubscription: PushSubscriptionJSON = {
    endpoint: "https://push.reclive-notify.net/subscription-a",
    expirationTime: null,
    keys: {p256dh: "p256dh-fixture", auth: "auth-fixture"},
};
const validPayload = {subscription: validSubscription, facilityId: 1186, sectionKey: "overall", threshold: 40};
const validRule = {id: 7, facilityId: 1186, sectionKey: "overall", threshold: 40,
    createdAt: "2026-08-31T12:00:00Z", expiresAt: "2026-09-01T12:00:00Z", status: "pending" as const};

it("subscribes once and returns the safe server rule", async () => {
    let method = "";
    server.use(http.post("*/api/push/subscribe", ({request}) => {
        method = request.method;
        return HttpResponse.json({status: "ok", created: false, rule: validRule});
    }));
    await expect(subscribePushRule(validPayload)).resolves.toMatchObject({created: false, rule: {id: 7}});
    expect(method).toBe("POST");
});

it("lists and cancels with the current subscription", async () => {
    let deleteBody: unknown;
    server.use(
        http.post("*/api/push/rules/list", () => HttpResponse.json({status: "ok", rules: [validRule]})),
        http.delete("*/api/push/rules/7", async ({request}) => {
            deleteBody = await request.json();
            return HttpResponse.json({status: "ok", cancelled: 1});
        }),
    );
    await expect(listPushRules(validSubscription)).resolves.toEqual([validRule]);
    await expect(cancelPushRule(7, validSubscription)).resolves.toBeUndefined();
    expect(deleteBody).toEqual({subscription: validSubscription});
});
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `npm run test:run -- src/lib/api/pushNotifications.test.ts`

Expected: FAIL because the client exports `hasMatchingPushRule`, has no typed management API, and `upsertPushRule` returns no safe rule.

- [ ] **Step 3: Implement the typed wire contract**

```ts
export interface PushRule {
    id: number;
    facilityId: number;
    sectionKey: string;
    threshold: number;
    createdAt: string;
    expiresAt: string;
    status: "pending";
}

export const subscribePushRule = async (payload: SubscribePushRulePayload): Promise<{created: boolean; rule: PushRule}> => {
    const response = await fetch(resolveApiUrl("/api/push/subscribe"), {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload),
    });
    if (!response.ok) throw new Error("Could not save this alert right now.");
    return parseSubscribeResponse(await response.json());
};
```

Delete `PushRuleExistsPayload` and `hasMatchingPushRule`. Keep the PushSubscription only in outgoing subscribe/list/cancel requests; do not cache or render the endpoint, `p256dh`, or `auth`. Parse each response so malformed safe-rule data becomes a generic client error rather than UI state.

- [ ] **Step 4: Run the focused frontend API tests to verify they pass**

Run: `npm run test:run -- src/lib/api/pushNotifications.test.ts`

Expected: PASS; one request performs subscribe and the management calls carry the current browser subscription without exposing it in returned types.

- [ ] **Step 5: Commit the frontend API slice**

```bash
git add src/lib/api/pushNotifications.ts src/lib/api/pushNotifications.test.ts
git commit -m "feat(push): add server-backed browser rule API"
```

### Task 7: Render server-backed alert management and truthful expiry copy

**Files:**
- Modify: `src/facilities/CrowdAlertSubscriptionCard.tsx`
- Create: `src/facilities/CrowdAlertSubscriptionCard.test.tsx`
- Modify: `src/app/components/AlertsPanel.tsx`

**Interfaces:**
- Consumes: Task 6 browser API, `PushRule`, `getExistingPushSubscription`, existing selected facility/section options, and existing standalone-PWA gating.
- Produces: a "Manage alerts" section listing only server-returned active rules for the current subscription; accessible cancel-one/cancel-all controls; expiry text; and localStorage form-default updates only after a successful server operation.

- [ ] **Step 1: Write the failing component tests**

```tsx
import {render, screen} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {beforeEach, expect, it, vi} from "vitest";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import CrowdAlertSubscriptionCard from "./CrowdAlertSubscriptionCard";

const pushApi = vi.hoisted(() => ({
    getExistingPushSubscription: vi.fn(),
    getPushAvailability: vi.fn(),
    listPushRules: vi.fn(),
    cancelPushRule: vi.fn(),
    cancelAllPushRules: vi.fn(),
    subscribePushRule: vi.fn(),
    ensurePushSubscription: vi.fn(),
}));
vi.mock("../lib/api/pushNotifications", () => ({
    ...pushApi,
    isWebPushSupported: () => true,
}));

const validSubscription: PushSubscriptionJSON = {
    endpoint: "https://push.reclive-notify.net/subscription-a",
    expirationTime: null,
    keys: {p256dh: "p256dh-fixture", auth: "auth-fixture"},
};
const managedRules = [
    {id: 7, facilityId: 1186, sectionKey: "overall", threshold: 40, createdAt: "2026-08-31T12:00:00Z", expiresAt: "2026-09-01T12:00:00Z", status: "pending" as const},
    {id: 8, facilityId: 1186, sectionKey: "fitness", threshold: 30, createdAt: "2026-08-31T12:00:00Z", expiresAt: "2026-09-01T12:00:00Z", status: "pending" as const},
];
const summary: OccupancySummary = {
    count: 10, observedCapacity: 100, expectedOpenCapacity: 100, coverage: 1,
    percent: 10, observedLocations: 1, expectedLocations: 1,
    latestFetchedAt: "2026-08-31T12:00:00Z", oldestFetchedAt: "2026-08-31T12:00:00Z", status: "live",
};

function renderWithManagedRules() {
    return render(<CrowdAlertSubscriptionCard onClose={vi.fn()} facility={1186} isOpen sections={[
        {key: "overall", label: "Entire Facility", summary},
        {key: "fitness", label: "Fitness", summary},
    ]} />);
}

beforeEach(() => {
    vi.clearAllMocks();
    pushApi.getExistingPushSubscription.mockResolvedValue({toJSON: () => validSubscription});
    pushApi.getPushAvailability.mockResolvedValue({apiAvailable: true, dbAvailable: true, alertsAvailable: true, reason: null});
    pushApi.listPushRules.mockResolvedValue(managedRules);
    pushApi.cancelPushRule.mockResolvedValue(undefined);
    pushApi.cancelAllPushRules.mockResolvedValue(1);
});

it("shows server-backed active rules and their expiry", async () => {
    renderWithManagedRules();
    expect(await screen.findByText("Manage alerts")).toBeVisible();
    expect(screen.getAllByText(/expires Sep 1/i)).toHaveLength(2);
});

it("cancels one server rule and clears all remaining server rules", async () => {
    renderWithManagedRules();
    await userEvent.click((await screen.findAllByRole("button", {name: "Cancel alert"}))[0]);
    expect(pushApi.cancelPushRule).toHaveBeenCalledWith(7, validSubscription);
    await userEvent.click(screen.getByRole("button", {name: "Cancel all alerts"}));
    expect(pushApi.cancelAllPushRules).toHaveBeenCalledWith(validSubscription);
});
```

- [ ] **Step 2: Run the focused component tests to verify they fail**

Run: `npm run test:run -- src/facilities/CrowdAlertSubscriptionCard.test.tsx`

Expected: FAIL because the card checks a separate exists endpoint, closes after subscription, does not list server rules, and has no expiry or cancellation controls.

- [ ] **Step 3: Implement management UI without treating localStorage as truth**

```tsx
const [managedRules, setManagedRules] = useState<PushRule[]>([]);

const refreshManagedRules = async (subscription: PushSubscriptionJSON) => {
    setManagedRules(await listPushRules(subscription));
};

<Typography variant="caption" color="text.secondary">
    Alerts expire 24 hours after they are created unless you cancel them sooner.
</Typography>
<Typography variant="subtitle2">Manage alerts</Typography>
{managedRules.map((rule) => (
    <Stack key={rule.id} direction="row" justifyContent="space-between">
        <Typography>{`${rule.sectionKey} at ${rule.threshold}% — expires ${formatExpiry(rule.expiresAt)}`}</Typography>
        <Button aria-label="Cancel alert" onClick={() => void handleCancelRule(rule.id)}>Cancel</Button>
    </Stack>
))}
```

Import `OccupancySummary` from `src/shared/occupancy/computeOccupancySummary` and make every `AlertSectionOption` exactly `{ key: string; label: string; summary: OccupancySummary }`. Replace all old `total`, `max`, and `percent` reads with `selectedSection.summary`. Enable thresholds only for `summary.status === "live" || summary.status === "partial"` with a non-null `summary.percent`; partial selections display their observed-capacity coverage and unknown/insufficient selections are unavailable. The component test must pass the concrete `summary` object above, never the obsolete fields.

When the panel opens, read the active browser subscription and call `listPushRules`; retain a visible safe error if that lookup fails. Subscribe exactly once with `subscribePushRule`, refresh the managed list from the response/list endpoint, show an idempotent duplicate as an existing alert rather than an error, and leave the panel open so the user can manage it. Write the existing localStorage shape only as the next form default after a successful subscribe; remove matching convenience values after successful cancellation, but never use its contents to decide which rules exist. Add an explicit `Cancel all alerts` button only when the server list is non-empty and give the list a polite live region for subscribe/cancel success. Preserve the desktop modal and mobile drawer behavior by passing no new route state through `AlertsPanel`.

- [ ] **Step 4: Run the focused component tests to verify they pass**

Run: `npm run test:run -- src/facilities/CrowdAlertSubscriptionCard.test.tsx && npm run build`

Expected: PASS; expiry and management state come from the API, cancellation calls endpoint-owned routes, and TypeScript builds.

- [ ] **Step 5: Commit the management UI slice**

```bash
git add src/facilities/CrowdAlertSubscriptionCard.tsx src/facilities/CrowdAlertSubscriptionCard.test.tsx src/app/components/AlertsPanel.tsx
git commit -m "feat(push): manage active alerts in the dashboard"
```

### Task 8: Run Phase 5 integration verification without exposing push identities

**Files:**
- Modify: `tests/backend/test_push_lifecycle.py`
- Modify: `src/lib/api/pushNotifications.test.ts`
- Modify: `src/facilities/CrowdAlertSubscriptionCard.test.tsx`

**Interfaces:**
- Consumes: all Phase 5 routes, migrations, evaluator state transitions, and UI contracts from Tasks 1-7.
- Produces: verified Phase 5 behavior with no raw endpoint values in test output, response payloads, logs, or repository changes.

- [ ] **Step 1: Write the final missing regression tests**

```python
def test_threshold_above_100_is_rejected_and_success_uses_facility_route(push_test_client, monkeypatch, valid_subscription):
    rejected = push_test_client.post("/api/push/subscribe", json={
        "subscription": valid_subscription, "facilityId": 1656, "sectionKey": "overall", "threshold": 101,
    })
    assert rejected.status_code == 422
    created = push_test_client.post("/api/push/subscribe", json={
        "subscription": valid_subscription, "facilityId": 1656, "sectionKey": "overall", "threshold": 40,
    })
    assert created.status_code == 200
    sent: dict[str, str] = {}
    monkeypatch.setattr(api, "send_notification", lambda **kwargs: sent.update(url=kwargs["url"]))
    api.evaluate_rules_once()
    assert sent["url"] == "/bakke"
```

- [ ] **Step 2: Run the final integration regression and keep it green**

Run: `python -m pytest tests/backend/test_push_lifecycle.py::test_threshold_above_100_is_rejected_and_success_uses_facility_route -q`

Expected: PASS because Task 2 owns the strict range and Task 4 owns the direct-facility URL mapping. A failure identifies a regression in that owning task; do not add a second validation or route implementation here.

- [ ] **Step 3: Inspect the integrated response boundary**

Confirm the request is still parsed by Task 2's `StrictPushModel` hierarchy (`extra="forbid"`, strict integer threshold, literal facility IDs, bounded section key, bounded TTL) and the evaluator still calls Task 4's `facility_notification_url`. Keep all evaluator result fields numeric or fixed category strings. Do not include endpoint hashes, rule subscription JSON, exception messages, SQL, headers, or environment values in health, evaluator, or error responses.

- [ ] **Step 4: Run the Phase 5 verification commands**

Run: `python -m pytest tests/backend/test_push_lifecycle.py -q && npm run test:run -- src/lib/api/pushNotifications.test.ts src/facilities/CrowdAlertSubscriptionCard.test.tsx && npm run lint && npm run build && git diff --check`

Expected: PASS for every executed test, lint, build, and whitespace check. If MySQL 8.4 is unavailable, record the migration-backed backend tests as unexecuted; do not infer success.

- [ ] **Step 5: Commit the verified Phase 5 completion**

```bash
git add server/forecast_api.py server/prune_push_rate_limits.py .env.example README.md tests/backend/test_push_lifecycle.py src/lib/api/pushNotifications.ts src/lib/api/pushNotifications.test.ts src/facilities/CrowdAlertSubscriptionCard.tsx src/facilities/CrowdAlertSubscriptionCard.test.tsx src/app/components/AlertsPanel.tsx
git commit -m "feat(push): complete secure alert lifecycle"
```

## Self-Review

**Spec coverage:** Task 2 covers `request.stream()` bounded 16 KiB parsing, invalid `Content-Length` rejection, HTTPS/key/length checks, 1-100 thresholds, no raw endpoint error text, and durable fixed-window rate limits. Tasks 1 and 3 cover production key/token requirements, `compare_digest`, Phase 1 migrations, shared canonical endpoint/HMAC identity plus full-endpoint verification, generated nullable `active_identity`, terminal-row recreation, TTL/status/claim fields, idempotent subscribe, the same-connection HMAC-derived advisory lock that makes the ten-active-rule cap race-safe, and list/cancel-one/cancel-all. Task 4 covers evaluator lock-before-load, official schedule, ingestion freshness, relevant-row freshness, coverage, expiry, conditional claim, 404/410 invalidation, terminal status, and direct facility routes. Tasks 6-7 replace the browser exists/upsert flow with a server-backed management UI and expiry copy using Phase 3 `OccupancySummary` options. Task 5 documents pruning, and Task 8 validates all public contracts without logging identities.

**Placeholder scan:** This plan has no unresolved work markers, deferred implementation, generic validation direction, or cross-task shorthand. Every task declares its files, consumed and produced interfaces, a concrete failing test, a RED command/result, implementation code, a GREEN command/result, and a commit.

**Type consistency:** Backend safe-rule responses use `id`, `facilityId`, `sectionKey`, `threshold`, `createdAt`, `expiresAt`, and `status: "pending"` throughout Tasks 3, 6, and 7. Browser ownership always transmits `PushSubscriptionJSON`; raw subscription fields never appear in `PushRule`. The evaluator uses the same `status` vocabulary and `facility_notification_url` mapping defined in Task 4. Every alert option consumes `summary: OccupancySummary`, not the retired `total`, `max`, or `percent` fields.
