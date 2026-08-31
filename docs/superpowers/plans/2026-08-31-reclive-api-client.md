# RecLive API Configuration and Request Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace RecLive's duplicated frontend API configuration and request behavior with one runtime-validated client and independent stale-while-revalidate polling schedules.

**Architecture:** A pure public-config parser and one Axios-backed `requestJson` boundary normalize URLs, cancellation, timeouts, retry classification, jitter, `Retry-After`, and Zod validation. Existing facility, forecast, schedule, push, and cache modules become thin domain adapters. A visibility-aware polling hook provides distinct live, forecast, and schedule refresh keys without overlapping work or clearing current data.

**Tech Stack:** React 19, TypeScript 5.9, Vite 7, Axios 1.13, Zod, Vitest, React Testing Library, MSW, Python 3.12, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md`

## Global Constraints

- Preserve facility IDs `1186` and `1656`, routes `/nick` and `/bakke`, forecast behavior, push behavior, visual identity, and current-data-on-refresh behavior.
- The master plan's phase-level commit policy is authoritative: task commit snippets describe staging/review scope only; make the single Phase 6 source commit after Task 5.
- Frontend public configuration is limited to `VITE_API_BASE_URL` and `VITE_SITE_URL`; `LIVE_COUNTS_URL` remains backend-private.
- Production fails closed on missing required values, `change_me`, `YOUR_ACCOUNT_API_KEY`, wildcard CORS, or weak enabled-admin authentication without logging values.
- Retry only network failures, HTTP 408, 429, and 5xx. Never retry aborts, invalid JSON/schema data, or HTTP 400/401/403/404.
- Do not expose database values, upstream URLs containing credentials, admin tokens, VAPID private material, or push endpoints in logs or test output.
- Production code must follow a witnessed RED test before GREEN implementation.

---

### Task 1: Unify public and private runtime configuration

**Files:**
- Modify: `.env.example`
- Modify: `src/lib/config/env.ts`
- Modify: `src/vite-env.d.ts`
- Test: `src/lib/config/env.test.ts`
- Modify: `server/env_loader.py`
- Test: `tests/backend/test_env_loader.py`
- Modify locally without staging: ignored `.env` (rename frontend-only keys without displaying values; preserve every backend credential line byte-for-byte)

**Interfaces:**
- Produces: `parsePublicEnv(values: Record<string, unknown>, isProduction: boolean): PublicEnv`
- Produces: `env.apiBaseUrl`, `env.siteUrl`, `env.isDev`
- Produces: `validate_production_environment(values: Mapping[str, str], *, required_names: Sequence[str], cors_name: str | None, admin_enabled: bool) -> None`
- Consumes: current entry-point environment names and the Phase 5 admin-token contract.

- [ ] **Step 1: Write failing frontend configuration tests**

```ts
import {describe, expect, it} from "vitest";
import {parsePublicEnv} from "./env";

describe("parsePublicEnv", () => {
    it("normalizes the single API base and site URL", () => {
        expect(parsePublicEnv({
            VITE_API_BASE_URL: "https://api.example.test/",
            VITE_SITE_URL: "https://reclive.example.test/",
        }, true)).toMatchObject({
            apiBaseUrl: "https://api.example.test",
            siteUrl: "https://reclive.example.test",
        });
    });

    it.each(["change_me", "https://example.test/?AccountAPIKey=YOUR_ACCOUNT_API_KEY"])(
        "rejects production placeholder %s without including its value in the error",
        (value) => {
            expect(() => parsePublicEnv({VITE_API_BASE_URL: value, VITE_SITE_URL: "https://site.test"}, true))
                .toThrow(/VITE_API_BASE_URL/);
        },
    );

    it.each(["javascript:alert(1)", "https://user:password@api.example.test", "https://api.example.test/?token=hidden"])(
        "rejects an unsafe public URL without echoing it: %s",
        (value) => {
            expect(() => parsePublicEnv({VITE_API_BASE_URL: value, VITE_SITE_URL: "https://site.test"}, true))
                .toThrow(/VITE_API_BASE_URL/);
        },
    );
});
```

- [ ] **Step 2: Run the frontend test and verify RED**

Run: `npm run test:run -- src/lib/config/env.test.ts`

Expected: FAIL because `parsePublicEnv` and `apiBaseUrl` do not exist.

- [ ] **Step 3: Implement the public environment parser and replace Vite declarations**

```ts
export interface PublicEnv {
    apiBaseUrl: string;
    siteUrl: string;
    isDev: boolean;
}

const unsafePlaceholder = (value: string): boolean => (
    value === "change_me" || value.includes("YOUR_ACCOUNT_API_KEY")
);

const normalizedPublicUrl = (name: string, value: unknown, required: boolean): string => {
    const text = typeof value === "string" ? value.trim() : "";
    if (!text) {
        if (required) throw new Error(`${name} is not configured safely`);
        return "";
    }
    if (unsafePlaceholder(text)) throw new Error(`${name} is not configured safely`);
    try {
        const parsed = new URL(text);
        if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) {
            throw new Error("unsafe");
        }
        return parsed.toString().replace(/\/+$/, "");
    } catch {
        throw new Error(`${name} is not configured safely`);
    }
};

export const parsePublicEnv = (
    values: Record<string, unknown>,
    isProduction: boolean,
): PublicEnv => {
    const apiBaseUrl = normalizedPublicUrl("VITE_API_BASE_URL", values.VITE_API_BASE_URL, isProduction);
    const siteUrl = normalizedPublicUrl("VITE_SITE_URL", values.VITE_SITE_URL, isProduction);
    return {apiBaseUrl, siteUrl, isDev: !isProduction};
};

const parsed = parsePublicEnv(import.meta.env as Record<string, unknown>, import.meta.env.PROD);
export const env = {...parsed, isDev: import.meta.env.DEV};
```

Declare only `VITE_API_BASE_URL` and `VITE_SITE_URL` in `src/vite-env.d.ts`. Replace the three old public base variables in `.env.example` with those two names and keep `LIVE_COUNTS_URL` under the private backend section.

Migrate the ignored local `.env` mechanically without sending values to stdout or stderr. If `rg -q '^VITE_API_BASE_URL=' .env` succeeds, run `perl -i -ne 'print unless /^VITE_(?:FORECAST_API_BASE_URL|LIVE_COUNTS_URL|PUSH_API_BASE_URL)=/' .env`; otherwise run `perl -i -pe 's/^VITE_FORECAST_API_BASE_URL=/VITE_API_BASE_URL=/; $_="" if /^VITE_(?:LIVE_COUNTS_URL|PUSH_API_BASE_URL)=/' .env`. Verify only variable names with `awk -F= '/^[A-Za-z_][A-Za-z0-9_]*=/{print $1}' .env`. Do not stage `.env`, alter `LIVE_COUNTS_URL`, or alter any `GYM_DB_*` line.

- [ ] **Step 4: Run the frontend configuration test and verify GREEN**

Run: `npm run test:run -- src/lib/config/env.test.ts`

Expected: PASS.

- [ ] **Step 5: Write failing backend production-validation tests**

```python
import pytest

from server.env_loader import validate_production_environment


def safe_values() -> dict[str, str]:
    return {
        "APP_ENV": "production",
        "LIVE_COUNTS_URL": "https://upstream.invalid/live",
        "FORECAST_API_ALLOW_ORIGINS": "https://reclive.example",
        "PUSH_ADMIN_TOKEN": "a" * 32,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [("LIVE_COUNTS_URL", "change_me"), ("LIVE_COUNTS_URL", "YOUR_ACCOUNT_API_KEY")],
)
def test_production_rejects_placeholder_without_echoing_value(name: str, value: str) -> None:
    values = safe_values()
    values[name] = value
    with pytest.raises(RuntimeError) as error:
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name="FORECAST_API_ALLOW_ORIGINS",
            admin_enabled=True,
        )
    assert name in str(error.value)
    assert value not in str(error.value)


def test_production_rejects_wildcard_cors() -> None:
    values = safe_values() | {"FORECAST_API_ALLOW_ORIGINS": "*"}
    with pytest.raises(RuntimeError, match="FORECAST_API_ALLOW_ORIGINS"):
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name="FORECAST_API_ALLOW_ORIGINS",
            admin_enabled=True,
        )


def test_production_rejects_short_admin_token_when_routes_are_enabled() -> None:
    values = safe_values() | {"PUSH_ADMIN_TOKEN": "short"}
    with pytest.raises(RuntimeError, match="PUSH_ADMIN_TOKEN"):
        validate_production_environment(
            values,
            required_names=("LIVE_COUNTS_URL",),
            cors_name="FORECAST_API_ALLOW_ORIGINS",
            admin_enabled=True,
        )
```

- [ ] **Step 6: Run the backend test and verify RED**

Run: `pytest -q tests/backend/test_env_loader.py`

Expected: FAIL because `validate_production_environment` is undefined.

- [ ] **Step 7: Implement fail-closed backend validation**

```python
from collections.abc import Mapping, Sequence


def validate_production_environment(
    values: Mapping[str, str], *, required_names: Sequence[str],
    cors_name: str | None, admin_enabled: bool
) -> None:
    if str(values.get("APP_ENV", "development")).strip().lower() != "production":
        return
    for name in required_names:
        value = str(values.get(name, "")).strip()
        if not value or value == "change_me" or "YOUR_ACCOUNT_API_KEY" in value:
            raise RuntimeError(f"Unsafe production configuration: {name}")
    if cors_name:
        origins = [item.strip() for item in str(values.get(cors_name, "")).split(",") if item.strip()]
        if not origins or "*" in origins:
            raise RuntimeError(f"Unsafe production configuration: {cors_name}")
    token = str(values.get("PUSH_ADMIN_TOKEN", ""))
    if admin_enabled and len(token.encode("utf-8")) < 32:
        raise RuntimeError("Unsafe production configuration: PUSH_ADMIN_TOKEN")
```

Call this validator before each entry point performs I/O, without logging the environment mapping:

```python
# server/gym_fetch.py
validate_production_environment(
    os.environ,
    required_names=("LIVE_COUNTS_URL", "GYM_DB_HOST", "GYM_DB_PORT", "GYM_DB_USER", "GYM_DB_PASSWORD", "GYM_DB_NAME"),
    cors_name=None,
    admin_enabled=False,
)

# server/forecast_api.py, during lifespan startup
push_admin_routes_enabled = str(os.environ.get("PUSH_ADMIN_ROUTES_ENABLED", "false")).strip().lower() in {"1", "true", "yes", "on"}
validate_production_environment(
    os.environ,
    required_names=("GYM_DB_HOST", "GYM_DB_PORT", "GYM_DB_USER", "GYM_DB_PASSWORD", "GYM_DB_NAME", "FORECAST_JSON_PATH", "FACILITY_HOURS_JSON_PATH"),
    cors_name="FORECAST_API_ALLOW_ORIGINS",
    admin_enabled=push_admin_routes_enabled,
)

# server/forecast_job.py, at main() entry
validate_production_environment(
    os.environ,
    required_names=("GYM_DB_HOST", "GYM_DB_PORT", "GYM_DB_USER", "GYM_DB_PASSWORD", "GYM_DB_NAME", "MODEL_ARTIFACT_DIR", "MODEL_BASENAME", "FORECAST_JSON_PATH"),
    cors_name=None,
    admin_enabled=False,
)

# server/facility_hours_fetch.py, at main() entry
validate_production_environment(
    os.environ,
    required_names=("FACILITY_HOURS_JSON_PATH",),
    cors_name=None,
    admin_enabled=False,
)
```

`PUSH_ADMIN_ROUTES_ENABLED` is the Phase 5 environment setting. Tests import entry points with synthetic environment mappings and assert errors contain names only.

- [ ] **Step 8: Run focused and package checks**

Run: `pytest -q tests/backend/test_env_loader.py && npm run test:run -- src/lib/config/env.test.ts && npm run build`

Expected: all commands exit 0.

- [ ] **Step 9: Commit the configuration contract**

```bash
git add .env.example src/lib/config/env.ts src/lib/config/env.test.ts src/vite-env.d.ts server/env_loader.py tests/backend/test_env_loader.py server/gym_fetch.py server/forecast_api.py server/forecast_job.py server/facility_hours_fetch.py
git commit -m "feat: fail closed on unsafe runtime configuration"
```

### Task 2: Define Zod schemas for every untrusted frontend payload

**Files:**
- Modify: `package.json`
- Modify: `package-lock.json`
- Create: `src/lib/api/schemas.ts`
- Test: `src/lib/api/schemas.test.ts`
- Modify: `src/lib/types/facility.ts`
- Modify: `src/lib/types/forecast.ts`
- Modify: `src/lib/types/facilitySchedule.ts`

**Interfaces:**
- Produces: `liveCountsResponseSchema`, `forecastResponseSchema`, `actualHoursResponseSchema`, `facilityScheduleSchema`, `pushAvailabilitySchema`, `pushRuleResponseSchema`, `pushRuleListSchema`, `facilityCacheSchema`.
- Produces: TypeScript types inferred with `z.infer` for domain adapters.
- Consumes: Phase 2 live envelope, Phase 4 actual-hour contract, and Phase 5 push-rule response contract.

- [ ] **Step 1: Add Zod and write malformed-payload tests**

Run: `npm install zod`

Then create tests including:

```ts
import {describe, expect, it} from "vitest";
import {actualHoursResponseSchema, facilityCacheSchema, liveCountsResponseSchema} from "./schemas";

it("rejects a malformed live row instead of casting it", () => {
    expect(() => liveCountsResponseSchema.parse({
        ingestion: {lastSuccessfulFetchAt: null, ageSeconds: null, status: "unavailable"},
        rows: [{LocationId: "not-an-id", IsClosed: false, LastCount: 4, FetchedAt: null}],
    })).toThrow();
});

it("allows a low-coverage actual with a null actualCount", () => {
    const parsed = actualHoursResponseSchema.parse({
        facilityId: 1186,
        date: "2026-08-31",
        categories: [],
        totalHours: [{hourStart: "2026-08-31T10:00:00-05:00", observedCount: 4,
            observedCapacity: 10, expectedCapacity: 20, actualCoverage: 0.5,
            temporalCoverage: 1, coverageThreshold: 0.75, actualCount: null}],
    });
    expect(parsed.totalHours[0]?.actualCount).toBeNull();
});

it("rejects cachedAt values that are not finite numbers", () => {
    expect(() => facilityCacheSchema.parse({version: 3, cachedAt: "tomorrow", payload: {}})).toThrow();
});
```

- [ ] **Step 2: Run schema tests and verify RED**

Run: `npm run test:run -- src/lib/api/schemas.test.ts`

Expected: FAIL because `schemas.ts` does not exist.

- [ ] **Step 3: Implement strict reusable schemas**

```ts
import {z} from "zod";

const isoDateTime = z.string().datetime({offset: true});
const nullableIsoDateTime = isoDateTime.nullable();
const finiteNonnegative = z.number().finite().nonnegative();
const facilityId = z.union([z.literal(1186), z.literal(1656)]);

export const liveLocationRowSchema = z.object({
    LocationId: z.number().int().positive(),
    IsClosed: z.boolean().nullable(),
    LastCount: z.number().finite().nonnegative().nullable(),
    LastUpdatedDateAndTime: nullableIsoDateTime.optional(),
    FetchedAt: nullableIsoDateTime,
}).strict();

const legacyLiveLocationRowSchema = liveLocationRowSchema
    .omit({FetchedAt: true})
    .extend({FetchedAt: nullableIsoDateTime.optional()})
    .transform((row) => ({...row, FetchedAt: row.FetchedAt ?? null}));

const ingestionHealthSchema = z.object({
    lastSuccessfulFetchAt: nullableIsoDateTime,
    ageSeconds: finiteNonnegative.nullable(),
    status: z.enum(["healthy", "stale", "unavailable"]),
}).strict();

const canonicalLiveCountsSchema = z.object({
    ingestion: ingestionHealthSchema,
    rows: z.array(liveLocationRowSchema),
}).strict();

const unavailableIngestion = {
    lastSuccessfulFetchAt: null,
    ageSeconds: null,
    status: "unavailable" as const,
};

export const liveCountsResponseSchema = z.union([
    canonicalLiveCountsSchema,
    z.array(legacyLiveLocationRowSchema).transform((rows) => ({ingestion: unavailableIngestion, rows})),
    z.object({data: z.array(legacyLiveLocationRowSchema)}).strict()
        .transform(({data}) => ({ingestion: unavailableIngestion, rows: data})),
]);

export const occupancyThresholdSchema = z.object({
    lowMax: z.number().finite().min(0).max(99),
    peakMin: z.number().finite().min(1).max(100),
}).strict().refine(({lowMax, peakMin}) => lowMax < peakMin, "lowMax must be below peakMin");

export const forecastHourSchema = z.object({
    hourStart: isoDateTime,
    expectedCount: finiteNonnegative,
    expectedPct: z.number().finite().min(0).max(1).nullable().optional(),
    actualCount: finiteNonnegative.nullable().optional(),
    actualPct: z.number().finite().min(0).max(1).nullable().optional(),
    actualSampleCount: z.number().int().nonnegative().optional(),
    actualCoverage: z.number().finite().min(0).max(1).nullable().optional(),
}).strict();

const forecastWindowSchema = z.object({
    start: isoDateTime,
    end: isoDateTime,
    startHour: z.number().int().min(0).max(23).optional(),
    endHour: z.number().int().min(0).max(24).optional(),
    windowHours: finiteNonnegative.optional(),
    expectedTotal: finiteNonnegative.optional(),
    expectedAvg: finiteNonnegative.optional(),
    sampleCountMin: z.number().int().nonnegative().optional(),
}).strict();

const forecastBandSchema = z.object({
    start: isoDateTime,
    end: isoDateTime,
    level: z.enum(["low", "medium", "peak"]),
}).strict();

const forecastCategorySchema = z.object({
    key: z.string().trim().min(1).max(100),
    title: z.string().trim().min(1).max(160),
    maxCapacity: finiteNonnegative.nullable().optional(),
    hours: z.array(forecastHourSchema),
}).strict();

const forecastDaySchema = z.object({
    dayName: z.string().trim().min(1).max(32),
    date: z.iso.date(),
    categories: z.array(forecastCategorySchema).optional(),
    totalHours: z.array(forecastHourSchema).optional(),
    avoidWindows: z.array(forecastWindowSchema).optional(),
    bestWindows: z.array(forecastWindowSchema).optional(),
    crowdBands: z.array(forecastBandSchema).optional(),
}).strict();

export const forecastResponseSchema = z.object({
    facilityId,
    facilityName: z.string().trim().min(1).max(160),
    forecastDayStartHour: z.number().int().min(0).max(23).optional(),
    forecastDayEndHour: z.number().int().min(0).max(23).optional(),
    occupancyThresholds: occupancyThresholdSchema.nullable().optional(),
    sectionOccupancyThresholds: z.record(z.string(), occupancyThresholdSchema).nullable().optional(),
    locationOccupancyThresholds: z.record(z.string().regex(/^\d+$/), occupancyThresholdSchema).nullable().optional(),
    weeklyForecast: z.array(forecastDaySchema),
}).strict();

export const actualHourSchema = z.object({
    hourStart: isoDateTime,
    observedCount: finiteNonnegative.nullable(),
    observedCapacity: finiteNonnegative,
    expectedCapacity: finiteNonnegative,
    actualCoverage: z.number().finite().min(0).max(1),
    temporalCoverage: z.number().finite().min(0).max(1),
    coverageThreshold: z.number().finite().min(0).max(1),
    actualCount: finiteNonnegative.nullable(),
    actualPct: z.number().finite().min(0).max(1).nullable().optional(),
}).strict();

const actualCategorySchema = z.object({
    key: z.string().trim().min(1).max(100),
    title: z.string().trim().min(1).max(160),
    hours: z.array(actualHourSchema),
}).strict();

export const actualHoursResponseSchema = z.object({
    facilityId,
    date: z.iso.date(),
    categories: z.array(actualCategorySchema),
    totalHours: z.array(actualHourSchema),
}).strict();

const scheduleRowSchema = z.object({
    label: z.string().trim().min(1).max(160),
    hours: z.string().trim().min(1).max(240),
}).strict();

const scheduleSectionSchema = z.object({
    title: z.string().trim().min(1).max(160),
    rows: z.array(scheduleRowSchema),
    note: z.string().trim().max(500).nullable().optional(),
}).strict();

export const facilityScheduleSchema = z.object({
    generatedAt: nullableIsoDateTime,
    sourceSite: z.string().url().nullable(),
    facilityId,
    facilityName: z.string().trim().min(1).max(160),
    slug: z.string().trim().min(1).max(80),
    url: z.string().url(),
    resolvedUrl: z.string().url().nullable(),
    status: z.enum(["ok", "stale", "error"]),
    source: z.enum(["direct_html", "wp_json"]).nullable(),
    sections: z.array(scheduleSectionSchema),
    sourceFetchedAt: nullableIsoDateTime.optional(),
    lastSuccessfulAt: nullableIsoDateTime.optional(),
    stale: z.boolean().optional(),
    error: z.string().trim().max(240).nullable(),
    errorCategory: z.enum(["anti_bot", "upstream_timeout", "upstream_http", "wp_payload_invalid", "parse_empty", "schema_invalid", "io_error"]).nullable().optional(),
    updatedAt: nullableIsoDateTime,
}).strict();

export const pushAvailabilitySchema = z.object({
    apiAvailable: z.boolean(),
    dbAvailable: z.boolean(),
    alertsAvailable: z.boolean(),
    reason: z.string().trim().max(100).nullable(),
}).strict();

export const pushPublicKeySchema = z.object({
    publicKey: z.string().trim().min(1).max(512),
}).strict();

export const pushRuleSchema = z.object({
    id: z.number().int().positive(),
    facilityId,
    sectionKey: z.string().trim().min(1).max(80),
    threshold: z.number().int().min(1).max(100),
    createdAt: isoDateTime,
    expiresAt: isoDateTime,
    status: z.literal("pending"),
}).strict();

export const pushRuleResponseSchema = z.object({
    status: z.literal("ok"),
    created: z.boolean(),
    rule: pushRuleSchema,
}).strict();

export const pushRuleListSchema = z.object({
    status: z.literal("ok"),
    rules: z.array(pushRuleSchema),
}).strict();

const locationSchema = z.object({
    facilityId,
    locationId: z.number().int().positive(),
    locationName: z.string().trim().min(1).max(160),
    floor: z.number().int(),
    isClosed: z.boolean().nullable(),
    currentCapacity: finiteNonnegative.nullable(),
    maxCapacity: finiteNonnegative.nullable(),
    lastUpdated: z.string().nullable(),
    fetchedAt: nullableIsoDateTime,
}).strict();

const facilityPayloadSchema = z.object({
    facilityId,
    facilityName: z.string().trim().min(1).max(160),
    floors: z.record(z.string().regex(/^\d+$/), z.array(locationSchema)),
    locations: z.array(locationSchema),
    liveDataSource: z.enum(["facility_api", "fallback_api", "cache"]).optional(),
}).strict();

export const facilityCacheSchema = z.object({
    version: z.literal(3),
    cachedAt: z.number().finite().nonnegative(),
    payload: facilityPayloadSchema,
}).strict();

export type LiveCountsResponse = z.infer<typeof liveCountsResponseSchema>;
export type ForecastResponse = z.infer<typeof forecastResponseSchema>;
export type ActualHoursResponse = z.infer<typeof actualHoursResponseSchema>;
export type FacilityScheduleResponse = z.infer<typeof facilityScheduleSchema>;
export type PushRule = z.infer<typeof pushRuleSchema>;
```

Phase 6 accepts the current schedule record plus the known Phase 7 extension fields, but no unknown keys or unvalidated values. Phase 7 then makes `sourceFetchedAt`, `lastSuccessfulAt`, `stale`, and `errorCategory` required, requires nonempty sections for `ok`/`stale`, and keeps `error` rows out of the successful single-facility response.

- [ ] **Step 4: Run schema tests and verify GREEN**

Run: `npm run test:run -- src/lib/api/schemas.test.ts`

Expected: PASS.

- [ ] **Step 5: Run type and dependency checks**

Run: `npm run build && npm audit --omit=dev`

Expected: build exits 0; audit result is recorded without suppressing findings.

- [ ] **Step 6: Commit runtime schemas**

```bash
git add package.json package-lock.json src/lib/api/schemas.ts src/lib/api/schemas.test.ts src/lib/types/facility.ts src/lib/types/forecast.ts src/lib/types/facilitySchedule.ts
git commit -m "feat: validate untrusted API and cache payloads"
```

### Task 3: Build the shared request client and retry policy

**Files:**
- Create: `src/lib/api/client.ts`
- Test: `src/lib/api/client.test.ts`
- Read: `src/shared/utils/retry.ts` (Task 4 removes it only after every caller migrates)

**Interfaces:**
- Produces: `requestJson<T>(path: string, schema: ZodType<T>, options?: RequestOptions): Promise<T>`
- Produces: `ApiError`, `shouldRetryApiError`, `parseRetryAfterMs`, `uniqueApiUrls`.
- Consumes: `env.apiBaseUrl` and Zod schemas from Tasks 1-2.

- [ ] **Step 1: Write RED tests for retries, aborts, schema errors, and duplicate URLs**

```ts
import {HttpResponse, http} from "msw";
import {expect, it, vi} from "vitest";
import {z} from "zod";
import {server} from "../../test/msw/server";
import {requestJson, uniqueApiUrls} from "./client";

it("retries 503 then succeeds", async () => {
    let calls = 0;
    server.use(http.get("*/api/example", () => (++calls === 1
        ? HttpResponse.json({}, {status: 503})
        : HttpResponse.json({ok: true}))));
    await expect(requestJson("/api/example", z.object({ok: z.literal(true)}), {
        attempts: 2, jitter: () => 0, sleep: async () => undefined,
    })).resolves.toEqual({ok: true});
    expect(calls).toBe(2);
});

it.each([400, 401, 403, 404])("does not retry HTTP %s", async (status) => {
    let calls = 0;
    server.use(http.get("*/api/example", () => { calls += 1; return new HttpResponse(null, {status}); }));
    await expect(requestJson("/api/example", z.object({}), {attempts: 3})).rejects.toMatchObject({status});
    expect(calls).toBe(1);
});

it("does not retry schema failures", async () => {
    server.use(http.get("*/api/example", () => HttpResponse.json({ok: "wrong"})));
    await expect(requestJson("/api/example", z.object({ok: z.boolean()}), {attempts: 3}))
        .rejects.toMatchObject({kind: "schema"});
});

it("does not retry invalid JSON", async () => {
    let calls = 0;
    server.use(http.get("*/api/example", () => {
        calls += 1;
        return new HttpResponse("not-json", {headers: {"Content-Type": "application/json"}});
    }));
    await expect(requestJson("/api/example", z.object({}), {attempts: 3}))
        .rejects.toMatchObject({kind: "invalid_json"});
    expect(calls).toBe(1);
});

it("honors Retry-After for 429 responses", async () => {
    const sleep = vi.fn().mockResolvedValue(undefined);
    let calls = 0;
    server.use(http.get("*/api/example", () => (++calls === 1
        ? new HttpResponse(null, {status: 429, headers: {"Retry-After": "2"}})
        : HttpResponse.json({ok: true}))));
    await requestJson("/api/example", z.object({ok: z.literal(true)}), {attempts: 2, sleep});
    expect(sleep).toHaveBeenCalledWith(2_000, undefined);
});

it("aborts without retrying", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(requestJson("/api/example", z.object({}), {attempts: 3, signal: controller.signal}))
        .rejects.toMatchObject({kind: "aborted"});
});

it("deduplicates equivalent candidate URLs", () => {
    expect(uniqueApiUrls(["/api/live-counts", "/api/live-counts", "//api/live-counts"]))
        .toEqual(["/api/live-counts"]);
});
```

- [ ] **Step 2: Run client tests and verify RED**

Run: `npm run test:run -- src/lib/api/client.test.ts`

Expected: FAIL because `client.ts` does not exist.

- [ ] **Step 3: Implement normalized errors and transient-only retry**

```ts
import axios from "axios";
import type {ZodType} from "zod";
import {env} from "../config/env";

export type ApiErrorKind = "aborted" | "network" | "timeout" | "http" | "invalid_json" | "schema";
export class ApiError extends Error {
    constructor(public readonly kind: ApiErrorKind, message: string,
        public readonly status: number | null = null, public readonly retryAfterMs: number | null = null) {
        super(message);
    }
}

export const shouldRetryApiError = (error: ApiError): boolean => (
    error.kind === "network" || error.kind === "timeout" || error.status === 408
    || error.status === 429 || (error.status !== null && error.status >= 500)
);

export const uniqueApiUrls = (values: string[]): string[] => (
    [...new Set(values.map((value) => `/${value.replace(/^\/+/, "")}`))]
);

export interface RequestOptions {
    signal?: AbortSignal;
    timeoutMs?: number;
    attempts?: number;
    params?: Record<string, string | number>;
    method?: "GET" | "POST" | "DELETE";
    body?: unknown;
    jitter?: () => number;
    sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
}

export const parseRetryAfterMs = (value: unknown, now = Date.now()): number | null => {
    if (typeof value !== "string" || !value.trim()) return null;
    const seconds = Number(value);
    if (Number.isFinite(seconds) && seconds >= 0) return Math.round(seconds * 1000);
    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? Math.max(0, timestamp - now) : null;
};

const abortableSleep = (ms: number, signal?: AbortSignal): Promise<void> => new Promise((resolve, reject) => {
    const finish = () => {
        signal?.removeEventListener("abort", abort);
        resolve();
    };
    const timer = globalThis.setTimeout(finish, ms);
    const abort = () => {
        globalThis.clearTimeout(timer);
        signal?.removeEventListener("abort", abort);
        reject(new ApiError("aborted", "Request aborted"));
    };
    if (signal?.aborted) { abort(); return; }
    signal?.addEventListener("abort", abort, {once: true});
});

const normalizeApiError = (cause: unknown): ApiError => {
    if (cause instanceof ApiError) return cause;
    if (axios.isCancel(cause)) return new ApiError("aborted", "Request aborted");
    if (axios.isAxiosError(cause)) {
        if (cause.code === "ECONNABORTED") return new ApiError("timeout", "Request timed out");
        if (!cause.response) return new ApiError("network", "Network request failed");
        return new ApiError(
            "http",
            `API returned HTTP ${cause.response.status}`,
            cause.response.status,
            parseRetryAfterMs(cause.response.headers["retry-after"]),
        );
    }
    return new ApiError("network", "Network request failed");
};

export async function requestJson<T>(path: string, schema: ZodType<T>, options: RequestOptions = {}): Promise<T> {
    const attempts = Math.max(1, options.attempts ?? 3);
    for (let attempt = 1; attempt <= attempts; attempt += 1) {
        try {
            const response = await axios.request({
                method: options.method ?? "GET", url: `${env.apiBaseUrl}${path}`,
                signal: options.signal, timeout: options.timeoutMs ?? 10_000,
                params: options.params, data: options.body,
                responseType: "text", transformResponse: [(value) => value],
            });
            let decoded: unknown;
            try { decoded = JSON.parse(response.data as string); }
            catch { throw new ApiError("invalid_json", "API returned invalid JSON"); }
            const parsed = schema.safeParse(decoded);
            if (!parsed.success) throw new ApiError("schema", "API response did not match its contract");
            return parsed.data;
        } catch (cause) {
            const error = normalizeApiError(cause);
            if (attempt >= attempts || !shouldRetryApiError(error)) throw error;
            const exponential = 500 * (2 ** (attempt - 1));
            const delay = error.retryAfterMs ?? Math.round(exponential * (0.5 + (options.jitter?.() ?? Math.random())));
            await (options.sleep ?? abortableSleep)(delay, options.signal);
        }
    }
    throw new ApiError("network", "API request failed");
}
```

Keep all error messages limited to stable categories and status codes. Do not include response bodies or URLs.

- [ ] **Step 4: Run client tests and verify GREEN**

Run: `npm run test:run -- src/lib/api/client.test.ts`

Expected: PASS with fake sleep and no real delay.

- [ ] **Step 5: Remove the old retry helper after callers migrate in Task 4**

Do not delete `src/shared/utils/retry.ts` until `rg -n "retryAsync" src` returns no callers.

- [ ] **Step 6: Commit the shared request boundary**

```bash
git add src/lib/api/client.ts src/lib/api/client.test.ts
git commit -m "feat: centralize API retry and error behavior"
```

### Task 4: Migrate domain adapters and preserve qualified data

**Files:**
- Modify: `src/lib/api/facilityParser.ts`
- Test: `src/lib/api/facilityParser.test.ts`
- Modify: `src/lib/api/forecastParser.ts`
- Test: `src/lib/api/forecastParser.test.ts`
- Modify: `src/lib/api/facilityScheduleParser.ts`
- Test: `src/lib/api/facilityScheduleParser.test.ts`
- Modify: `src/lib/api/pushNotifications.ts`
- Test: `src/lib/api/pushNotifications.test.ts`
- Modify: `src/lib/storage/facilityCache.ts`
- Test: `src/lib/storage/facilityCache.test.ts`
- Remove: `src/shared/utils/retry.ts`

**Interfaces:**
- Consumes: `requestJson` and schemas from Tasks 2-3.
- Preserves: `fetchFacility`, `fetchForecastDays`, `fetchFacilityHours`, and Phase 5 push helper public names where consumers already use them.
- Produces: one normalized request per endpoint and schema-validated cache reads.

- [ ] **Step 1: Add RED adapter tests**

Tests must prove:

```ts
const validForecast = {
    facilityId: 1186,
    facilityName: "Nicholas Recreation Center",
    weeklyForecast: [{
        dayName: "Monday",
        date: "2026-08-31",
        categories: [],
        totalHours: [{hourStart: "2026-08-31T12:00:00Z", expectedCount: 20}],
        avoidWindows: [],
        bestWindows: [],
        crowdBands: [],
    }],
};

it("does not retry the same live URL as its own fallback", async () => {
    let requests = 0;
    server.use(http.get("*/api/live-counts", () => {
        requests += 1;
        return new HttpResponse(null, {status: 404});
    }));
    await expect(fetchFacility(1186)).rejects.toMatchObject({status: 404});
    expect(requests).toBe(1);
});

it("keeps forecast data when actual-hours validation fails", async () => {
    server.use(
        http.get("*/api/forecast/facilities/1186", () => HttpResponse.json(validForecast)),
        http.get("*/api/forecast/facilities/1186/actual-hours", () => HttpResponse.json({bad: true})),
    );
    const payload = await fetchForecastDays(1186);
    expect(payload.days).toHaveLength(1);
    expect(payload.days[0]?.totalHours[0]?.actualCount).toBeUndefined();
});

it("rejects a schema-invalid cached payload", () => {
    localStorage.setItem(CACHE_KEY, JSON.stringify({1186: {version: 3, cachedAt: Date.now(), payload: {bad: true}}}));
    expect(getFacilityCache(1186)).toBeNull();
});
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run: `npm run test:run -- src/lib/api/facilityParser.test.ts src/lib/api/forecastParser.test.ts src/lib/api/facilityScheduleParser.test.ts src/lib/api/pushNotifications.test.ts src/lib/storage/facilityCache.test.ts`

Expected: at least the duplicate-URL, schema-validation, and cache-contract assertions fail under existing behavior.

- [ ] **Step 3: Replace per-module Axios/fetch calls with `requestJson`**

```ts
export async function fetchFacilityHours(facilityId: FacilityId, signal?: AbortSignal) {
    return requestJson(
        `/api/facility-hours/facilities/${facilityId}`,
        facilityScheduleSchema,
        {signal},
    );
}
```

Build live candidate paths through `uniqueApiUrls`; after Phase 2 the only production candidate is `/api/live-counts`, because `requestJson` adds `env.apiBaseUrl` exactly once. Never pass an absolute URL into `uniqueApiUrls` or `requestJson`. Fetch forecast and actual concurrently:

```ts
const [forecastResult, actualResult] = await Promise.allSettled([
    requestJson(forecastPath, forecastResponseSchema, {signal, params: {compact: 1}}),
    requestJson(`${forecastPath}/actual-hours`, actualHoursResponseSchema, {signal, params: {date: today}}),
]);
if (forecastResult.status === "rejected") throw forecastResult.reason;
const days = actualResult.status === "fulfilled"
    ? mergeActualHoursIntoDays(forecastResult.value.weeklyForecast, actualResult.value)
    : forecastResult.value.weeklyForecast;
```

Parse cache entries with `facilityCacheSchema.safeParse`, then apply Phase 3's future/staleness checks. Return stable user-facing errors without `console.error` objects that can contain Axios config or URLs.

- [ ] **Step 4: Verify all adapter tests GREEN and remove old retry helper**

Run: `npm run test:run -- src/lib/api src/lib/storage/facilityCache.test.ts && ! rg -n "retryAsync|VITE_LIVE_COUNTS_URL|VITE_FORECAST_API_BASE_URL|VITE_PUSH_API_BASE_URL" src`

Expected: tests pass and `rg` returns no matches. Delete `src/shared/utils/retry.ts` only after this check.

- [ ] **Step 5: Run lint and build**

Run: `npm run lint && npm run build`

Expected: both exit 0.

- [ ] **Step 6: Commit adapter migration**

```bash
git add src/lib/api src/lib/storage/facilityCache.ts src/lib/storage/facilityCache.test.ts src/shared/utils/retry.ts
git commit -m "refactor: route frontend data through one API client"
```

### Task 5: Separate visibility-aware polling and retain data during refresh

**Files:**
- Create: `src/app/hooks/useVisibilityPolling.ts`
- Test: `src/app/hooks/useVisibilityPolling.test.tsx`
- Modify: `src/app/hooks/useLiveFacilityData.ts`
- Test: `src/app/hooks/useLiveFacilityData.test.tsx`
- Modify: `src/app/hooks/useForecastData.ts`
- Test: `src/app/hooks/useForecastData.test.tsx`
- Modify: `src/app/hooks/useFacilityHours.ts`
- Test: `src/app/hooks/useFacilityHours.test.tsx`
- Modify: `src/app/hooks/usePullToRefresh.ts`
- Modify: `src/app/App.tsx`

**Interfaces:**
- Produces: `useVisibilityPolling({intervalMs, onRefresh, enabled, refreshOnVisible}): void`.
- Produces: independent `liveRefreshKey`, `forecastRefreshKey`, and `scheduleRefreshKey` state in `App`.
- Consumes: existing data-hook return contracts and shared request cancellation.

- [ ] **Step 1: Write RED visibility and no-overlap tests**

```tsx
it("pauses while hidden and refreshes immediately when visible", async () => {
    vi.useFakeTimers();
    const onRefresh = vi.fn();
    Object.defineProperty(document, "visibilityState", {value: "hidden", configurable: true});
    renderHook(() => useVisibilityPolling({intervalMs: 90_000, onRefresh, enabled: true, refreshOnVisible: true}));
    await vi.advanceTimersByTimeAsync(180_000);
    expect(onRefresh).not.toHaveBeenCalled();

    Object.defineProperty(document, "visibilityState", {value: "visible", configurable: true});
    document.dispatchEvent(new Event("visibilitychange"));
    expect(onRefresh).toHaveBeenCalledTimes(1);
});

it("retains forecast rows while a refresh is pending", async () => {
    let resolveRefresh!: (value: FacilityForecastPayload) => void;
    const pendingRefresh = new Promise<FacilityForecastPayload>((resolve) => { resolveRefresh = resolve; });
    mockedFetchForecastDays.mockResolvedValueOnce(firstPayload).mockReturnValueOnce(pendingRefresh);
    const {result, rerender} = renderHook(({key}) => useForecastData({facility: 1186, refreshKey: key}), {initialProps: {key: 0}});
    await waitFor(() => expect(result.current.forecastDays).toEqual(firstPayload.days));
    rerender({key: 1});
    expect(result.current.forecastDays).toEqual(firstPayload.days);
});

it("retains a schema-valid cached snapshot while offline and labels its source", async () => {
    const cachedLocation = {
        facilityId: 1186, locationId: 5761, locationName: "Power House", floor: 1,
        isClosed: false, currentCapacity: 30, maxCapacity: 50,
        lastUpdated: "2026-08-31T11:55:00Z", fetchedAt: "2026-08-31T12:00:00Z",
    } satisfies Location;
    const cachedPayload = {
        facilityId: 1186, facilityName: "Nick",
        floors: {1: [cachedLocation]}, locations: [cachedLocation], liveDataSource: "facility_api",
    } satisfies FacilityPayload;
    setFacilityCache(1186, cachedPayload);

    const {result} = renderHook(() => useLiveFacilityData({facility: 1186, refreshKey: 0, isOffline: true}));

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current).toMatchObject({
        data: cachedPayload, liveDataSource: "cache", liveOutageState: "cache", error: null,
    });
    expect(mockedFetchFacility).not.toHaveBeenCalled();
});
```

Import `waitFor`/`renderHook`, `fetchFacility`, `setFacilityCache`, `FacilityPayload`, and `Location` explicitly in the hook test; after its existing `vi.mock` call define `const mockedFetchFacility = vi.mocked(fetchFacility)`. Reset localStorage and mocks in the shared test cleanup. This assertion protects the required offline cached-snapshot behavior while the visibility and refresh assertions provide the RED behavior for this task.

- [ ] **Step 2: Run hook tests and verify RED**

Run: `npm run test:run -- src/app/hooks/useVisibilityPolling.test.tsx src/app/hooks/useLiveFacilityData.test.tsx src/app/hooks/useForecastData.test.tsx src/app/hooks/useFacilityHours.test.tsx`

Expected: FAIL because visibility polling is absent and existing hooks clear some current-facility state.

- [ ] **Step 3: Implement the polling hook**

```ts
import {useEffect, useEffectEvent} from "react";

export function useVisibilityPolling({intervalMs, onRefresh, enabled, refreshOnVisible = false}: {
    intervalMs: number; onRefresh: () => void; enabled: boolean; refreshOnVisible?: boolean;
}): void {
    const callback = useEffectEvent(onRefresh);
    useEffect(() => {
        if (!enabled) return;
        let lastRun = Date.now();
        const tick = () => {
            if (document.visibilityState !== "visible" || Date.now() - lastRun < intervalMs) return;
            lastRun = Date.now();
            callback();
        };
        const timer = window.setInterval(tick, Math.min(intervalMs, 30_000));
        const visible = () => {
            if (document.visibilityState === "visible" && refreshOnVisible) {
                lastRun = Date.now();
                callback();
            }
        };
        document.addEventListener("visibilitychange", visible);
        return () => { window.clearInterval(timer); document.removeEventListener("visibilitychange", visible); };
    }, [enabled, intervalMs, refreshOnVisible]);
}
```

- [ ] **Step 4: Split refresh keys and prevent overlap**

In `App.tsx`, create:

```ts
const [liveRefreshKey, bumpLiveRefresh] = useReducer((value: number) => value + 1, 0);
const [forecastRefreshKey, bumpForecastRefresh] = useReducer((value: number) => value + 1, 0);
const [scheduleRefreshKey, bumpScheduleRefresh] = useReducer((value: number) => value + 1, 0);
useVisibilityPolling({intervalMs: 90_000, onRefresh: bumpLiveRefresh, enabled: !isOffline, refreshOnVisible: true});
useVisibilityPolling({intervalMs: 15 * 60_000, onRefresh: bumpForecastRefresh, enabled: !isOffline});
useVisibilityPolling({intervalMs: 4 * 60 * 60_000, onRefresh: bumpScheduleRefresh, enabled: !isOffline});
```

Pull-to-refresh calls `bumpLiveRefresh` only. Each data hook keeps an `inFlightRef`; a new request aborts the prior request for the same resource and never clears valid current-facility data before replacement succeeds.

- [ ] **Step 5: Run hook tests and verify GREEN**

Run: `npm run test:run -- src/app/hooks`

Expected: PASS with fake timers and no overlapping mock requests.

- [ ] **Step 6: Run Phase 6 regression checks**

Run: `npm run lint && npm run build && npm run test:run`

Expected: all commands exit 0.

- [ ] **Step 7: Commit Phase 6**

```bash
git add src/app/App.tsx src/app/hooks/useVisibilityPolling.ts src/app/hooks/useVisibilityPolling.test.tsx src/app/hooks/useLiveFacilityData.ts src/app/hooks/useLiveFacilityData.test.tsx src/app/hooks/useForecastData.ts src/app/hooks/useForecastData.test.tsx src/app/hooks/useFacilityHours.ts src/app/hooks/useFacilityHours.test.tsx src/app/hooks/usePullToRefresh.ts
git commit -m "feat: unify API validation and refresh behavior"
```
