import {z} from "zod";

const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER;
const MAX_PUSH_RULE_TTL_MS = 604_800_000;
const MAX_ACTIVE_PUSH_RULES = 10;
const COVERAGE_DECIMAL_SCALE = 10_000;
const COVERAGE_ROUNDING_TOLERANCE = 0.5 / COVERAGE_DECIMAL_SCALE;
const FLOAT_COMPARISON_EPSILON = 1e-12;

const daysInMonth = (year: number, month: number): number => {
    if (month === 2) {
        const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
        return leapYear ? 29 : 28;
    }
    return [4, 6, 9, 11].includes(month) ? 30 : 31;
};

const DATE_PATTERN = /^(\d{4})-(\d{2})-(\d{2})$/;
const EXPLICIT_TIMESTAMP_PATTERN = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?(?:Z|[+-](\d{2}):(\d{2}))$/;
const LOCAL_TIMESTAMP_PATTERN = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d{1,9})?$/;

const isValidDateParts = (year: number, month: number, day: number): boolean => (
    year >= 1
    && month >= 1
    && month <= 12
    && day >= 1
    && day <= daysInMonth(year, month)
);

const isCalendarDate = (value: string): boolean => {
    const match = DATE_PATTERN.exec(value);
    if (!match) return false;
    return isValidDateParts(Number(match[1]), Number(match[2]), Number(match[3]));
};

const isCalendarDateTime = (value: string, explicitZone: boolean): boolean => {
    const match = (explicitZone ? EXPLICIT_TIMESTAMP_PATTERN : LOCAL_TIMESTAMP_PATTERN).exec(value);
    if (!match) return false;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const hour = Number(match[4]);
    const minute = Number(match[5]);
    const second = Number(match[6]);
    const offsetHour = explicitZone && match[7] !== undefined ? Number(match[7]) : 0;
    const offsetMinute = explicitZone && match[8] !== undefined ? Number(match[8]) : 0;
    if (
        !isValidDateParts(year, month, day)
        || hour > 23
        || minute > 59
        || second > 59
        || offsetHour > 23
        || offsetMinute > 59
    ) {
        return false;
    }

    const parseTarget = explicitZone ? value : `${value}Z`;
    return Number.isFinite(Date.parse(parseTarget));
};

const explicitIsoDateTimeSchema = z.string()
    .min(20)
    .max(64)
    .refine((value) => isCalendarDateTime(value, true), "timestamp must include a valid explicit offset");

const localIsoDateTimeSchema = z.string()
    .min(19)
    .max(48)
    .refine((value) => isCalendarDateTime(value, false), "timestamp must be a valid local datetime");

const isoDateSchema = z.string()
    .length(10)
    .refine(isCalendarDate, "date must be a valid YYYY-MM-DD value");

const nullableIsoDateTimeSchema = explicitIsoDateTimeSchema.nullable();
const finiteNonnegativeSchema = z.number().finite().nonnegative();
const safeNonnegativeIntegerSchema = z.number().int().min(0).max(MAX_SAFE_INTEGER);
const safePositiveIntegerSchema = z.number().int().min(1).max(MAX_SAFE_INTEGER);
const unitIntervalSchema = z.number().finite().min(0).max(1);

export const facilityIdSchema = z.union([z.literal(1186), z.literal(1656)]);

export const liveLocationRowSchema = z.object({
    LocationId: safePositiveIntegerSchema,
    IsClosed: z.boolean().nullable(),
    LastCount: safeNonnegativeIntegerSchema.nullable(),
    LastUpdatedDateAndTime: nullableIsoDateTimeSchema.optional(),
    FetchedAt: nullableIsoDateTimeSchema,
}).strict();

const legacyLiveLocationRowSchema = z.object({
    LocationId: safePositiveIntegerSchema,
    IsClosed: z.boolean().nullable(),
    LastCount: safeNonnegativeIntegerSchema.nullable(),
    LastUpdatedDateAndTime: nullableIsoDateTimeSchema.optional(),
    FetchedAt: nullableIsoDateTimeSchema.optional(),
}).strict().transform((row) => ({...row, FetchedAt: null}));

export const ingestionHealthSchema = z.object({
    lastSuccessfulFetchAt: nullableIsoDateTimeSchema,
    ageSeconds: safeNonnegativeIntegerSchema.nullable(),
    status: z.enum(["healthy", "stale", "unavailable"]),
}).strict().superRefine((health, context) => {
    const isUnavailable = health.status === "unavailable";
    const hasFreshnessEvidence = health.lastSuccessfulFetchAt !== null && health.ageSeconds !== null;
    if ((isUnavailable && hasFreshnessEvidence) || (!isUnavailable && !hasFreshnessEvidence)) {
        context.addIssue({
            code: "custom",
            message: "ingestion status and freshness evidence are inconsistent",
        });
    }
    if (isUnavailable && (health.lastSuccessfulFetchAt !== null || health.ageSeconds !== null)) {
        context.addIssue({
            code: "custom",
            message: "unavailable ingestion cannot claim freshness evidence",
        });
    }
});

const canonicalLiveCountsSchema = z.object({
    ingestion: ingestionHealthSchema,
    rows: z.array(liveLocationRowSchema).min(1),
}).strict();

const unavailableIngestion = () => ({
    lastSuccessfulFetchAt: null,
    ageSeconds: null,
    status: "unavailable" as const,
});

const legacyLiveRowsSchema = z.array(legacyLiveLocationRowSchema).min(1);

export const liveCountsResponseSchema = z.union([
    canonicalLiveCountsSchema,
    legacyLiveRowsSchema.transform((rows) => ({
        ingestion: unavailableIngestion(),
        rows,
    })),
    z.object({data: legacyLiveRowsSchema}).strict().transform(({data}) => ({
        ingestion: unavailableIngestion(),
        rows: data,
    })),
]);

export const occupancyThresholdSchema = z.object({
    lowMax: z.number().finite().min(0).max(99),
    peakMin: z.number().finite().min(1).max(100),
}).strict().refine(
    ({lowMax, peakMin}) => lowMax < peakMin,
    "lowMax must be below peakMin",
);

export const forecastHourSchema = z.object({
    hour: z.number().int().min(0).max(23).optional(),
    hourStart: explicitIsoDateTimeSchema,
    expectedCount: safeNonnegativeIntegerSchema,
    expectedPct: unitIntervalSchema.nullable().optional(),
    actualCount: safeNonnegativeIntegerSchema.nullable().optional(),
    actualPct: unitIntervalSchema.nullable().optional(),
    actualSampleCount: safeNonnegativeIntegerSchema.optional(),
    actualCoverage: unitIntervalSchema.nullable().optional(),
    spikeAdjusted: z.boolean().optional(),
}).strict();

const forecastWindowSchema = z.object({
    start: explicitIsoDateTimeSchema,
    end: explicitIsoDateTimeSchema,
    startHour: z.number().int().min(0).max(23).optional(),
    endHour: z.number().int().min(0).max(24).optional(),
    windowHours: finiteNonnegativeSchema.optional(),
    expectedTotal: finiteNonnegativeSchema.optional(),
    expectedAvg: finiteNonnegativeSchema.optional(),
    sampleCountMin: safeNonnegativeIntegerSchema.optional(),
}).strict().superRefine((window, context) => {
    if (Date.parse(window.end) <= Date.parse(window.start)) {
        context.addIssue({code: "custom", message: "forecast window must end after it starts"});
    }
});

const forecastBandSchema = z.object({
    start: explicitIsoDateTimeSchema,
    end: explicitIsoDateTimeSchema,
    level: z.enum(["low", "medium", "peak"]),
}).strict().superRefine((band, context) => {
    if (Date.parse(band.end) <= Date.parse(band.start)) {
        context.addIssue({code: "custom", message: "forecast band must end after it starts"});
    }
});

const forecastCategorySchema = z.object({
    key: z.string().trim().min(1).max(100),
    title: z.string().trim().min(1).max(160),
    maxCapacity: safeNonnegativeIntegerSchema.nullable().optional(),
    hours: z.array(forecastHourSchema),
}).strict();

const forecastDaySchema = z.object({
    dayName: z.string().trim().min(1).max(32),
    date: isoDateSchema,
    categories: z.array(forecastCategorySchema).optional(),
    totalHours: z.array(forecastHourSchema).optional(),
    avoidWindows: z.array(forecastWindowSchema).optional(),
    bestWindows: z.array(forecastWindowSchema).optional(),
    crowdBands: z.array(forecastBandSchema).optional(),
}).strict();

const thresholdRecordKeySchema = z.string().trim().min(1).max(100);
const locationThresholdKeySchema = z.string().regex(/^[1-9][0-9]*$/).refine(
    (value) => Number.isSafeInteger(Number(value)),
    "location threshold key must be a safe positive integer",
);

export const forecastResponseSchema = z.object({
    facilityId: facilityIdSchema,
    facilityName: z.string().trim().min(1).max(160),
    forecastDayStartHour: z.number().int().min(0).max(23).optional(),
    forecastDayEndHour: z.number().int().min(0).max(23).optional(),
    occupancyThresholds: occupancyThresholdSchema.nullable().optional(),
    sectionOccupancyThresholds: z.record(
        thresholdRecordKeySchema,
        occupancyThresholdSchema,
    ).nullable().optional(),
    locationOccupancyThresholds: z.record(
        locationThresholdKeySchema,
        occupancyThresholdSchema,
    ).nullable().optional(),
    weeklyForecast: z.array(forecastDaySchema),
}).strict();

export const actualHourSchema = z.object({
    hourStart: explicitIsoDateTimeSchema,
    observedCount: safeNonnegativeIntegerSchema.nullable(),
    observedCapacity: safeNonnegativeIntegerSchema,
    expectedCapacity: safeNonnegativeIntegerSchema,
    actualCoverage: unitIntervalSchema,
    temporalCoverage: unitIntervalSchema,
    coverageThreshold: unitIntervalSchema,
    actualCount: safeNonnegativeIntegerSchema.nullable(),
    actualPct: unitIntervalSchema.nullable().optional(),
}).strict().superRefine((hour, context) => {
    if (
        (hour.observedCount === null && hour.observedCapacity !== 0)
        || (hour.observedCount !== null && hour.observedCapacity === 0)
    ) {
        context.addIssue({
            code: "custom",
            message: "observed count and capacity are inconsistent",
        });
    }

    const rawActualCoverage = hour.expectedCapacity > 0
        ? Math.min(1, hour.observedCapacity / hour.expectedCapacity)
        : 0;
    const roundedActualCoverage = Math.round(
        hour.actualCoverage * COVERAGE_DECIMAL_SCALE,
    ) / COVERAGE_DECIMAL_SCALE;
    const hasAtMostFourDecimals = Math.abs(
        hour.actualCoverage - roundedActualCoverage,
    ) <= FLOAT_COMPARISON_EPSILON;
    const matchesObservedCapacity = Math.abs(
        hour.actualCoverage - rawActualCoverage,
    ) <= COVERAGE_ROUNDING_TOLERANCE + FLOAT_COMPARISON_EPSILON;
    if (!hasAtMostFourDecimals || !matchesObservedCapacity) {
        context.addIssue({
            code: "custom",
            message: "actual coverage does not match observed and expected capacity",
        });
    }

    const qualified = (
        hour.observedCount !== null
        && hour.expectedCapacity > 0
        && hour.actualCoverage >= hour.coverageThreshold
        && hour.temporalCoverage >= hour.coverageThreshold
    );
    if (qualified && hour.actualCount !== hour.observedCount) {
        context.addIssue({
            code: "custom",
            message: "qualified actual count must retain the observed count",
        });
    }
    if (!qualified && hour.actualCount !== null) {
        context.addIssue({
            code: "custom",
            message: "unqualified actual data must remain null",
        });
    }
    if (hour.actualCount === null && hour.actualPct !== undefined && hour.actualPct !== null) {
        context.addIssue({
            code: "custom",
            message: "unqualified actual data cannot include a percentage",
        });
    }
});

const actualCategorySchema = z.object({
    key: z.string().trim().min(1).max(100),
    title: z.string().trim().min(1).max(160),
    hours: z.array(actualHourSchema),
}).strict();

export const actualHoursResponseSchema = z.object({
    facilityId: facilityIdSchema,
    date: isoDateSchema,
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

const scheduleUrlSchema = z.string().url().max(2_048);
const scheduleErrorCategorySchema = z.enum([
    "anti_bot",
    "upstream_timeout",
    "upstream_http",
    "wp_payload_invalid",
    "parse_empty",
    "schema_invalid",
    "io_error",
]);

export const facilityScheduleSchema = z.object({
    generatedAt: nullableIsoDateTimeSchema,
    sourceSite: scheduleUrlSchema.nullable(),
    facilityId: facilityIdSchema,
    facilityName: z.string().trim().min(1).max(160),
    slug: z.string().trim().min(1).max(80),
    url: scheduleUrlSchema,
    resolvedUrl: scheduleUrlSchema.nullable().optional(),
    status: z.enum(["ok", "stale", "error"]),
    source: z.enum(["direct_html", "wp_json"]).nullable(),
    sourceModifiedGmt: localIsoDateTimeSchema.nullable().optional(),
    sections: z.array(scheduleSectionSchema),
    sourceFetchedAt: nullableIsoDateTimeSchema.optional(),
    lastSuccessfulAt: nullableIsoDateTimeSchema.optional(),
    stale: z.boolean().optional(),
    error: z.string().trim().max(240).nullable(),
    errorCategory: scheduleErrorCategorySchema.nullable().optional(),
    updatedAt: nullableIsoDateTimeSchema,
}).strict();

const BASE64URL_PATTERN = /^[A-Za-z0-9_-]+$/;
const VAPID_PUBLIC_KEY_BYTES = 65;
const bigIntFromHexFragments = (fragments: readonly string[]): bigint => BigInt(
    `0x${fragments.join("")}`,
);
const P256_FIELD_PRIME = bigIntFromHexFragments([
    "ffffffff", "00000001", "00000000", "00000000",
    "00000000", "ffffffff", "ffffffff", "ffffffff",
]);
const P256_CURVE_B = bigIntFromHexFragments([
    "5ac635d8", "aa3a93e7", "b3ebbd55", "769886bc",
    "651d06b0", "cc53b0f6", "3bce3c3e", "27d2604b",
]);

const decodedCoordinate = (decoded: string, start: number, end: number): bigint => {
    let coordinate = 0n;
    for (let index = start; index < end; index += 1) {
        coordinate = (coordinate << 8n) | BigInt(decoded.charCodeAt(index));
    }
    return coordinate;
};

const p256FieldValue = (value: bigint): bigint => (
    ((value % P256_FIELD_PRIME) + P256_FIELD_PRIME) % P256_FIELD_PRIME
);

const isP256CurvePoint = (decoded: string): boolean => {
    const x = decodedCoordinate(decoded, 1, 33);
    const y = decodedCoordinate(decoded, 33, 65);
    if (x >= P256_FIELD_PRIME || y >= P256_FIELD_PRIME) return false;

    const left = (y * y) % P256_FIELD_PRIME;
    const xSquared = (x * x) % P256_FIELD_PRIME;
    const right = p256FieldValue(
        ((xSquared * x) % P256_FIELD_PRIME) - (3n * x) + P256_CURVE_B,
    );
    return left === right;
};

const isCanonicalVapidPublicKey = (value: string): boolean => {
    const paddingLength = (4 - (value.length % 4)) % 4;
    try {
        const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
        const decoded = globalThis.atob(`${base64}${"=".repeat(paddingLength)}`);
        if (
            decoded.length !== VAPID_PUBLIC_KEY_BYTES
            || decoded.charCodeAt(0) !== 0x04
        ) {
            return false;
        }
        const canonical = globalThis.btoa(decoded)
            .replace(/\+/g, "-")
            .replace(/\//g, "_")
            .replace(/=+$/, "");
        return canonical === value && isP256CurvePoint(decoded);
    } catch {
        return false;
    }
};

export const pushPublicKeySchema = z.object({
    publicKey: z.string()
        .length(87)
        .regex(BASE64URL_PATTERN)
        .refine(isCanonicalVapidPublicKey, "public key must be canonical uncompressed P-256"),
}).strict();

const pushAvailabilityWireSchema = z.object({
    apiAvailable: z.literal(true),
    dbAvailable: z.boolean(),
    alertsAvailable: z.boolean(),
    reason: z.enum([
        "push_rules_db_unavailable",
        "push_vapid_unconfigured",
        "push_identity_unconfigured",
    ]).nullable(),
    storeBackend: z.literal("db"),
}).strict().superRefine((availability, context) => {
    const validState = (
        (
            availability.dbAvailable === false
            && availability.alertsAvailable === false
            && availability.reason === "push_rules_db_unavailable"
        )
        || (
            availability.dbAvailable === true
            && availability.alertsAvailable === false
            && (
                availability.reason === "push_vapid_unconfigured"
                || availability.reason === "push_identity_unconfigured"
            )
        )
        || (
            availability.dbAvailable === true
            && availability.alertsAvailable === true
            && availability.reason === null
        )
    );
    if (!validState) {
        context.addIssue({code: "custom", message: "push availability state is inconsistent"});
    }
});

export const pushAvailabilitySchema = pushAvailabilityWireSchema.transform((availability) => ({
    apiAvailable: availability.apiAvailable,
    dbAvailable: availability.dbAvailable,
    alertsAvailable: availability.alertsAvailable,
    reason: availability.reason,
}));

const canonicalSectionKeySchema = z.string().min(1).max(80).refine(
    (value) => value.trim().toLowerCase().replace(/\s+/g, " ") === value,
    "section key must be canonical",
);

export const pushRuleSchema = z.object({
    id: safePositiveIntegerSchema,
    facilityId: facilityIdSchema,
    sectionKey: canonicalSectionKeySchema,
    threshold: z.number().int().min(1).max(100),
    createdAt: explicitIsoDateTimeSchema,
    expiresAt: explicitIsoDateTimeSchema,
    status: z.literal("pending"),
}).strict().superRefine((rule, context) => {
    const createdAt = Date.parse(rule.createdAt);
    const expiresAt = Date.parse(rule.expiresAt);
    if (expiresAt <= createdAt) {
        context.addIssue({code: "custom", message: "push rule expiry must follow creation"});
    } else if (expiresAt - createdAt > MAX_PUSH_RULE_TTL_MS) {
        context.addIssue({code: "custom", message: "push rule expiry exceeds the maximum TTL"});
    }
});

export const pushRuleResponseSchema = z.object({
    status: z.literal("ok"),
    created: z.boolean(),
    rule: pushRuleSchema,
}).strict();

export const pushRuleListSchema = z.object({
    status: z.literal("ok"),
    rules: z.array(pushRuleSchema).max(MAX_ACTIVE_PUSH_RULES),
}).strict().superRefine((response, context) => {
    const ids = new Set<number>();
    for (const rule of response.rules) {
        if (ids.has(rule.id)) {
            context.addIssue({code: "custom", message: "push rule IDs must be unique"});
            return;
        }
        ids.add(rule.id);
    }
});

export const pushCancelOneResponseSchema = z.object({
    status: z.literal("ok"),
    cancelled: z.literal(1),
}).strict();

export const pushCancelAllResponseSchema = z.object({
    status: z.literal("ok"),
    cancelled: safeNonnegativeIntegerSchema.max(MAX_ACTIVE_PUSH_RULES),
}).strict();

export const pushCancelResponseSchema = pushCancelOneResponseSchema;

const locationSchema = z.object({
    facilityId: facilityIdSchema,
    locationId: safePositiveIntegerSchema,
    locationName: z.string().trim().min(1).max(160),
    floor: safeNonnegativeIntegerSchema,
    isClosed: z.boolean().nullable(),
    currentCapacity: safeNonnegativeIntegerSchema.nullable(),
    maxCapacity: safeNonnegativeIntegerSchema.nullable(),
    lastUpdated: nullableIsoDateTimeSchema,
    fetchedAt: nullableIsoDateTimeSchema,
}).strict();

type CacheLocation = z.infer<typeof locationSchema>;

const cacheLocationsMatch = (left: CacheLocation, right: CacheLocation): boolean => (
    left.facilityId === right.facilityId
    && left.locationId === right.locationId
    && left.locationName === right.locationName
    && left.floor === right.floor
    && left.isClosed === right.isClosed
    && left.currentCapacity === right.currentCapacity
    && left.maxCapacity === right.maxCapacity
    && left.lastUpdated === right.lastUpdated
    && left.fetchedAt === right.fetchedAt
);

const floorKeySchema = z.string().regex(/^(?:0|[1-9][0-9]*)$/).refine(
    (value) => Number.isSafeInteger(Number(value)),
    "floor key must be a safe non-negative integer",
);

const facilityPayloadSchema = z.object({
    facilityId: facilityIdSchema,
    facilityName: z.string().trim().min(1).max(160),
    floors: z.record(floorKeySchema, z.array(locationSchema)),
    locations: z.array(locationSchema),
    liveDataSource: z.enum(["facility_api", "fallback_api", "cache"]).optional(),
}).strict().superRefine((payload, context) => {
    const flattenedIds: number[] = [];
    for (const [floorKey, locations] of Object.entries(payload.floors)) {
        const floor = Number(floorKey);
        for (const location of locations) {
            flattenedIds.push(location.locationId);
            if (location.facilityId !== payload.facilityId || location.floor !== floor) {
                context.addIssue({
                    code: "custom",
                    message: "cached floor location does not match its facility and floor",
                });
            }
        }
    }

    const listedIds = payload.locations.map((location) => location.locationId);
    const listedById = new Map(
        payload.locations.map((location) => [location.locationId, location]),
    );
    if (payload.locations.some((location) => location.facilityId !== payload.facilityId)) {
        context.addIssue({
            code: "custom",
            message: "cached location does not match its facility",
        });
    }
    if (new Set(flattenedIds).size !== flattenedIds.length || new Set(listedIds).size !== listedIds.length) {
        context.addIssue({code: "custom", message: "cached location IDs must be unique"});
    }
    const sortedFlattened = [...flattenedIds].sort((left, right) => left - right);
    const sortedListed = [...listedIds].sort((left, right) => left - right);
    if (
        sortedFlattened.length !== sortedListed.length
        || sortedFlattened.some((locationId, index) => locationId !== sortedListed[index])
    ) {
        context.addIssue({
            code: "custom",
            message: "cached floors and flat locations must describe the same locations",
        });
    }
    if (Object.values(payload.floors).some((locations) => (
        locations.some((location) => {
            const listedLocation = listedById.get(location.locationId);
            return listedLocation === undefined
                || !cacheLocationsMatch(location, listedLocation);
        })
    ))) {
        context.addIssue({
            code: "custom",
            message: "cached floors and flat locations must contain identical location records",
        });
    }
});

export const facilityCacheSchema = z.object({
    version: z.literal(3),
    cachedAt: safeNonnegativeIntegerSchema,
    payload: facilityPayloadSchema,
}).strict();

export type LiveCountsResponse = z.infer<typeof liveCountsResponseSchema>;
export type ForecastResponse = z.infer<typeof forecastResponseSchema>;
export type ActualHoursResponse = z.infer<typeof actualHoursResponseSchema>;
export type FacilityScheduleResponse = z.infer<typeof facilityScheduleSchema>;
export type PushAvailability = z.infer<typeof pushAvailabilitySchema>;
export type PushRule = z.infer<typeof pushRuleSchema>;
export type FacilityCache = z.infer<typeof facilityCacheSchema>;
