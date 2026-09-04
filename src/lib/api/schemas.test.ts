import {describe, expect, it} from "vitest";
import {
    actualHoursResponseSchema,
    facilityCacheSchema,
    facilityScheduleSchema,
    forecastResponseSchema,
    liveCountsResponseSchema,
    pushAvailabilitySchema,
    pushCancelAllResponseSchema,
    pushCancelOneResponseSchema,
    pushPublicKeySchema,
    pushRuleListSchema,
    pushRuleResponseSchema,
} from "./schemas";

const canonicalLiveRow = {
    LocationId: 5761,
    IsClosed: false,
    LastCount: 47,
    LastUpdatedDateAndTime: "2026-08-31T11:59:00Z",
    FetchedAt: "2026-08-31T12:00:00Z",
};

const validRule = {
    id: 7,
    facilityId: 1186,
    sectionKey: "fitness floors",
    threshold: 40,
    createdAt: "2026-09-01T08:00:00-04:00",
    expiresAt: "2026-09-02T08:00:00-04:00",
    status: "pending",
};

const validLocation = {
    facilityId: 1186,
    locationId: 5761,
    locationName: "Nick Power House",
    floor: 0,
    isClosed: false,
    currentCapacity: 47,
    maxCapacity: 120,
    lastUpdated: "2026-08-31T11:59:00Z",
    fetchedAt: "2026-08-31T12:00:00Z",
};

const P256_GENERATOR_BYTES = [
    0x04,
    0x6b, 0x17, 0xd1, 0xf2, 0xe1, 0x2c, 0x42, 0x47,
    0xf8, 0xbc, 0xe6, 0xe5, 0x63, 0xa4, 0x40, 0xf2,
    0x77, 0x03, 0x7d, 0x81, 0x2d, 0xeb, 0x33, 0xa0,
    0xf4, 0xa1, 0x39, 0x45, 0xd8, 0x98, 0xc2, 0x96,
    0x4f, 0xe3, 0x42, 0xe2, 0xfe, 0x1a, 0x7f, 0x9b,
    0x8e, 0xe7, 0xeb, 0x4a, 0x7c, 0x0f, 0x9e, 0x16,
    0x2b, 0xce, 0x33, 0x57, 0x6b, 0x31, 0x5e, 0xce,
    0xcb, 0xb6, 0x40, 0x68, 0x37, 0xbf, 0x51, 0xf5,
] as const;

const base64UrlFromBytes = (bytes: readonly number[]): string => globalThis.btoa(
    String.fromCharCode(...bytes),
).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");

const validVapidPublicKey = base64UrlFromBytes(P256_GENERATOR_BYTES);

describe("liveCountsResponseSchema", () => {
    it("rejects malformed canonical rows instead of casting them", () => {
        expect(() => liveCountsResponseSchema.parse({
            ingestion: {
                lastSuccessfulFetchAt: null,
                ageSeconds: null,
                status: "unavailable",
            },
            rows: [{...canonicalLiveRow, LocationId: "not-an-id"}],
        })).toThrow();
    });

    it.each([
        [],
        {data: []},
        {
            ingestion: {
                lastSuccessfulFetchAt: null,
                ageSeconds: null,
                status: "unavailable",
            },
            rows: [],
        },
    ])("rejects an empty live payload: %#", (payload) => {
        expect(() => liveCountsResponseSchema.parse(payload)).toThrow();
    });

    it.each([
        ["array", [canonicalLiveRow]],
        ["data envelope", {data: [canonicalLiveRow]}],
    ])("forces supplied legacy FetchedAt to null for the %s shape", (_label, payload) => {
        const parsed = liveCountsResponseSchema.parse(payload);

        expect(parsed).toEqual({
            ingestion: {
                lastSuccessfulFetchAt: null,
                ageSeconds: null,
                status: "unavailable",
            },
            rows: [{...canonicalLiveRow, FetchedAt: null}],
        });
    });

    it("preserves a canonical row's explicit fetch provenance", () => {
        const parsed = liveCountsResponseSchema.parse({
            ingestion: {
                lastSuccessfulFetchAt: "2026-08-31T12:00:00Z",
                ageSeconds: 30,
                status: "healthy",
            },
            rows: [canonicalLiveRow],
        });

        expect(parsed.rows[0]?.FetchedAt).toBe("2026-08-31T12:00:00Z");
    });

    it("rejects contradictory ingestion health instead of inventing freshness", () => {
        expect(() => liveCountsResponseSchema.parse({
            ingestion: {
                lastSuccessfulFetchAt: null,
                ageSeconds: null,
                status: "healthy",
            },
            rows: [canonicalLiveRow],
        })).toThrow();
    });
});

describe("forecast and actual-hour schemas", () => {
    it("retains the real compact hour and spikeAdjusted fields", () => {
        const parsed = forecastResponseSchema.parse({
            facilityId: 1186,
            facilityName: "Nicholas Recreation Center",
            occupancyThresholds: {lowMax: 34, peakMin: 70},
            sectionOccupancyThresholds: {},
            locationOccupancyThresholds: {},
            weeklyForecast: [{
                dayName: "Monday",
                date: "2026-08-31",
                categories: [],
                totalHours: [{
                    hour: 10,
                    hourStart: "2026-08-31T10:00:00-05:00",
                    expectedCount: 20,
                    spikeAdjusted: true,
                }],
                avoidWindows: [],
                bestWindows: [],
                crowdBands: [],
            }],
        });

        expect(parsed.weeklyForecast[0]?.totalHours?.[0]).toMatchObject({
            hour: 10,
            spikeAdjusted: true,
        });
    });

    it("does not widen compact forecast hours to unrelated interval fields", () => {
        expect(() => forecastResponseSchema.parse({
            facilityId: 1186,
            facilityName: "Nick",
            weeklyForecast: [{
                dayName: "Monday",
                date: "2026-08-31",
                totalHours: [{
                    hour: 10,
                    hourStart: "2026-08-31T10:00:00-05:00",
                    expectedCount: 20,
                    expectedCountP90: 30,
                }],
            }],
        })).toThrow();
    });

    it("allows a low-coverage actual with a null actualCount", () => {
        const parsed = actualHoursResponseSchema.parse({
            facilityId: 1186,
            date: "2026-08-31",
            categories: [],
            totalHours: [{
                hourStart: "2026-08-31T10:00:00-05:00",
                observedCount: 4,
                observedCapacity: 10,
                expectedCapacity: 20,
                actualCoverage: 0.5,
                temporalCoverage: 1,
                coverageThreshold: 0.75,
                actualCount: null,
            }],
        });

        expect(parsed.totalHours[0]?.actualCount).toBeNull();
    });

    it("rejects claimed actual coverage that contradicts observed capacity", () => {
        expect(() => actualHoursResponseSchema.parse({
            facilityId: 1186,
            date: "2026-08-31",
            categories: [],
            totalHours: [{
                hourStart: "2026-08-31T10:00:00-05:00",
                observedCount: 1,
                observedCapacity: 1,
                expectedCapacity: 100,
                actualCoverage: 1,
                temporalCoverage: 1,
                coverageThreshold: 0.75,
                actualCount: 1,
                actualPct: 0.01,
            }],
        })).toThrow();
    });

    it.each([
        ["nonzero coverage with zero expected capacity", 1, 0, 0.0001],
        ["coverage that was not rounded to four decimals", 1, 100, 0.01001],
    ])("rejects %s", (_label, observedCapacity, expectedCapacity, actualCoverage) => {
        expect(() => actualHoursResponseSchema.parse({
            facilityId: 1186,
            date: "2026-08-31",
            categories: [],
            totalHours: [{
                hourStart: "2026-08-31T10:00:00-05:00",
                observedCount: 1,
                observedCapacity,
                expectedCapacity,
                actualCoverage,
                temporalCoverage: 1,
                coverageThreshold: 0.75,
                actualCount: null,
            }],
        })).toThrow();
    });

    it("accepts legitimate four-decimal backend coverage rounding", () => {
        const parsed = actualHoursResponseSchema.parse({
            facilityId: 1186,
            date: "2026-08-31",
            categories: [],
            totalHours: [{
                hourStart: "2026-08-31T10:00:00-05:00",
                observedCount: 1,
                observedCapacity: 1,
                expectedCapacity: 32,
                actualCoverage: 0.0312,
                temporalCoverage: 1,
                coverageThreshold: 0.75,
                actualCount: null,
            }],
        });

        expect(parsed.totalHours[0]?.actualCoverage).toBe(0.0312);
    });

    it("allows a closed or missing interval to remain unknown rather than zero", () => {
        const parsed = actualHoursResponseSchema.parse({
            facilityId: 1656,
            date: "2026-08-31",
            categories: [],
            totalHours: [{
                hourStart: "2026-08-31T10:00:00-05:00",
                observedCount: null,
                observedCapacity: 0,
                expectedCapacity: 200,
                actualCoverage: 0,
                temporalCoverage: 0,
                coverageThreshold: 0.75,
                actualCount: null,
            }],
        });

        expect(parsed.totalHours[0]).toMatchObject({
            observedCount: null,
            actualCount: null,
        });
    });

    it.each([
        ["low capacity coverage", {actualCoverage: 0.5, temporalCoverage: 1}],
        ["low temporal coverage", {actualCoverage: 1, temporalCoverage: 0.5}],
    ])("rejects a non-null actualCount for %s", (_label, coverage) => {
        expect(() => actualHoursResponseSchema.parse({
            facilityId: 1186,
            date: "2026-08-31",
            categories: [],
            totalHours: [{
                hourStart: "2026-08-31T10:00:00-05:00",
                observedCount: 4,
                observedCapacity: 10,
                expectedCapacity: 20,
                ...coverage,
                coverageThreshold: 0.75,
                actualCount: 4,
                actualPct: 0.2,
            }],
        })).toThrow();
    });
});

describe("facilityScheduleSchema", () => {
    const currentSchedule = {
        generatedAt: "2026-08-31T12:00:00Z",
        sourceSite: "https://recwell.wisc.edu",
        facilityId: 1186,
        facilityName: "Nick",
        slug: "nick",
        url: "https://recwell.wisc.edu/nick/",
        status: "ok",
        source: "wp_json",
        sourceModifiedGmt: "2026-08-31T11:45:00",
        sections: [{
            title: "Building Hours",
            rows: [{label: "Mon-Fri", hours: "6:00 am - 10:00 pm"}],
            note: null,
        }],
        sourceFetchedAt: "2026-08-31T12:00:00Z",
        lastSuccessfulAt: "2026-08-31T12:00:00Z",
        stale: false,
        error: null,
        errorCategory: null,
        updatedAt: "2026-08-31T12:00:00Z",
    };

    it("accepts a fresh schedule while allowing resolvedUrl to be absent", () => {
        expect(facilityScheduleSchema.parse(currentSchedule)).toEqual(currentSchedule);
    });

    it("accepts resolvedUrl without accepting unknown fields", () => {
        const futureSchedule = {
            ...currentSchedule,
            resolvedUrl: "https://recwell.wisc.edu/nick/",
        };

        expect(facilityScheduleSchema.parse(futureSchedule)).toEqual(futureSchedule);
        expect(() => facilityScheduleSchema.parse({...futureSchedule, privateDebug: "no"})).toThrow();
    });

    it("rejects malformed provenance metadata", () => {
        expect(() => facilityScheduleSchema.parse({
            ...currentSchedule,
            stale: "false",
            sourceFetchedAt: "not-a-timestamp",
        })).toThrow();
    });
});

describe("push response schemas", () => {
    it.each([
        [{
            apiAvailable: true,
            dbAvailable: false,
            alertsAvailable: false,
            reason: "push_rules_db_unavailable",
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: false,
            alertsAvailable: false,
            reason: "push_rules_db_unavailable",
        }],
        [{
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_vapid_unconfigured",
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_vapid_unconfigured",
        }],
        [{
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_identity_unconfigured",
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_identity_unconfigured",
        }],
        [{
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
        }],
    ])("accepts only a safe availability state and strips the backend marker", (wire, safe) => {
        expect(pushAvailabilitySchema.parse(wire)).toEqual(safe);
    });

    it.each([
        {
            apiAvailable: false,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "db",
        },
        {
            apiAvailable: true,
            dbAvailable: false,
            alertsAvailable: true,
            reason: "push_rules_db_unavailable",
            storeBackend: "db",
        },
        {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: null,
            storeBackend: "db",
        },
        {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "memory",
        },
    ])("rejects an inconsistent availability state: %#", (wire) => {
        expect(() => pushAvailabilitySchema.parse(wire)).toThrow();
    });

    it("accepts a canonical uncompressed P-256 public key", () => {
        expect(pushPublicKeySchema.parse({publicKey: validVapidPublicKey})).toEqual({
            publicKey: validVapidPublicKey,
        });
    });

    it("rejects a canonical uncompressed point that is not on P-256", () => {
        const offCurvePublicKey = base64UrlFromBytes([
            ...P256_GENERATOR_BYTES.slice(0, -1),
            0xf4,
        ]);

        expect(() => pushPublicKeySchema.parse({publicKey: offCurvePublicKey})).toThrow();
    });

    it.each([
        ["invalid characters", "not canonical!"],
        ["padding", `${validVapidPublicKey}=`],
        ["wrong decoded length", "BAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0-Pw"],
        ["wrong point prefix", "AwECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8gISIjJCUmJygpKissLS4vMDEyMzQ1Njc4OTo7PD0-P0A"],
        ["noncanonical trailing bits", `${validVapidPublicKey.slice(0, -1)}B`],
    ])("rejects a VAPID public key with %s", (_label, publicKey) => {
        expect(() => pushPublicKeySchema.parse({publicKey})).toThrow();
    });

    it("accepts a strict subscribe response with a canonical pending rule", () => {
        expect(pushRuleResponseSchema.parse({
            status: "ok",
            created: true,
            rule: validRule,
        })).toEqual({status: "ok", created: true, rule: validRule});
    });

    it.each([
        ["unsafe id", {...validRule, id: Number.MAX_SAFE_INTEGER + 1}],
        ["noncanonical section", {...validRule, sectionKey: "Fitness  Floors"}],
        ["invalid calendar date", {...validRule, createdAt: "2026-02-30T12:00:00Z"}],
        ["timezone-less date", {...validRule, createdAt: "2026-09-01T12:00:00"}],
        ["non-increasing expiry", {...validRule, expiresAt: validRule.createdAt}],
    ])("rejects a rule with %s", (_label, rule) => {
        expect(() => pushRuleResponseSchema.parse({
            status: "ok",
            created: false,
            rule,
        })).toThrow();
    });

    it("rejects duplicate rule IDs atomically", () => {
        expect(() => pushRuleListSchema.parse({
            status: "ok",
            rules: [validRule, {...validRule}],
        })).toThrow();
    });

    it("accepts only the exact cancel-one and bounded cancel-all counts", () => {
        expect(pushCancelOneResponseSchema.parse({status: "ok", cancelled: 1})).toEqual({
            status: "ok",
            cancelled: 1,
        });
        expect(() => pushCancelOneResponseSchema.parse({status: "ok", cancelled: 0})).toThrow();
        expect(pushCancelAllResponseSchema.parse({status: "ok", cancelled: 10})).toEqual({
            status: "ok",
            cancelled: 10,
        });
        expect(() => pushCancelAllResponseSchema.parse({
            status: "ok",
            cancelled: Number.MAX_SAFE_INTEGER + 1,
        })).toThrow();
    });
});

describe("facilityCacheSchema", () => {
    const validCache = {
        version: 3,
        cachedAt: 1_800_000_000_000,
        payload: {
            facilityId: 1186,
            facilityName: "Nick",
            floors: {"0": [validLocation]},
            locations: [validLocation],
            liveDataSource: "facility_api",
        },
    };

    it("preserves a version 3 payload's freshness and provenance", () => {
        expect(facilityCacheSchema.parse(validCache)).toEqual(validCache);
    });

    it.each(["tomorrow", Number.NaN, Number.POSITIVE_INFINITY])(
        "rejects invalid cachedAt %s",
        (cachedAt) => {
            expect(() => facilityCacheSchema.parse({...validCache, cachedAt})).toThrow();
        },
    );

    it("rejects cross-facility location data", () => {
        const foreignLocation = {...validLocation, facilityId: 1656};
        expect(() => facilityCacheSchema.parse({
            ...validCache,
            payload: {
                ...validCache.payload,
                floors: {"0": [foreignLocation]},
                locations: [foreignLocation],
            },
        })).toThrow();
    });

    it("rejects a location stored beneath the wrong floor", () => {
        expect(() => facilityCacheSchema.parse({
            ...validCache,
            payload: {
                ...validCache.payload,
                floors: {"1": [validLocation]},
            },
        })).toThrow();
    });

    it.each([
        ["current capacity", {currentCapacity: 48}],
        ["fetch provenance", {fetchedAt: "2026-08-31T12:01:00Z"}],
    ])("rejects divergent %s for the same location ID", (_label, change) => {
        expect(() => facilityCacheSchema.parse({
            ...validCache,
            payload: {
                ...validCache.payload,
                locations: [{...validLocation, ...change}],
            },
        })).toThrow();
    });
});
