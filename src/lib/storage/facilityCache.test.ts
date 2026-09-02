import type {FacilityPayload, Location} from "../types/facility";
import {
    CACHE_KEY,
    CACHE_MAX_AGE_MS,
    CACHE_VERSION,
    getFacilityCache,
    setFacilityCache,
} from "./facilityCache";

const location: Location = {
    facilityId: 1186,
    locationId: 5761,
    locationName: "Nick Power House",
    floor: 0,
    isClosed: false,
    currentCapacity: 30,
    maxCapacity: 50,
    lastUpdated: "2026-08-31T11:55:00Z",
    fetchedAt: "2026-08-31T12:00:00Z",
};

const payload: FacilityPayload = {
    facilityId: 1186,
    facilityName: "Nick",
    floors: {0: [location]},
    locations: [location],
    liveDataSource: "facility_api",
};

const storeEntry = (entry: unknown) => {
    window.localStorage.setItem(CACHE_KEY, JSON.stringify({
        "1186": entry,
    }));
};

const entryAt = (cachedAt: number, entryPayload: unknown = payload) => ({
    version: CACHE_VERSION,
    cachedAt,
    payload: entryPayload,
});

describe("facility cache trust boundary", () => {
    it("uses the manual cache format version 3", () => {
        expect(CACHE_VERSION).toBe(3);
    });

    it("rejects a future cache timestamp", () => {
        vi.useFakeTimers();
        const now = Date.parse("2026-08-31T12:00:00Z");
        vi.setSystemTime(now);
        storeEntry(entryAt(now + 1));

        expect(getFacilityCache(1186)).toBeNull();
    });

    it.each([
        ["nonfinite", Number.NaN],
        ["stale", Date.now() - CACHE_MAX_AGE_MS - 1],
    ])("rejects a %s cache timestamp", (_label, cachedAt) => {
        storeEntry(entryAt(cachedAt));

        expect(getFacilityCache(1186)).toBeNull();
    });

    it("rejects a schema-invalid cached payload", () => {
        storeEntry(entryAt(Date.now(), {bad: true}));

        expect(getFacilityCache(1186)).toBeNull();
    });

    it("rejects a cache entry stored under a different facility identity", () => {
        storeEntry(entryAt(Date.now(), {
            facilityId: 1656,
            facilityName: "Bakke",
            floors: {},
            locations: [],
            liveDataSource: "facility_api",
        }));

        expect(getFacilityCache(1186)).toBeNull();
    });

    it("rejects floors and locations that disagree about the same record", () => {
        storeEntry(entryAt(Date.now(), {
            ...payload,
            locations: [{...location, currentCapacity: 31}],
        }));

        expect(getFacilityCache(1186)).toBeNull();
    });

    it.each([
        ["missing", undefined],
        ["recursive cache", "cache"],
    ])("rejects %s upstream provenance", (_label, liveDataSource) => {
        storeEntry(entryAt(Date.now(), {...payload, liveDataSource}));

        expect(getFacilityCache(1186)).toBeNull();
    });

    it("accepts the exact freshness boundary", () => {
        vi.useFakeTimers();
        const now = Date.parse("2026-08-31T12:00:00Z");
        vi.setSystemTime(now);
        storeEntry(entryAt(now - CACHE_MAX_AGE_MS));

        expect(getFacilityCache(1186)?.payload).toEqual(payload);
    });

    it("does not write a payload for the wrong facility", () => {
        setFacilityCache(1186, {
            ...payload,
            facilityId: 1656,
            floors: {},
            locations: [],
        });

        expect(getFacilityCache(1186)).toBeNull();
    });
});
