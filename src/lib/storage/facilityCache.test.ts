import type {FacilityPayload} from "../types/facility";
import {
    CACHE_KEY,
    CACHE_MAX_AGE_MS,
    CACHE_VERSION,
    getFacilityCache,
} from "./facilityCache";

const payload: FacilityPayload = {
    facilityId: 1186,
    facilityName: "Nick",
    floors: {},
    locations: [],
};

const storeEntry = (entry: {version: number; cachedAt: number}) => {
    window.localStorage.setItem(CACHE_KEY, JSON.stringify({
        "1186": {...entry, payload},
    }));
};

describe("facility cache trust boundary", () => {
    it("uses the manual cache format version 3", () => {
        expect(CACHE_VERSION).toBe(3);
    });

    it("rejects a future cache timestamp", () => {
        vi.useFakeTimers();
        const now = Date.parse("2026-08-31T12:00:00Z");
        vi.setSystemTime(now);
        storeEntry({version: CACHE_VERSION, cachedAt: now + 1});

        expect(getFacilityCache(1186)).toBeNull();
    });

    it.each([
        ["nonfinite", Number.NaN],
        ["stale", Date.now() - CACHE_MAX_AGE_MS - 1],
    ])("rejects a %s cache timestamp", (_label, cachedAt) => {
        storeEntry({version: 3, cachedAt});

        expect(getFacilityCache(1186)).toBeNull();
    });

    it("accepts the exact freshness boundary", () => {
        vi.useFakeTimers();
        const now = Date.parse("2026-08-31T12:00:00Z");
        vi.setSystemTime(now);
        storeEntry({version: 3, cachedAt: now - CACHE_MAX_AGE_MS});

        expect(getFacilityCache(1186)?.payload).toEqual(payload);
    });
});
