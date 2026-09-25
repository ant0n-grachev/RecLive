import type {Location} from "../../lib/types/facility";
import parityFixture from "../../../tests/fixtures/occupancy_summary_parity.json";
import {computeOccupancySummary} from "./computeOccupancySummary";

const NOW = Date.parse("2026-08-31T12:00:00Z");

const location = (overrides: Partial<Location> = {}): Location => ({
    facilityId: 1186,
    locationId: 1,
    locationName: "Test location",
    floor: 1,
    isClosed: false,
    currentCapacity: 20,
    maxCapacity: 100,
    lastUpdated: "2026-08-31T11:55:00Z",
    fetchedAt: "2026-08-31T12:00:00Z",
    ...overrides,
});

describe("computeOccupancySummary", () => {
    it.each(parityFixture.cases)("matches the shared $name cross-layer fixture", (fixtureCase) => {
        const locations = fixtureCase.locations.map((entry): Location => ({
            facilityId: 1186,
            locationId: entry.locationId,
            locationName: `Fixture ${entry.locationId}`,
            floor: 1,
            isClosed: entry.row?.isClosed ?? null,
            currentCapacity: entry.row?.currentCapacity ?? null,
            maxCapacity: entry.maxCapacity,
            lastUpdated: null,
            fetchedAt: entry.row?.fetchedAt ?? null,
        }));

        const summary = computeOccupancySummary(locations, {nowMs: Date.parse(parityFixture.now)});

        expect(summary).toMatchObject(fixtureCase.expected);
        expect(summary.status === "live" && summary.coverage >= 0.8).toBe(
            fixtureCase.evaluatorEligible
        );
    });

    it("includes an observation at the exact ten-minute freshness boundary", () => {
        const result = computeOccupancySummary([
            location({fetchedAt: "2026-08-31T11:50:00Z"}),
        ], {nowMs: NOW});

        expect(result).toEqual({
            count: 20,
            observedCapacity: 100,
            expectedOpenCapacity: 100,
            coverage: 1,
            percent: 20,
            observedLocations: 1,
            expectedLocations: 1,
            latestFetchedAt: "2026-08-31T11:50:00Z",
            oldestFetchedAt: "2026-08-31T11:50:00Z",
            status: "live",
        });
    });

    it("does not turn a just-stale room into a zero observation", () => {
        const result = computeOccupancySummary([
            location({locationId: 1, currentCapacity: 20, fetchedAt: "2026-08-31T12:00:00Z"}),
            location({locationId: 2, currentCapacity: 80, fetchedAt: "2026-08-31T11:49:59.999Z"}),
        ], {nowMs: NOW});

        expect(result).toEqual({
            count: 20,
            observedCapacity: 100,
            expectedOpenCapacity: 200,
            coverage: 0.5,
            percent: 20,
            observedLocations: 1,
            expectedLocations: 2,
            latestFetchedAt: "2026-08-31T12:00:00Z",
            oldestFetchedAt: "2026-08-31T12:00:00Z",
            status: "partial",
        });
    });

    it.each([
        ["future", "2026-08-31T12:00:00.001Z"],
        ["missing", null],
        ["invalid", "not-a-timestamp"],
    ])("treats a %s fetch timestamp as untrusted", (_label, fetchedAt) => {
        const result = computeOccupancySummary([
            location({fetchedAt}),
        ], {nowMs: NOW});

        expect(result).toMatchObject({
            count: null,
            observedCapacity: 0,
            expectedOpenCapacity: 100,
            coverage: 0,
            percent: null,
            observedLocations: 0,
            expectedLocations: 1,
            latestFetchedAt: null,
            oldestFetchedAt: null,
            status: "insufficient",
        });
    });

    it("does not treat a null closure state as a confirmed open observation", () => {
        const result = computeOccupancySummary([
            location({isClosed: null}),
        ], {nowMs: NOW});

        expect(result).toEqual({
            count: null,
            observedCapacity: 0,
            expectedOpenCapacity: 100,
            coverage: 0,
            percent: null,
            observedLocations: 0,
            expectedLocations: 1,
            latestFetchedAt: null,
            oldestFetchedAt: null,
            status: "insufficient",
        });
    });

    it.each([
        ["missing", null],
        ["negative", -1],
        ["NaN", Number.NaN],
        ["positive infinity", Number.POSITIVE_INFINITY],
        ["negative infinity", Number.NEGATIVE_INFINITY],
    ])("rejects a %s count instead of converting it to zero", (_label, currentCapacity) => {
        const result = computeOccupancySummary([
            location({currentCapacity}),
        ], {nowMs: NOW});

        expect(result).toMatchObject({
            count: null,
            observedCapacity: 0,
            expectedOpenCapacity: 100,
            coverage: 0,
            percent: null,
            observedLocations: 0,
            expectedLocations: 1,
            status: "insufficient",
        });
    });

    it.each([
        ["missing", null],
        ["zero", 0],
        ["negative", -1],
        ["NaN", Number.NaN],
        ["positive infinity", Number.POSITIVE_INFINITY],
        ["negative infinity", Number.NEGATIVE_INFINITY],
    ])("does not create an expected or observed location from a %s capacity", (_label, maxCapacity) => {
        const result = computeOccupancySummary([
            location({maxCapacity}),
        ], {nowMs: NOW});

        expect(result).toEqual({
            count: null,
            observedCapacity: 0,
            expectedOpenCapacity: 0,
            coverage: 0,
            percent: null,
            observedLocations: 0,
            expectedLocations: 0,
            latestFetchedAt: null,
            oldestFetchedAt: null,
            status: "unknown",
        });
    });

    it("returns unknown for an empty expected set", () => {
        expect(computeOccupancySummary([], {nowMs: NOW})).toEqual({
            count: null,
            observedCapacity: 0,
            expectedOpenCapacity: 0,
            coverage: 0,
            percent: null,
            observedLocations: 0,
            expectedLocations: 0,
            latestFetchedAt: null,
            oldestFetchedAt: null,
            status: "unknown",
        });
    });

    it("excludes only fresh confirmed closures from expected capacity", () => {
        const result = computeOccupancySummary([
            location({locationId: 1, isClosed: true, fetchedAt: "2026-08-31T11:59:00Z"}),
            location({locationId: 2, isClosed: true, fetchedAt: "2026-08-31T11:49:59Z"}),
            location({locationId: 3, currentCapacity: 30, fetchedAt: "2026-08-31T11:58:00Z"}),
        ], {nowMs: NOW});

        expect(result).toEqual({
            count: 30,
            observedCapacity: 100,
            expectedOpenCapacity: 200,
            coverage: 0.5,
            percent: 30,
            observedLocations: 1,
            expectedLocations: 2,
            latestFetchedAt: "2026-08-31T11:59:00Z",
            oldestFetchedAt: "2026-08-31T11:58:00Z",
            status: "partial",
        });
    });

    it("returns closed only when every configured expected location is freshly confirmed closed", () => {
        const result = computeOccupancySummary([
            location({locationId: 1, isClosed: true, fetchedAt: "2026-08-31T11:59:00Z"}),
            location({locationId: 2, isClosed: true, fetchedAt: "2026-08-31T11:58:00Z"}),
        ], {nowMs: NOW});

        expect(result).toEqual({
            count: null,
            observedCapacity: 0,
            expectedOpenCapacity: 0,
            coverage: 0,
            percent: null,
            observedLocations: 0,
            expectedLocations: 0,
            latestFetchedAt: "2026-08-31T11:59:00Z",
            oldestFetchedAt: "2026-08-31T11:58:00Z",
            status: "closed",
        });
    });

    it("keeps a real observed count while hiding percentage below 0.5 coverage", () => {
        const result = computeOccupancySummary([
            location({locationId: 1, currentCapacity: 4, maxCapacity: 40}),
            location({locationId: 2, currentCapacity: 60, maxCapacity: 60, fetchedAt: null}),
        ], {nowMs: NOW});

        expect(result).toMatchObject({
            count: 4,
            observedCapacity: 40,
            expectedOpenCapacity: 100,
            coverage: 0.4,
            percent: null,
            observedLocations: 1,
            expectedLocations: 2,
            status: "insufficient",
        });
    });

    it.each([
        ["partial", 100, 200, 0.5, "partial"],
        ["live", 400, 500, 0.8, "live"],
    ] as const)("uses the exact %s coverage boundary", (_label, observedCapacity, expectedCapacity, coverage, status) => {
        const result = computeOccupancySummary([
            location({locationId: 1, currentCapacity: observedCapacity / 10, maxCapacity: observedCapacity}),
            location({locationId: 2, currentCapacity: 0, maxCapacity: expectedCapacity - observedCapacity, fetchedAt: null}),
        ], {nowMs: NOW});

        expect(result).toMatchObject({
            count: observedCapacity / 10,
            observedCapacity,
            expectedOpenCapacity: expectedCapacity,
            coverage,
            percent: 10,
            status,
        });
    });
});
