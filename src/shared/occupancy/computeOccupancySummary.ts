import type {Location} from "../../lib/types/facility";

export interface OccupancySummary {
    count: number | null;
    observedCapacity: number;
    expectedOpenCapacity: number;
    coverage: number;
    percent: number | null;
    observedLocations: number;
    expectedLocations: number;
    latestFetchedAt: string | null;
    oldestFetchedAt: string | null;
    status: "live" | "partial" | "insufficient" | "closed" | "unknown";
}

export interface ComputeOccupancySummaryOptions {
    nowMs?: number;
    freshnessMs?: number;
}

export const OCCUPANCY_FRESHNESS_MS = 10 * 60 * 1000;

const isPositiveFinite = (value: number | null): value is number => (
    typeof value === "number" && Number.isFinite(value) && value > 0
);

const isNonnegativeFinite = (value: number | null): value is number => (
    typeof value === "number" && Number.isFinite(value) && value >= 0
);

export function computeOccupancySummary(
    locations: readonly Location[],
    options: ComputeOccupancySummaryOptions = {}
): OccupancySummary {
    const nowMs = options.nowMs ?? Date.now();
    const freshnessMs = options.freshnessMs ?? OCCUPANCY_FRESHNESS_MS;
    const canEstablishFreshness = Number.isFinite(nowMs)
        && Number.isFinite(freshnessMs)
        && freshnessMs >= 0;

    let configuredLocations = 0;
    let closedLocations = 0;
    let expectedLocations = 0;
    let expectedOpenCapacity = 0;
    let observedLocations = 0;
    let observedCapacity = 0;
    let observedCount = 0;
    let latestFetchedAt: string | null = null;
    let latestFetchedMs = Number.NEGATIVE_INFINITY;
    let oldestFetchedAt: string | null = null;
    let oldestFetchedMs = Number.POSITIVE_INFINITY;

    const recordTrustworthyTimestamp = (fetchedAt: string, fetchedMs: number) => {
        if (fetchedMs > latestFetchedMs) {
            latestFetchedMs = fetchedMs;
            latestFetchedAt = fetchedAt;
        }
        if (fetchedMs < oldestFetchedMs) {
            oldestFetchedMs = fetchedMs;
            oldestFetchedAt = fetchedAt;
        }
    };

    for (const location of locations) {
        if (!isPositiveFinite(location.maxCapacity)) continue;

        configuredLocations += 1;

        const fetchedAt = location.fetchedAt;
        const fetchedMs = fetchedAt === null ? Number.NaN : Date.parse(fetchedAt);
        const isFresh = canEstablishFreshness
            && Number.isFinite(fetchedMs)
            && fetchedMs <= nowMs
            && nowMs - fetchedMs <= freshnessMs;

        if (isFresh && location.isClosed === true && fetchedAt !== null) {
            closedLocations += 1;
            recordTrustworthyTimestamp(fetchedAt, fetchedMs);
            continue;
        }

        expectedLocations += 1;
        expectedOpenCapacity += location.maxCapacity;

        if (
            isFresh
            && location.isClosed === false
            && isNonnegativeFinite(location.currentCapacity)
            && fetchedAt !== null
        ) {
            observedLocations += 1;
            observedCapacity += location.maxCapacity;
            observedCount += location.currentCapacity;
            recordTrustworthyTimestamp(fetchedAt, fetchedMs);
        }
    }

    const coverage = expectedOpenCapacity > 0
        ? observedCapacity / expectedOpenCapacity
        : 0;
    const allConfiguredLocationsClosed = configuredLocations > 0
        && closedLocations === configuredLocations;

    let status: OccupancySummary["status"];
    if (allConfiguredLocationsClosed) {
        status = "closed";
    } else if (expectedOpenCapacity <= 0) {
        status = "unknown";
    } else if (coverage >= 0.8) {
        status = "live";
    } else if (coverage >= 0.5) {
        status = "partial";
    } else {
        status = "insufficient";
    }

    const hasObservedCount = observedLocations > 0;
    const showsPercent = status === "live" || status === "partial";

    return {
        count: hasObservedCount ? observedCount : null,
        observedCapacity,
        expectedOpenCapacity,
        coverage,
        percent: showsPercent ? (observedCount / observedCapacity) * 100 : null,
        observedLocations,
        expectedLocations,
        latestFetchedAt,
        oldestFetchedAt,
        status,
    };
}
