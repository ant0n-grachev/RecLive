import type {FacilityId, FacilityPayload, LiveDataSource, Location} from "../types/facility";
import {nick} from "../data/nick";
import {bakke} from "../data/bakke";
import {FACILITY_DISPLAY_NAMES} from "../config/facilitySections";
import {computeOccupancySummary} from "../../shared/occupancy/computeOccupancySummary";
import {ApiError, requestJson, uniqueApiUrls} from "./client";
import {liveCountsResponseSchema} from "./schemas";
import {fetchPublicLiveCounts, type PublicLiveCountRow} from "./publicLiveCounts";

const LIVE_COUNTS_PATHS = uniqueApiUrls(["/api/live-counts"]);
const BACKUP_TIMEOUT_MS = 5_000;

const FACILITY_LAYOUTS: Record<FacilityId, Record<number, Location[]>> = {
    1186: nick,
    1656: bakke,
};

type BackupLiveCountRow = ReturnType<typeof liveCountsResponseSchema.parse>["rows"][number];
type FacilityLiveCountRow = BackupLiveCountRow | PublicLiveCountRow;

const cloneLayout = (layout: Record<number, Location[]>): Record<number, Location[]> =>
    Object.fromEntries(
        Object.entries(layout).map(([floor, locations]) => [
            Number(floor),
            locations.map((location) => ({
                ...location,
                isClosed: null,
                currentCapacity: null,
                lastUpdated: null,
                fetchedAt: null,
            })),
        ])
    );

const flatten = (floors: Record<number, Location[]>) => {
    return Object.values(floors)
        .flat()
        .sort(
            (a, b) =>
                a.floor - b.floor || a.locationName.localeCompare(b.locationName)
        );
};

const rowMatchesFacility = (
    row: FacilityLiveCountRow,
    facilityId: FacilityId,
): boolean => (
    !("FacilityId" in row) || row.FacilityId === facilityId
);

const buildFacilityPayload = (
    facilityId: FacilityId,
    rows: readonly FacilityLiveCountRow[],
    liveDataSource: LiveDataSource,
): FacilityPayload => {
    const layout = cloneLayout(FACILITY_LAYOUTS[facilityId]);
    const index: Record<number, Location> = {};

    for (const floor of Object.values(layout)) {
        for (const location of floor) {
            index[location.locationId] = location;
        }
    }

    for (const row of rows) {
        if (!rowMatchesFacility(row, facilityId)) continue;
        const location = index[row.LocationId];
        if (!location) continue;

        location.isClosed = row.IsClosed;
        location.currentCapacity = row.LastCount;
        location.lastUpdated = row.LastUpdatedDateAndTime ?? null;
        location.fetchedAt = row.FetchedAt;
    }

    return {
        facilityId,
        facilityName: FACILITY_DISPLAY_NAMES[facilityId],
        floors: layout,
        locations: flatten(layout),
        liveDataSource,
    };
};

const hasUsableSelectedFacilityData = (payload: FacilityPayload): boolean => {
    const summary = computeOccupancySummary(payload.locations, {nowMs: Date.now()});
    return summary.status === "live" || summary.status === "closed";
};

const fetchBackupRows = async (signal?: AbortSignal): Promise<BackupLiveCountRow[]> => {
    const path = LIVE_COUNTS_PATHS[0];
    if (!path) {
        throw new ApiError("client", "Live counts endpoint is unavailable");
    }
    const payload = await requestJson(path, liveCountsResponseSchema, {
        signal,
        timeoutMs: BACKUP_TIMEOUT_MS,
        attempts: 1,
    });
    return payload.rows;
};

const isCallerAbort = (cause: unknown, signal?: AbortSignal): boolean => (
    signal?.aborted === true
    || (cause instanceof ApiError && cause.kind === "aborted")
);

export async function fetchFacility(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<FacilityPayload> {
    try {
        const officialRows = await fetchPublicLiveCounts({signal});
        const officialPayload = buildFacilityPayload(facilityId, officialRows, "official_api");
        if (hasUsableSelectedFacilityData(officialPayload)) return officialPayload;
    } catch (cause) {
        if (isCallerAbort(cause, signal)) throw cause;
    }

    const backupRows = await fetchBackupRows(signal);
    const backupPayload = buildFacilityPayload(facilityId, backupRows, "facility_api");
    if (!hasUsableSelectedFacilityData(backupPayload)) {
        throw new ApiError("schema", "Live counts did not include enough current facility data");
    }
    return backupPayload;
}
