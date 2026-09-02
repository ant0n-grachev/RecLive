import type {FacilityId, FacilityPayload, LiveDataSource, Location} from "../types/facility";
import {nick} from "../data/nick";
import {bakke} from "../data/bakke";
import {FACILITY_DISPLAY_NAMES} from "../config/facilitySections";
import {requestJson, uniqueApiUrls} from "./client";
import {liveCountsResponseSchema} from "./schemas";

const LIVE_COUNTS_PATHS = uniqueApiUrls(["/api/live-counts"]);

const FACILITY_LAYOUTS: Record<FacilityId, Record<number, Location[]>> = {
    1186: nick,
    1656: bakke,
};

const cloneLayout = (layout: Record<number, Location[]>): Record<number, Location[]> =>
    Object.fromEntries(
        Object.entries(layout).map(([floor, locations]) => [
            Number(floor),
            locations.map((location) => ({...location})),
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

const fetchLiveRows = async (
    signal?: AbortSignal
): Promise<{rows: ReturnType<typeof liveCountsResponseSchema.parse>["rows"]; source: LiveDataSource}> => {
    const path = LIVE_COUNTS_PATHS[0];
    if (!path) {
        throw new Error("Live counts endpoint is unavailable");
    }
    const payload = await requestJson(path, liveCountsResponseSchema, {
        signal,
    });
    return {
        rows: payload.rows,
        source: "facility_api",
    };
};

export async function fetchFacility(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<FacilityPayload> {
    const layout = cloneLayout(FACILITY_LAYOUTS[facilityId]);
    const {rows: live, source} = await fetchLiveRows(signal);

    const index: Record<number, Location> = {};

    for (const floor of Object.values(layout)) {
        for (const loc of floor) {
            index[loc.locationId] = loc;

            loc.isClosed = null;
            loc.currentCapacity = null;
            loc.lastUpdated = null;
            loc.fetchedAt = null;
        }
    }

    for (const row of live) {
        const loc = index[row.LocationId];
        if (!loc) continue;

        loc.isClosed = row.IsClosed;
        loc.currentCapacity = row.LastCount;
        loc.lastUpdated = row.LastUpdatedDateAndTime ?? null;
        loc.fetchedAt = row.FetchedAt;
    }

    return {
        facilityId,
        facilityName: FACILITY_DISPLAY_NAMES[facilityId],
        floors: layout,
        locations: flatten(layout),
        liveDataSource: source,
    };
}
