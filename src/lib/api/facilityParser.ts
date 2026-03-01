import axios from "axios";
import type {FacilityId, FacilityPayload, Location} from "../types/facility";
import {nick} from "../data/nick";
import {bakke} from "../data/bakke";

const URL =
    "https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount?AccountAPIKey=7938fc89-a15c-492d-9566-12c961bc1f27";

interface LiveLocationRow {
    LocationId: number;
    IsClosed: boolean | null;
    LastCount: number | null;
    LastUpdatedDateAndTime: string | null;
}

const FACILITY_NAMES: Record<FacilityId, string> = {
    1186: "Nicholas Recreation Center",
    1656: "Bakke Recreation & Wellbeing Center",
};

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

export async function fetchFacility(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<FacilityPayload> {
    const layout = cloneLayout(FACILITY_LAYOUTS[facilityId]);

    const resp = await axios.get<LiveLocationRow[]>(URL, {signal});
    const live = resp.data ?? [];

    const index: Record<number, Location> = {};

    for (const floor of Object.values(layout)) {
        for (const loc of floor) {
            index[loc.locationId] = loc;

            loc.isClosed = null;
            loc.currentCapacity = null;
            loc.lastUpdated = null;
        }
    }

    for (const row of live) {
        const loc = index[row.LocationId];
        if (!loc) continue;

        loc.isClosed = row.IsClosed;
        loc.currentCapacity = row.LastCount;
        loc.lastUpdated = row.LastUpdatedDateAndTime ?? null;
    }

    return {
        facilityId,
        facilityName: FACILITY_NAMES[facilityId],
        floors: layout,
        locations: flatten(layout),
    };
}
