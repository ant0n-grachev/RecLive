import axios from "axios";
import type {FacilityId, FacilityPayload, LiveDataSource, Location} from "../types/facility";
import {nick} from "../data/nick";
import {bakke} from "../data/bakke";
import {env} from "../config/env";
import {FACILITY_DISPLAY_NAMES} from "../config/facilitySections";

const FORECAST_API_BASE_URL = env.forecastApiBaseUrl;
const LIVE_COUNTS_FACILITY_URL = env.liveCountsUrl;
const LIVE_COUNTS_FALLBACK_URL = `${FORECAST_API_BASE_URL}/api/live-counts`;
const LIVE_COUNTS_REQUEST_TIMEOUT_MS = 10_000;

interface LiveLocationRow {
    LocationId: number;
    IsClosed: boolean | null;
    LastCount: number | null;
    LastUpdatedDateAndTime: string | null;
}

interface LiveCountsPayloadEnvelope {
    rows?: LiveLocationRow[];
    data?: LiveLocationRow[];
}

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

const normalizeLiveRows = (payload: unknown): LiveLocationRow[] => {
    if (Array.isArray(payload)) return payload as LiveLocationRow[];
    if (!payload || typeof payload !== "object") {
        throw new Error("Unexpected live counts payload");
    }

    const envelope = payload as LiveCountsPayloadEnvelope;
    if (Array.isArray(envelope.rows)) return envelope.rows;
    if (Array.isArray(envelope.data)) return envelope.data;
    throw new Error("Unexpected live counts payload");
};

const ensureNonEmptyLiveRows = (rows: LiveLocationRow[]): LiveLocationRow[] => {
    if (rows.length === 0) {
        throw new Error("Live counts feed is empty");
    }
    return rows;
};

const fetchLiveRows = async (
    signal?: AbortSignal
): Promise<{rows: LiveLocationRow[]; source: LiveDataSource}> => {
    try {
        const directResp = await axios.get<unknown>(LIVE_COUNTS_FACILITY_URL, {
            signal,
            timeout: LIVE_COUNTS_REQUEST_TIMEOUT_MS,
        });
        return {
            rows: ensureNonEmptyLiveRows(normalizeLiveRows(directResp.data)),
            source: "facility_api",
        };
    } catch (directError) {
        if (signal?.aborted) throw directError;
    }

    const fallbackResp = await axios.get<unknown>(LIVE_COUNTS_FALLBACK_URL, {
        signal,
        timeout: LIVE_COUNTS_REQUEST_TIMEOUT_MS,
    });
    return {
        rows: ensureNonEmptyLiveRows(normalizeLiveRows(fallbackResp.data)),
        source: "fallback_api",
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
        facilityName: FACILITY_DISPLAY_NAMES[facilityId],
        floors: layout,
        locations: flatten(layout),
        liveDataSource: source,
    };
}
