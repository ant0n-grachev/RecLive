import {useCallback, useEffect, useRef, useState} from "react";
import {fetchFacility} from "../../lib/api/facilityParser";
import {getFacilityCache, setFacilityCache} from "../../lib/storage/facilityCache";
import type {FacilityId, FacilityPayload, LiveDataSource} from "../../lib/types/facility";
import type {LiveOutageState} from "../warningStatus";

interface LatestLiveSnapshot {
    payload: FacilityPayload | null;
    source: LiveDataSource | null;
    outage: LiveOutageState;
}

interface UseLiveFacilityDataArgs {
    facility: FacilityId;
    refreshKey: number;
    isOffline: boolean;
}

export interface LiveFacilityDataState {
    data: FacilityPayload | null;
    isLoading: boolean;
    error: string | null;
    liveDataSource: LiveDataSource | null;
    liveOutageState: LiveOutageState;
    hasPendingLiveRetry: boolean;
    cacheTimestampMs: number | null;
    prepareRefresh: () => void;
}

export const useLiveFacilityData = ({
    facility,
    refreshKey,
    isOffline,
}: UseLiveFacilityDataArgs): LiveFacilityDataState => {
    const [data, setData] = useState<FacilityPayload | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [liveDataSource, setLiveDataSource] = useState<LiveDataSource | null>(null);
    const [liveOutageState, setLiveOutageState] = useState<LiveOutageState>("none");
    const [hasPendingLiveRetry, setHasPendingLiveRetry] = useState(false);
    const [cacheTimestampMs, setCacheTimestampMs] = useState<number | null>(null);
    const latestLiveSnapshotRef = useRef<LatestLiveSnapshot>({
        payload: null,
        source: null,
        outage: "none",
    });

    useEffect(() => {
        latestLiveSnapshotRef.current = {
            payload: data,
            source: liveDataSource,
            outage: liveOutageState,
        };
    }, [data, liveDataSource, liveOutageState]);

    useEffect(() => {
        const controller = new AbortController();
        let isCancelled = false;

        const cached = getFacilityCache(facility);
        const latestSnapshot = latestLiveSnapshotRef.current;
        const hasCurrentFacilityData = latestSnapshot.payload?.facilityId === facility;
        const hasVisibleCachedSnapshot = hasCurrentFacilityData
            ? latestSnapshot.source === "cache"
            : cached !== null;

        if (!hasCurrentFacilityData && cached) {
            setData(cached.payload);
            setLiveDataSource("cache");
            setLiveOutageState("cache");
            setCacheTimestampMs(cached.cachedAt);
        } else if (!hasCurrentFacilityData) {
            setData(null);
            setLiveDataSource(null);
            setLiveOutageState("none");
            setCacheTimestampMs(null);
        }

        const load = async () => {
            setIsLoading(true);
            setError(null);
            setLiveOutageState(hasVisibleCachedSnapshot ? "cache" : "none");

            if (isOffline) {
                setHasPendingLiveRetry(false);
                if (hasCurrentFacilityData) {
                    setLiveOutageState("cache");
                    setError(null);
                } else if (cached) {
                    setData(cached.payload);
                    setLiveDataSource("cache");
                    setLiveOutageState("cache");
                    setCacheTimestampMs(cached.cachedAt);
                    setError(null);
                } else {
                    setData(null);
                    setLiveDataSource(null);
                    setLiveOutageState("no_cache");
                    setCacheTimestampMs(null);
                    setError("No internet connection right now, and there is no recent saved snapshot. Reconnect to load live occupancy and predictions.");
                }
                setIsLoading(false);
                return;
            }

            try {
                const payload = await fetchFacility(facility, controller.signal);
                if (isCancelled || controller.signal.aborted) return;

                setData(payload);
                setError(null);
                setLiveDataSource(payload.liveDataSource ?? "facility_api");
                setLiveOutageState("none");
                setHasPendingLiveRetry(false);
                setCacheTimestampMs(null);
                setFacilityCache(facility, payload);
            } catch {
                if (isCancelled || controller.signal.aborted) return;

                const currentSnapshot = latestLiveSnapshotRef.current;
                const canKeepVisibleLiveData = currentSnapshot.payload?.facilityId === facility
                    && currentSnapshot.outage === "none"
                    && currentSnapshot.source !== null
                    && currentSnapshot.source !== "cache";
                const canKeepVisibleCachedData = currentSnapshot.payload?.facilityId === facility
                    && currentSnapshot.outage === "cache"
                    && currentSnapshot.source === "cache";

                if (canKeepVisibleLiveData) {
                    setHasPendingLiveRetry(true);
                    setError(null);
                    return;
                }

                if (canKeepVisibleCachedData) {
                    setLiveOutageState("cache");
                    setHasPendingLiveRetry(false);
                    setError(null);
                    return;
                }

                const fallback = getFacilityCache(facility);
                if (fallback) {
                    setData(fallback.payload);
                    setLiveDataSource("cache");
                    setLiveOutageState("cache");
                    setCacheTimestampMs(fallback.cachedAt);
                    setHasPendingLiveRetry(false);
                    setError(null);
                } else {
                    setData(null);
                    setLiveDataSource(null);
                    setLiveOutageState("no_cache");
                    setCacheTimestampMs(null);
                    setHasPendingLiveRetry(false);
                    setError("Live and prediction services are temporarily unavailable, and no recent saved snapshot is available right now. Please try again shortly.");
                }
            } finally {
                if (!isCancelled && !controller.signal.aborted) {
                    setIsLoading(false);
                }
            }
        };

        void load();

        return () => {
            isCancelled = true;
            controller.abort();
        };
    }, [facility, refreshKey, isOffline]);

    const prepareRefresh = useCallback(() => {
        setIsLoading(true);
        setError(null);
    }, []);

    return {
        data,
        isLoading,
        error,
        liveDataSource,
        liveOutageState,
        hasPendingLiveRetry,
        cacheTimestampMs,
        prepareRefresh,
    };
};
