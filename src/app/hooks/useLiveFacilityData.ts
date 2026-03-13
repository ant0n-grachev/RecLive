import {useCallback, useEffect, useRef, useState} from "react";
import {fetchFacility} from "../../lib/api/facilityParser";
import {getFacilityCache, setFacilityCache} from "../../lib/storage/facilityCache";
import type {FacilityId, FacilityPayload, LiveDataSource} from "../../lib/types/facility";
import {retryAsync} from "../../shared/utils/retry";
import type {LiveOutageState} from "../warningStatus";

const FETCH_RETRY_ATTEMPTS = 3;
const FETCH_RETRY_DELAY_MS = 1200;

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
        const hasStableLiveSnapshot = hasCurrentFacilityData
            && latestSnapshot.outage === "none"
            && latestSnapshot.source !== null
            && latestSnapshot.source !== "cache";

        if (!hasCurrentFacilityData && cached) {
            setData(cached.payload);
            setLiveDataSource("cache");
            setLiveOutageState("none");
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
            setLiveOutageState("none");

            if (isOffline) {
                setHasPendingLiveRetry(false);
                if (cached) {
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
                const payload = await retryAsync(
                    () => fetchFacility(facility, controller.signal),
                    {
                        attempts: FETCH_RETRY_ATTEMPTS,
                        initialDelayMs: FETCH_RETRY_DELAY_MS,
                        backoffMultiplier: 1.5,
                        signal: controller.signal,
                        shouldRetryResult: (result) =>
                            (hasStableLiveSnapshot || Boolean(cached))
                            && result.liveDataSource === "fallback_api",
                    }
                );
                if (isCancelled || controller.signal.aborted) return;

                setData(payload);
                setError(null);
                setLiveDataSource(payload.liveDataSource ?? "facility_api");
                setLiveOutageState("none");
                setHasPendingLiveRetry(false);
                setCacheTimestampMs(null);
                setFacilityCache(facility, payload);
            } catch (loadError) {
                if (isCancelled || controller.signal.aborted) return;

                console.error("Failed to fetch facility data", loadError);
                const currentSnapshot = latestLiveSnapshotRef.current;
                const canKeepVisibleLiveData = currentSnapshot.payload?.facilityId === facility
                    && currentSnapshot.outage === "none"
                    && currentSnapshot.source !== null
                    && currentSnapshot.source !== "cache";

                if (canKeepVisibleLiveData) {
                    setHasPendingLiveRetry(true);
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
