import {useEffect, useRef, useState} from "react";
import {fetchForecastDays} from "../../lib/api/forecastParser";
import type {FacilityId} from "../../lib/types/facility";
import type {ForecastDay} from "../../lib/types/forecast";
import type {OccupancyThresholds} from "../../shared/utils/styles";
import {retryAsync} from "../../shared/utils/retry";

const FETCH_RETRY_ATTEMPTS = 3;
const FETCH_RETRY_DELAY_MS = 1200;

export interface ForecastHourBounds {
    startHour: number | null;
    endHour: number | null;
}

interface LatestForecastSnapshot {
    facility: FacilityId | null;
    hasData: boolean;
}

interface UseForecastDataArgs {
    facility: FacilityId;
    refreshKey: number;
}

export interface ForecastDataState {
    forecastDays: ForecastDay[];
    forecastOccupancyThresholds: OccupancyThresholds | null;
    forecastSectionOccupancyThresholds: Partial<Record<string, OccupancyThresholds>>;
    forecastLocationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>>;
    forecastHourBounds: ForecastHourBounds;
    forecastError: string | null;
    isForecastLoading: boolean;
    hasPendingForecastRetry: boolean;
}

export const useForecastData = ({
    facility,
    refreshKey,
}: UseForecastDataArgs): ForecastDataState => {
    const [forecastDays, setForecastDays] = useState<ForecastDay[]>([]);
    const [forecastDataFacility, setForecastDataFacility] = useState<FacilityId | null>(null);
    const [forecastOccupancyThresholds, setForecastOccupancyThresholds] = useState<OccupancyThresholds | null>(null);
    const [forecastSectionOccupancyThresholds, setForecastSectionOccupancyThresholds] = useState<Partial<Record<string, OccupancyThresholds>>>({});
    const [forecastLocationOccupancyThresholds, setForecastLocationOccupancyThresholds] = useState<Partial<Record<number, OccupancyThresholds>>>({});
    const [forecastHourBounds, setForecastHourBounds] = useState<ForecastHourBounds>({
        startHour: null,
        endHour: null,
    });
    const [forecastError, setForecastError] = useState<string | null>(null);
    const [isForecastLoading, setIsForecastLoading] = useState(false);
    const [hasPendingForecastRetry, setHasPendingForecastRetry] = useState(false);
    const latestForecastSnapshotRef = useRef<LatestForecastSnapshot>({
        facility: null,
        hasData: false,
    });

    useEffect(() => {
        latestForecastSnapshotRef.current = {
            facility: forecastDataFacility,
            hasData: forecastDays.length > 0,
        };
    }, [forecastDataFacility, forecastDays]);

    useEffect(() => {
        const controller = new AbortController();
        let isCancelled = false;
        const latestForecastSnapshot = latestForecastSnapshotRef.current;
        const hasForecastForCurrentFacility = (
            latestForecastSnapshot.facility === facility
            && latestForecastSnapshot.hasData
        );

        const loadForecast = async () => {
            setIsForecastLoading(true);
            setForecastError(null);
            if (!hasForecastForCurrentFacility) {
                setForecastDataFacility(null);
                setForecastDays([]);
                setForecastOccupancyThresholds(null);
                setForecastSectionOccupancyThresholds({});
                setForecastLocationOccupancyThresholds({});
                setForecastHourBounds({startHour: null, endHour: null});
            }

            try {
                const forecastPayload = await retryAsync(
                    () => fetchForecastDays(facility, controller.signal),
                    {
                        attempts: FETCH_RETRY_ATTEMPTS,
                        initialDelayMs: FETCH_RETRY_DELAY_MS,
                        backoffMultiplier: 1.5,
                        signal: controller.signal,
                    }
                );
                if (isCancelled || controller.signal.aborted) return;
                setForecastDataFacility(facility);
                setHasPendingForecastRetry(false);
                setForecastDays(forecastPayload.days);
                setForecastOccupancyThresholds(forecastPayload.occupancyThresholds);
                setForecastSectionOccupancyThresholds(forecastPayload.sectionOccupancyThresholds);
                setForecastLocationOccupancyThresholds(forecastPayload.locationOccupancyThresholds);
                setForecastHourBounds({
                    startHour: forecastPayload.forecastDayStartHour,
                    endHour: forecastPayload.forecastDayEndHour,
                });
            } catch (loadError) {
                if (isCancelled || controller.signal.aborted) return;
                console.error("Failed to fetch forecast data", loadError);
                if (hasForecastForCurrentFacility) {
                    setHasPendingForecastRetry(true);
                    setForecastError(null);
                    return;
                }
                setForecastDataFacility(null);
                setForecastDays([]);
                setForecastOccupancyThresholds(null);
                setForecastSectionOccupancyThresholds({});
                setForecastLocationOccupancyThresholds({});
                setForecastError("Forecast unavailable right now.");
                setForecastHourBounds({startHour: null, endHour: null});
            } finally {
                if (!isCancelled && !controller.signal.aborted) {
                    setIsForecastLoading(false);
                }
            }
        };

        void loadForecast();

        return () => {
            isCancelled = true;
            controller.abort();
        };
    }, [facility, refreshKey]);

    return {
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        isForecastLoading,
        hasPendingForecastRetry,
    };
};
