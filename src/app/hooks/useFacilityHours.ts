import {useEffect, useMemo, useRef, useState} from "react";
import {fetchFacilityHours} from "../../lib/api/facilityScheduleParser";
import type {FacilityId} from "../../lib/types/facility";
import type {FacilityHoursFacilityPayload} from "../../lib/types/facilitySchedule";
import {retryAsync} from "../../shared/utils/retry";

const FETCH_RETRY_ATTEMPTS = 3;
const FETCH_RETRY_DELAY_MS = 1200;

interface UseFacilityHoursArgs {
    facility: FacilityId;
    refreshKey: number;
}

export interface FacilityHoursState {
    activeSchedule: FacilityHoursFacilityPayload | null;
    isFacilityHoursLoading: boolean;
    facilityHoursError: string | null;
    hasPendingScheduleRetry: boolean;
}

export const useFacilityHours = ({
    facility,
    refreshKey,
}: UseFacilityHoursArgs): FacilityHoursState => {
    const [facilityHoursByFacility, setFacilityHoursByFacility] = useState<Partial<Record<FacilityId, FacilityHoursFacilityPayload>>>({});
    const [isFacilityHoursLoading, setIsFacilityHoursLoading] = useState(false);
    const [facilityHoursError, setFacilityHoursError] = useState<string | null>(null);
    const [hasPendingScheduleRetry, setHasPendingScheduleRetry] = useState(false);
    const latestFacilityHoursByFacilityRef = useRef<Partial<Record<FacilityId, FacilityHoursFacilityPayload>>>({});

    useEffect(() => {
        latestFacilityHoursByFacilityRef.current = facilityHoursByFacility;
    }, [facilityHoursByFacility]);

    useEffect(() => {
        const controller = new AbortController();
        let isCancelled = false;
        const hasScheduleForCurrentFacility = Boolean(
            latestFacilityHoursByFacilityRef.current[facility]
        );

        const loadSchedule = async () => {
            setIsFacilityHoursLoading(true);
            setFacilityHoursError(null);

            try {
                const schedulePayload = await retryAsync(
                    () => fetchFacilityHours(facility, controller.signal),
                    {
                        attempts: FETCH_RETRY_ATTEMPTS,
                        initialDelayMs: FETCH_RETRY_DELAY_MS,
                        backoffMultiplier: 1.5,
                        signal: controller.signal,
                    }
                );
                if (isCancelled || controller.signal.aborted) return;
                setHasPendingScheduleRetry(false);
                setFacilityHoursByFacility((prev) => ({
                    ...prev,
                    [facility]: schedulePayload,
                }));
            } catch (loadError) {
                if (isCancelled || controller.signal.aborted) return;
                console.error("Failed to fetch facility hours", loadError);
                if (hasScheduleForCurrentFacility) {
                    setHasPendingScheduleRetry(true);
                    setFacilityHoursError(null);
                    return;
                }
                setFacilityHoursError("Schedule unavailable right now.");
            } finally {
                if (!isCancelled && !controller.signal.aborted) {
                    setIsFacilityHoursLoading(false);
                }
            }
        };

        void loadSchedule();

        return () => {
            isCancelled = true;
            controller.abort();
        };
    }, [facility, refreshKey]);

    const activeSchedule = useMemo(
        () => facilityHoursByFacility[facility] ?? null,
        [facility, facilityHoursByFacility]
    );

    return {
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        hasPendingScheduleRetry,
    };
};
