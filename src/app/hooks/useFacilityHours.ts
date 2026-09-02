import {useEffect, useMemo, useRef, useState} from "react";
import {fetchFacilityHours} from "../../lib/api/facilityScheduleParser";
import type {FacilityScheduleResponse} from "../../lib/api/schemas";
import type {FacilityId} from "../../lib/types/facility";

interface UseFacilityHoursArgs {
    facility: FacilityId;
    refreshKey: number;
}

export interface FacilityHoursState {
    activeSchedule: FacilityScheduleResponse | null;
    isFacilityHoursLoading: boolean;
    facilityHoursError: string | null;
    hasPendingScheduleRetry: boolean;
}

export const useFacilityHours = ({
    facility,
    refreshKey,
}: UseFacilityHoursArgs): FacilityHoursState => {
    const [facilityHoursByFacility, setFacilityHoursByFacility] = useState<Partial<Record<FacilityId, FacilityScheduleResponse>>>({});
    const [isFacilityHoursLoading, setIsFacilityHoursLoading] = useState(false);
    const [facilityHoursError, setFacilityHoursError] = useState<string | null>(null);
    const [hasPendingScheduleRetry, setHasPendingScheduleRetry] = useState(false);
    const latestFacilityHoursByFacilityRef = useRef<Partial<Record<FacilityId, FacilityScheduleResponse>>>({});

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
            if (!hasScheduleForCurrentFacility) {
                setHasPendingScheduleRetry(false);
            }

            try {
                const schedulePayload = await fetchFacilityHours(facility, controller.signal);
                if (isCancelled || controller.signal.aborted) return;
                setHasPendingScheduleRetry(false);
                setFacilityHoursByFacility((prev) => ({
                    ...prev,
                    [facility]: schedulePayload,
                }));
            } catch {
                if (isCancelled || controller.signal.aborted) return;
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
