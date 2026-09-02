import type {FacilityId} from "../types/facility";
import {ApiError, requestJson} from "./client";
import {facilityScheduleSchema} from "./schemas";

export async function fetchFacilityHours(
    facilityId: FacilityId,
    signal?: AbortSignal
) {
    const schedule = await requestJson(
        `/api/facility-hours/facilities/${facilityId}`,
        facilityScheduleSchema,
        {signal}
    );
    if (schedule.facilityId !== facilityId) {
        throw new ApiError("schema", "API response did not match its contract");
    }
    return schedule;
}
