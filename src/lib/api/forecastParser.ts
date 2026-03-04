import axios from "axios";
import type {FacilityId} from "../types/facility";
import type {FacilityForecastResponse, ForecastDay, ForecastOccupancyThresholds} from "../types/forecast";
import {env} from "../config/env";
import type {OccupancyThresholds} from "../../shared/utils/styles";

const FORECAST_API_BASE_URL = env.forecastApiBaseUrl;
const FORECAST_REQUEST_TIMEOUT_MS = 10_000;

const CHICAGO_TIMEZONE = "America/Chicago";

export interface FacilityForecastPayload {
    days: ForecastDay[];
    forecastDayStartHour: number | null;
    forecastDayEndHour: number | null;
    occupancyThresholds: OccupancyThresholds | null;
    sectionOccupancyThresholds: Partial<Record<string, OccupancyThresholds>>;
    locationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>>;
}

interface ActualHourPayload {
    hourStart: string;
    actualCount: number;
    actualPct?: number | null;
    actualSampleCount?: number;
    actualCoverage?: number | null;
}

interface ActualCategoryPayload {
    key?: string;
    title?: string;
    hours?: ActualHourPayload[];
}

interface FacilityActualHoursResponse {
    facilityId: number;
    date: string;
    categories?: ActualCategoryPayload[];
}

const getChicagoDateISO = (value = new Date()): string => {
    const parts = new Intl.DateTimeFormat("en-US", {
        timeZone: CHICAGO_TIMEZONE,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    }).formatToParts(value);

    const year = parts.find((part) => part.type === "year")?.value;
    const month = parts.find((part) => part.type === "month")?.value;
    const day = parts.find((part) => part.type === "day")?.value;

    if (!year || !month || !day) {
        return value.toISOString().slice(0, 10);
    }

    return `${year}-${month}-${day}`;
};

const normalizeCategoryKey = (value: string | null | undefined): string =>
    String(value ?? "")
        .trim()
        .toLowerCase()
        .replace(/\s+/g, "_");

const normalizeThresholdKey = (value: string | null | undefined): string =>
    String(value ?? "")
        .trim()
        .toLowerCase()
        .replace(/[_\s]+/g, " ");

const parseOccupancyThresholds = (
    value: ForecastOccupancyThresholds | null | undefined
): OccupancyThresholds | null => {
    if (!value || typeof value !== "object") return null;

    const lowMax = Number(value.lowMax);
    const peakMin = Number(value.peakMin);
    if (!Number.isFinite(lowMax) || !Number.isFinite(peakMin)) {
        return null;
    }

    const normalizedLowMax = Math.max(0, Math.min(99, Math.round(lowMax)));
    const normalizedPeakMin = Math.max(normalizedLowMax + 1, Math.min(100, Math.round(peakMin)));
    return {
        lowMax: normalizedLowMax,
        peakMin: normalizedPeakMin,
    };
};

const parseSectionOccupancyThresholds = (
    value: Record<string, ForecastOccupancyThresholds> | null | undefined
): Partial<Record<string, OccupancyThresholds>> => {
    if (!value || typeof value !== "object") {
        return {};
    }

    const parsed: Partial<Record<string, OccupancyThresholds>> = {};
    for (const [key, thresholds] of Object.entries(value)) {
        const normalizedKey = normalizeThresholdKey(key);
        if (!normalizedKey) continue;
        const parsedThresholds = parseOccupancyThresholds(thresholds);
        if (!parsedThresholds) continue;
        parsed[normalizedKey] = parsedThresholds;
    }
    return parsed;
};

const parseLocationOccupancyThresholds = (
    value: Record<string, ForecastOccupancyThresholds> | null | undefined
): Partial<Record<number, OccupancyThresholds>> => {
    if (!value || typeof value !== "object") {
        return {};
    }

    const parsed: Partial<Record<number, OccupancyThresholds>> = {};
    for (const [key, thresholds] of Object.entries(value)) {
        const locationId = Number(key);
        if (!Number.isInteger(locationId) || locationId <= 0) continue;
        const parsedThresholds = parseOccupancyThresholds(thresholds);
        if (!parsedThresholds) continue;
        parsed[locationId] = parsedThresholds;
    }
    return parsed;
};

const mergeActualHoursIntoDays = (
    days: ForecastDay[],
    actualPayload: FacilityActualHoursResponse
): ForecastDay[] => {
    if (!actualPayload?.date || !Array.isArray(actualPayload.categories)) {
        return days;
    }

    const byCategory = new Map<string, Map<string, ActualHourPayload>>();
    for (const category of actualPayload.categories) {
        const key = normalizeCategoryKey(category.key ?? category.title);
        if (!key || !Array.isArray(category.hours)) continue;

        const byHourStart = new Map<string, ActualHourPayload>();
        for (const hour of category.hours) {
            if (!hour?.hourStart) continue;
            byHourStart.set(hour.hourStart, hour);
        }
        if (byHourStart.size > 0) {
            byCategory.set(key, byHourStart);
        }
    }

    if (byCategory.size === 0) {
        return days;
    }

    return days.map((day) => {
        if (day.date !== actualPayload.date || !Array.isArray(day.categories)) {
            return day;
        }

        const categories = day.categories.map((category) => {
            const categoryKey = normalizeCategoryKey(category.key ?? category.title);
            const hoursByStart = byCategory.get(categoryKey);
            if (!hoursByStart || !Array.isArray(category.hours)) {
                return category;
            }

            const hours = category.hours.map((hour) => {
                const actualHour = hoursByStart.get(hour.hourStart);
                if (!actualHour) return hour;
                return {
                    ...hour,
                    actualCount: actualHour.actualCount,
                    actualPct: actualHour.actualPct ?? null,
                    actualSampleCount: actualHour.actualSampleCount,
                    actualCoverage: actualHour.actualCoverage ?? null,
                };
            });
            return {...category, hours};
        });
        return {...day, categories};
    });
};

export async function fetchForecastDays(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<FacilityForecastPayload> {
    const today = getChicagoDateISO();
    const url = `${FORECAST_API_BASE_URL}/api/forecast/facilities/${facilityId}`;

    const resp = await axios.get<FacilityForecastResponse>(url, {
        signal,
        timeout: FORECAST_REQUEST_TIMEOUT_MS,
        params: {compact: 1},
    });
    const weekly = resp.data?.weeklyForecast;
    if (!Array.isArray(weekly)) {
        throw new Error("Forecast weekly payload missing");
    }

    const normalizeHour = (value: unknown): number | null => {
        if (typeof value !== "number" || !Number.isInteger(value)) {
            return null;
        }
        if (value < 0 || value > 23) {
            return null;
        }
        return value;
    };

    let upcomingDays = weekly
        .filter((day) => day?.date >= today)
        .sort((a, b) => a.date.localeCompare(b.date));

    if (upcomingDays.length === 0) {
        throw new Error("No upcoming forecast days in payload");
    }

    try {
        const actualUrl = `${url}/actual-hours`;
        const actualResp = await axios.get<FacilityActualHoursResponse>(actualUrl, {
            signal,
            timeout: FORECAST_REQUEST_TIMEOUT_MS,
            params: {date: today},
        });
        upcomingDays = mergeActualHoursIntoDays(upcomingDays, actualResp.data);
    } catch (actualError) {
        if (signal?.aborted) {
            throw actualError;
        }
        // Keep forecast UX resilient if the optional actual-hours endpoint is unavailable.
        console.info("Actual-hours overlay unavailable; continuing with forecast-only payload.");
    }

    return {
        days: upcomingDays,
        forecastDayStartHour: normalizeHour(resp.data?.forecastDayStartHour),
        forecastDayEndHour: normalizeHour(resp.data?.forecastDayEndHour),
        occupancyThresholds: parseOccupancyThresholds(resp.data?.occupancyThresholds),
        sectionOccupancyThresholds: parseSectionOccupancyThresholds(resp.data?.sectionOccupancyThresholds),
        locationOccupancyThresholds: parseLocationOccupancyThresholds(resp.data?.locationOccupancyThresholds),
    };
}
