import type {FacilityId} from "../types/facility";
import type {
    ForecastDay,
    ForecastHour,
    ForecastOccupancyThresholds,
} from "../types/forecast";
import type {OccupancyThresholds} from "../../shared/utils/styles";
import {ApiError, requestJson} from "./client";
import {actualHoursResponseSchema, forecastResponseSchema} from "./schemas";

const CHICAGO_TIMEZONE = "America/Chicago";
// Chicago hourly buckets also start on UTC hours, including both fall-back offsets.
const MS_PER_HOUR = 60 * 60 * 1000;

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
    hourEpoch: number;
    observedCount: number | null;
    observedCapacity: number;
    expectedCapacity: number;
    actualCoverage: number;
    temporalCoverage: number;
    coverageThreshold: number;
    actualCount: number | null;
    actualPct?: number | null;
}

interface ActualCategoryPayload {
    key: string;
    hours: ActualHourPayload[];
}

interface FacilityActualHoursResponse {
    facilityId: number;
    date: string;
    categories: ActualCategoryPayload[];
    totalHours: ActualHourPayload[];
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

const isRecord = (value: unknown): value is Record<string, unknown> => (
    typeof value === "object" && value !== null && !Array.isArray(value)
);

const isFiniteNumber = (value: unknown): value is number => (
    typeof value === "number" && Number.isFinite(value)
);

const isNullableFiniteNumber = (value: unknown): value is number | null => (
    value === null || isFiniteNumber(value)
);

const EXPLICIT_TIMEZONE_DESIGNATOR = /(?:Z|[+-]\d{2}:\d{2})$/;

const parseExplicitHourEpoch = (value: unknown): number | null => {
    if (typeof value !== "string" || !EXPLICIT_TIMEZONE_DESIGNATOR.test(value)) {
        return null;
    }
    const epoch = Date.parse(value);
    return Number.isFinite(epoch) ? epoch : null;
};

const parseActualHour = (value: unknown): ActualHourPayload | null => {
    if (!isRecord(value) || typeof value.hourStart !== "string") {
        return null;
    }

    const hourEpoch = parseExplicitHourEpoch(value.hourStart);
    if (
        hourEpoch === null
        || !isNullableFiniteNumber(value.observedCount)
        || !isFiniteNumber(value.observedCapacity)
        || value.observedCapacity < 0
        || !isFiniteNumber(value.expectedCapacity)
        || value.expectedCapacity < 0
        || !isFiniteNumber(value.actualCoverage)
        || !isFiniteNumber(value.temporalCoverage)
        || !isFiniteNumber(value.coverageThreshold)
        || !isNullableFiniteNumber(value.actualCount)
        || (value.actualPct !== undefined && !isNullableFiniteNumber(value.actualPct))
    ) {
        return null;
    }

    return {
        hourStart: value.hourStart,
        hourEpoch,
        observedCount: value.observedCount,
        observedCapacity: value.observedCapacity,
        expectedCapacity: value.expectedCapacity,
        actualCoverage: value.actualCoverage,
        temporalCoverage: value.temporalCoverage,
        coverageThreshold: value.coverageThreshold,
        actualCount: value.actualCount,
        ...(value.actualPct === undefined ? {} : {actualPct: value.actualPct}),
    };
};

const parseActualHours = (value: unknown): ActualHourPayload[] => {
    if (!Array.isArray(value)) {
        return [];
    }
    return value.flatMap((candidate) => {
        const parsed = parseActualHour(candidate);
        return parsed ? [parsed] : [];
    });
};

const parseActualCategory = (value: unknown): ActualCategoryPayload | null => {
    if (!isRecord(value)) {
        return null;
    }
    const rawKey = typeof value.key === "string"
        ? value.key
        : typeof value.title === "string"
            ? value.title
            : null;
    if (rawKey === null) {
        return null;
    }

    const key = normalizeCategoryKey(rawKey);
    const hours = parseActualHours(value.hours);
    if (!key || hours.length === 0) {
        return null;
    }
    return {key, hours};
};

const parseFacilityActualHoursResponse = (value: unknown): FacilityActualHoursResponse | null => {
    if (
        !isRecord(value)
        || !Number.isInteger(value.facilityId)
        || typeof value.facilityId !== "number"
        || value.facilityId <= 0
        || typeof value.date !== "string"
        || !/^\d{4}-\d{2}-\d{2}$/.test(value.date)
    ) {
        return null;
    }

    const categories = Array.isArray(value.categories)
        ? value.categories.flatMap((candidate) => {
            const parsed = parseActualCategory(candidate);
            return parsed ? [parsed] : [];
        })
        : [];

    return {
        facilityId: value.facilityId,
        date: value.date,
        categories,
        totalHours: parseActualHours(value.totalHours),
    };
};

const indexUnambiguousActualHours = (
    hours: ActualHourPayload[]
): Map<number, ActualHourPayload> => {
    const unambiguous = new Map<number, ActualHourPayload>();
    const ambiguousEpochs = new Set<number>();
    for (const hour of hours) {
        if (ambiguousEpochs.has(hour.hourEpoch)) {
            continue;
        }
        if (unambiguous.has(hour.hourEpoch)) {
            unambiguous.delete(hour.hourEpoch);
            ambiguousEpochs.add(hour.hourEpoch);
            continue;
        }
        unambiguous.set(hour.hourEpoch, hour);
    }

    return unambiguous;
};

const mergeActualHour = (
    forecastHour: ForecastHour,
    actualHour: ActualHourPayload
): ForecastHour => ({
    ...forecastHour,
    actualCount: actualHour.actualCount,
    actualPct: actualHour.actualPct ?? null,
    observedCount: actualHour.observedCount,
    observedCapacity: actualHour.observedCapacity,
    expectedCapacity: actualHour.expectedCapacity,
    actualCoverage: actualHour.actualCoverage,
    temporalCoverage: actualHour.temporalCoverage,
    coverageThreshold: actualHour.coverageThreshold,
});

export const mergeActualHoursIntoDays = (
    days: ForecastDay[],
    rawActualPayload: unknown
): ForecastDay[] => {
    const parsedActualPayload = actualHoursResponseSchema.safeParse(rawActualPayload);
    if (!parsedActualPayload.success) {
        return days;
    }
    const actualPayload = parseFacilityActualHoursResponse(parsedActualPayload.data);
    if (!actualPayload) {
        return days;
    }

    const byCategory = new Map<string, Map<number, ActualHourPayload>>();
    const seenCategoryKeys = new Set<string>();
    const ambiguousCategoryKeys = new Set<string>();
    for (const category of actualPayload.categories) {
        if (ambiguousCategoryKeys.has(category.key)) {
            continue;
        }
        if (seenCategoryKeys.has(category.key)) {
            byCategory.delete(category.key);
            ambiguousCategoryKeys.add(category.key);
            continue;
        }
        seenCategoryKeys.add(category.key);

        const byHourStart = indexUnambiguousActualHours(category.hours);
        if (byHourStart.size > 0) {
            byCategory.set(category.key, byHourStart);
        }
    }

    const totalHoursByStart = indexUnambiguousActualHours(actualPayload.totalHours);

    if (byCategory.size === 0 && totalHoursByStart.size === 0) {
        return days;
    }

    return days.map((day) => {
        if (day.date !== actualPayload.date) return day;

        const categories = Array.isArray(day.categories)
            ? day.categories.map((category) => {
                const categoryKey = normalizeCategoryKey(category.key ?? category.title);
                const hoursByStart = byCategory.get(categoryKey);
                if (!hoursByStart || !Array.isArray(category.hours)) {
                    return category;
                }

                const hours = category.hours.map((hour) => {
                    const hourEpoch = parseExplicitHourEpoch(hour.hourStart);
                    if (hourEpoch === null) return hour;
                    const actualHour = hoursByStart.get(Math.floor(hourEpoch / MS_PER_HOUR) * MS_PER_HOUR);
                    if (!actualHour) return hour;
                    return mergeActualHour(hour, actualHour);
                });
                return {...category, hours};
            })
            : day.categories;

        let totalHours = day.totalHours;
        if (Array.isArray(day.totalHours) && totalHoursByStart.size > 0) {
            totalHours = day.totalHours.map((hour) => {
                const hourEpoch = parseExplicitHourEpoch(hour.hourStart);
                if (hourEpoch === null) return hour;
                const actualHour = totalHoursByStart.get(Math.floor(hourEpoch / MS_PER_HOUR) * MS_PER_HOUR);
                if (!actualHour) return hour;
                return mergeActualHour(hour, actualHour);
            });
        }

        return {...day, categories, totalHours};
    });
};

export async function fetchForecastDays(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<FacilityForecastPayload> {
    const today = getChicagoDateISO();
    const path = `/api/forecast/facilities/${facilityId}`;
    const [forecastResult, actualResult] = await Promise.allSettled([
        requestJson(path, forecastResponseSchema, {
            signal,
            attempts: 3,
            params: {compact: 1},
        }),
        requestJson(`${path}/actual-hours`, actualHoursResponseSchema, {
            signal,
            attempts: 3,
            params: {date: today},
        }),
    ]);

    if (signal?.aborted) {
        throw new ApiError("aborted", "Request aborted");
    }
    if (forecastResult.status === "rejected") {
        throw forecastResult.reason;
    }

    const forecast = forecastResult.value;
    if (forecast.facilityId !== facilityId) {
        throw new ApiError("schema", "API response did not match its contract");
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

    let upcomingDays = forecast.weeklyForecast
        .filter((day) => day?.date >= today)
        .sort((a, b) => a.date.localeCompare(b.date));

    if (upcomingDays.length === 0) {
        throw new Error("No upcoming forecast days in payload");
    }

    if (
        actualResult.status === "fulfilled"
        && actualResult.value.facilityId === facilityId
        && actualResult.value.date === today
    ) {
        upcomingDays = mergeActualHoursIntoDays(upcomingDays, actualResult.value);
    }

    if (signal?.aborted) {
        throw new ApiError("aborted", "Request aborted");
    }

    return {
        days: upcomingDays,
        forecastDayStartHour: normalizeHour(forecast.forecastDayStartHour),
        forecastDayEndHour: normalizeHour(forecast.forecastDayEndHour),
        occupancyThresholds: parseOccupancyThresholds(forecast.occupancyThresholds),
        sectionOccupancyThresholds: parseSectionOccupancyThresholds(forecast.sectionOccupancyThresholds),
        locationOccupancyThresholds: parseLocationOccupancyThresholds(forecast.locationOccupancyThresholds),
    };
}
