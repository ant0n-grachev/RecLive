import type {ForecastDay} from "../../lib/types/forecast";
import {formatChicagoTime, getChicagoTimestampMs} from "../../shared/utils/chicagoTime";
import type {FacilityOpenWindow} from "../../shared/utils/facilityScheduleStatus";
import {getOccupancyTone, type OccupancyThresholds} from "../../shared/utils/styles";
import {
    buildBandTimeRanges,
    getLevelAtTimestamp,
    occupancyToneToBandLevel,
    type CrowdBand,
} from "./forecastBands";
import type {ForecastDisplaySlot} from "./forecastHistogram";

export const FORECAST_DISPLAY_SLOT_MINUTES = 30;
const MINUTES_PER_DAY = 24 * 60;
const MS_PER_DAY = 24 * 60 * 60 * 1000;

const chicagoHourMinuteFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
});

const chicagoDateFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
});

export const formatTime = (value?: string): string => {
    const formatted = formatChicagoTime(value);
    return formatted ?? "N/A";
};

export const formatRange = (start?: string, end?: string): string => {
    const formattedStart = formatTime(start);
    const formattedEnd = formatTime(end);
    return `${formattedStart} – ${formattedEnd}`;
};

export const formatWindow = (window: {start?: string; end?: string}): string => {
    const start = formatTime(window.start);
    const end = formatTime(window.end);
    return `${start} – ${end}`;
};

export const parseShortDate = (value?: string | null): string | null => {
    if (!value) return null;
    const match = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return null;

    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isInteger(month) || !Number.isInteger(day)) return null;
    return `${month}/${day}`;
};

const pad2 = (value: number): string => String(value).padStart(2, "0");

const shiftDateKeyByDays = (dateKey: string, dayShift: number): string | null => {
    const match = dateKey.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
        return null;
    }

    const utcDate = new Date(Date.UTC(year, month - 1, day + dayShift));
    if (Number.isNaN(utcDate.getTime())) return null;
    return `${utcDate.getUTCFullYear()}-${pad2(utcDate.getUTCMonth() + 1)}-${pad2(utcDate.getUTCDate())}`;
};

export const getDateKeyDayIndex = (dateKey: string | null | undefined): number | null => {
    if (!dateKey) return null;
    const match = dateKey.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
        return null;
    }

    const utc = Date.UTC(year, month - 1, day);
    const check = new Date(utc);
    if (
        check.getUTCFullYear() !== year
        || check.getUTCMonth() !== month - 1
        || check.getUTCDate() !== day
    ) {
        return null;
    }

    return Math.floor(utc / MS_PER_DAY);
};

export const getChicagoTimestampForDateMinute = (dateKey: string, minuteOffset: number): number | null => {
    const dayShift = Math.floor(minuteOffset / MINUTES_PER_DAY);
    const minuteInDay = ((minuteOffset % MINUTES_PER_DAY) + MINUTES_PER_DAY) % MINUTES_PER_DAY;
    const shiftedDate = shiftDateKeyByDays(dateKey, dayShift);
    if (!shiftedDate) return null;

    const hour = Math.floor(minuteInDay / 60);
    const minute = minuteInDay % 60;
    return getChicagoTimestampMs(`${shiftedDate}T${pad2(hour)}:${pad2(minute)}:00`);
};

export const formatMinuteLabel = (minuteOfDay: number, includeMinutes: boolean): string => {
    const normalized = ((minuteOfDay % MINUTES_PER_DAY) + MINUTES_PER_DAY) % MINUTES_PER_DAY;
    const hour = Math.floor(normalized / 60);
    const minute = normalized % 60;
    const hour12 = hour % 12 === 0 ? 12 : hour % 12;
    const suffix = hour < 12 ? "AM" : "PM";
    if (!includeMinutes && minute === 0) {
        return `${hour12} ${suffix}`;
    }
    return `${hour12}:${pad2(minute)} ${suffix}`;
};

export const formatHistogramRangeLabel = (startMinute: number, endMinute: number): string => {
    return `${formatMinuteLabel(startMinute, true)} – ${formatMinuteLabel(endMinute, true)}`;
};

export const getChicagoMinuteOfDayFromTimestamp = (timestampMs: number): number | null => {
    const parts = chicagoHourMinuteFormatter.formatToParts(new Date(timestampMs));
    const hour = Number(parts.find((part) => part.type === "hour")?.value);
    const minute = Number(parts.find((part) => part.type === "minute")?.value);
    if (!Number.isInteger(hour) || !Number.isInteger(minute)) return null;
    return Math.max(0, Math.min((24 * 60) - 1, hour * 60 + minute));
};

const getChicagoMinuteOfDay = (value: string): number | null => {
    const timestampMs = getChicagoTimestampMs(value);
    if (timestampMs === null) return null;
    return getChicagoMinuteOfDayFromTimestamp(timestampMs);
};

export const getChicagoDateKeyFromTimestamp = (timestampMs: number): string | null => {
    const parts = chicagoDateFormatter.formatToParts(new Date(timestampMs));
    const year = parts.find((part) => part.type === "year")?.value;
    const month = parts.find((part) => part.type === "month")?.value;
    const day = parts.find((part) => part.type === "day")?.value;
    if (!year || !month || !day) return null;
    return `${year}-${month}-${day}`;
};

const getChicagoHourStartTimestamp = (timestampMs: number): number | null => {
    const dateKey = getChicagoDateKeyFromTimestamp(timestampMs);
    const minuteOfDay = getChicagoMinuteOfDayFromTimestamp(timestampMs);
    if (!dateKey || minuteOfDay === null) return null;
    return getChicagoTimestampForDateMinute(dateKey, Math.floor(minuteOfDay / 60) * 60);
};

const isRangeWithinOpenWindows = (
    startMinute: number,
    endMinute: number,
    windows: FacilityOpenWindow[]
): boolean => windows.some((window) => startMinute >= window.startMinutes && endMinute <= window.endMinutes);

export const buildForecastDisplaySlots = (
    day: ForecastDay | null,
    openWindows: FacilityOpenWindow[],
    enforceWorkingHours: boolean,
    thresholds: OccupancyThresholds | null | undefined,
    fallbackBands: CrowdBand[],
    nowTs: number
): ForecastDisplaySlot[] => {
    if (!day) return [];

    const actualCutoffTs = getChicagoHourStartTimestamp(nowTs);
    const fallbackRanges = buildBandTimeRanges(fallbackBands);
    const totalHours = Array.isArray(day.totalHours) ? day.totalHours : [];

    if (totalHours.length > 0) {
        const bySlot = new Map<number, {
            startTs: number;
            endTs: number;
            sumCount: number;
            sumPct: number;
            pctCount: number;
            pointCount: number;
            actualPointCount: number;
        }>();

        for (const hour of totalHours) {
            const timestampMs = getChicagoTimestampMs(hour.hourStart);
            const minuteOfDay = getChicagoMinuteOfDay(hour.hourStart);
            if (timestampMs === null || minuteOfDay === null) continue;

            const slotStartMinute = Math.floor(minuteOfDay / FORECAST_DISPLAY_SLOT_MINUTES) * FORECAST_DISPLAY_SLOT_MINUTES;
            const slotEndMinute = slotStartMinute + FORECAST_DISPLAY_SLOT_MINUTES;
            if (enforceWorkingHours && !isRangeWithinOpenWindows(slotStartMinute, slotEndMinute, openWindows)) {
                continue;
            }

            const useActual = actualCutoffTs !== null
                && timestampMs < actualCutoffTs
                && typeof hour.actualCount === "number";
            const actualCount = typeof hour.actualCount === "number" ? hour.actualCount : null;
            const resolvedCount = useActual
                ? Math.max(0, actualCount ?? 0)
                : Math.max(0, hour.expectedCount ?? 0);
            const resolvedPct = useActual ? hour.actualPct : hour.expectedPct;
            const slotStartTs = getChicagoTimestampForDateMinute(day.date, slotStartMinute);
            const slotEndTs = getChicagoTimestampForDateMinute(day.date, slotEndMinute);
            if (slotStartTs === null || slotEndTs === null || slotEndTs <= slotStartTs) continue;

            const current = bySlot.get(slotStartMinute);
            bySlot.set(slotStartMinute, {
                startTs: current?.startTs ?? slotStartTs,
                endTs: current?.endTs ?? slotEndTs,
                sumCount: (current?.sumCount ?? 0) + resolvedCount,
                sumPct: (current?.sumPct ?? 0) + (
                    typeof resolvedPct === "number" && Number.isFinite(resolvedPct)
                        ? resolvedPct
                        : 0
                ),
                pctCount: (current?.pctCount ?? 0) + (
                    typeof resolvedPct === "number" && Number.isFinite(resolvedPct)
                        ? 1
                        : 0
                ),
                pointCount: (current?.pointCount ?? 0) + 1,
                actualPointCount: (current?.actualPointCount ?? 0) + (useActual ? 1 : 0),
            });
        }

        return [...bySlot.entries()]
            .sort((a, b) => a[0] - b[0])
            .map(([startMinute, bucket]) => {
                const percent = bucket.pctCount > 0 ? bucket.sumPct / bucket.pctCount : null;
                const tone = percent === null ? null : getOccupancyTone(percent * 100, thresholds);
                const level = occupancyToneToBandLevel(tone)
                    ?? getLevelAtTimestamp(bucket.startTs, fallbackRanges)
                    ?? "unknown";
                return {
                    startMinute,
                    endMinute: startMinute + FORECAST_DISPLAY_SLOT_MINUTES,
                    startTs: bucket.startTs,
                    endTs: bucket.endTs,
                    count: Math.max(0, Math.round(bucket.sumCount / Math.max(1, bucket.pointCount))),
                    percent,
                    source: bucket.actualPointCount <= 0
                        ? "predicted"
                        : bucket.actualPointCount >= bucket.pointCount
                            ? "actual"
                            : "mixed",
                    level,
                };
            });
    }

    const categorySources = Array.isArray(day.categories) ? day.categories.map((category) => category.hours ?? []) : [];
    if (categorySources.length === 0) {
        return [];
    }

    const byTimestamp = new Map<number, {
        minuteOfDay: number;
        resolvedCount: number;
        actualCategoryCount: number;
        totalCategoryCount: number;
    }>();

    for (const hours of categorySources) {
        for (const hour of hours) {
            const timestampMs = getChicagoTimestampMs(hour.hourStart);
            const minuteOfDay = getChicagoMinuteOfDay(hour.hourStart);
            if (timestampMs === null || minuteOfDay === null) continue;

            const slotStartMinute = Math.floor(minuteOfDay / FORECAST_DISPLAY_SLOT_MINUTES) * FORECAST_DISPLAY_SLOT_MINUTES;
            const slotEndMinute = slotStartMinute + FORECAST_DISPLAY_SLOT_MINUTES;
            if (enforceWorkingHours && !isRangeWithinOpenWindows(slotStartMinute, slotEndMinute, openWindows)) {
                continue;
            }

            const useActual = actualCutoffTs !== null
                && timestampMs < actualCutoffTs
                && typeof hour.actualCount === "number";
            const actualCount = typeof hour.actualCount === "number" ? hour.actualCount : null;
            const resolvedCount = useActual
                ? Math.max(0, actualCount ?? 0)
                : Math.max(0, hour.expectedCount ?? 0);

            const current = byTimestamp.get(timestampMs);
            byTimestamp.set(timestampMs, {
                minuteOfDay,
                resolvedCount: (current?.resolvedCount ?? 0) + resolvedCount,
                actualCategoryCount: (current?.actualCategoryCount ?? 0) + (useActual ? 1 : 0),
                totalCategoryCount: (current?.totalCategoryCount ?? 0) + 1,
            });
        }
    }

    const bySlot = new Map<number, {
        startTs: number;
        endTs: number;
        sumCount: number;
        pointCount: number;
        actualPointCount: number;
        mixedPointCount: number;
    }>();

    for (const point of byTimestamp.values()) {
        const slotStartMinute = Math.floor(point.minuteOfDay / FORECAST_DISPLAY_SLOT_MINUTES) * FORECAST_DISPLAY_SLOT_MINUTES;
        const slotStartTs = getChicagoTimestampForDateMinute(day.date, slotStartMinute);
        const slotEndTs = getChicagoTimestampForDateMinute(day.date, slotStartMinute + FORECAST_DISPLAY_SLOT_MINUTES);
        if (slotStartTs === null || slotEndTs === null || slotEndTs <= slotStartTs) continue;

        const pointSource: ForecastDisplaySlot["source"] = point.actualCategoryCount <= 0
            ? "predicted"
            : point.actualCategoryCount >= point.totalCategoryCount
                ? "actual"
                : "mixed";
        const current = bySlot.get(slotStartMinute);
        bySlot.set(slotStartMinute, {
            startTs: current?.startTs ?? slotStartTs,
            endTs: current?.endTs ?? slotEndTs,
            sumCount: (current?.sumCount ?? 0) + point.resolvedCount,
            pointCount: (current?.pointCount ?? 0) + 1,
            actualPointCount: (current?.actualPointCount ?? 0) + (pointSource === "actual" ? 1 : 0),
            mixedPointCount: (current?.mixedPointCount ?? 0) + (pointSource === "mixed" ? 1 : 0),
        });
    }

    return [...bySlot.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([startMinute, bucket]) => ({
            startMinute,
            endMinute: startMinute + FORECAST_DISPLAY_SLOT_MINUTES,
            startTs: bucket.startTs,
            endTs: bucket.endTs,
            count: Math.max(0, Math.round(bucket.sumCount / Math.max(1, bucket.pointCount))),
            percent: null,
            source: bucket.mixedPointCount > 0
                ? "mixed"
                : bucket.actualPointCount >= bucket.pointCount
                    ? "actual"
                    : "predicted",
            level: getLevelAtTimestamp(bucket.startTs, fallbackRanges) ?? "unknown",
        }));
};

const clipRangeToOpenWindows = (
    startValue: string,
    endValue: string,
    windows: FacilityOpenWindow[],
    dateKey: string
): Array<{start: string; end: string}> => {
    const startTs = getChicagoTimestampMs(startValue);
    const endTs = getChicagoTimestampMs(endValue);
    if (startTs === null || endTs === null || endTs <= startTs) return [];

    const clipped: Array<{start: string; end: string}> = [];
    for (const window of windows) {
        const windowStartTs = getChicagoTimestampForDateMinute(dateKey, window.startMinutes);
        const windowEndTs = getChicagoTimestampForDateMinute(dateKey, window.endMinutes);
        if (windowStartTs === null || windowEndTs === null || windowEndTs <= windowStartTs) continue;

        const clipStartTs = Math.max(startTs, windowStartTs);
        const clipEndTs = Math.min(endTs, windowEndTs);
        if (clipEndTs <= clipStartTs) continue;

        clipped.push({
            start: new Date(clipStartTs).toISOString(),
            end: new Date(clipEndTs).toISOString(),
        });
    }

    return clipped;
};

export const clipBandsToWorkingHours = (
    bands: CrowdBand[],
    openWindows: FacilityOpenWindow[],
    enforceWorkingHours: boolean,
    dateKey: string | null | undefined
): CrowdBand[] => {
    if (!enforceWorkingHours) return bands;
    if (openWindows.length === 0 || !dateKey) return [];

    return bands.flatMap((band) => (
        clipRangeToOpenWindows(band.start, band.end, openWindows, dateKey)
            .map((range) => ({
                start: range.start,
                end: range.end,
                level: band.level,
            }))
    ));
};

export const filterWindowsToWorkingHours = <T extends {start?: string; end?: string}>(
    windows: T[],
    openWindows: FacilityOpenWindow[],
    enforceWorkingHours: boolean,
    dateKey: string | null | undefined
): T[] => {
    if (!enforceWorkingHours) return windows;
    if (openWindows.length === 0 || !dateKey) return [];

    return windows.flatMap((window) => {
        if (!window.start || !window.end) return [];
        return clipRangeToOpenWindows(window.start, window.end, openWindows, dateKey)
            .map((range) => ({
                ...window,
                start: range.start,
                end: range.end,
            }));
    });
};
