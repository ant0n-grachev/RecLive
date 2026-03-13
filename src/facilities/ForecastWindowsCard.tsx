import {type TouchEvent, useEffect, useMemo, useRef, useState} from "react";
import {Alert, Box, Button, CircularProgress, Collapse, IconButton, Stack, Typography} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {AnimatePresence, motion, useReducedMotion} from "framer-motion";
import ModernCard from "../shared/components/ModernCard";
import type {ForecastDay} from "../lib/types/forecast";
import type {FacilityHoursFacilityPayload} from "../lib/types/facilitySchedule";
import {formatChicagoTime, getChicagoTimestampMs} from "../shared/utils/chicagoTime";
import {
    getOccupancyTone,
    INNER_SURFACE_SX,
    OCCUPANCY_MAIN_HEX,
    OCCUPANCY_SOFT_BG,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {getFacilityOpenWindowsForDate, type FacilityOpenWindow} from "../shared/utils/facilityScheduleStatus";

type CrowdBandLevel = "low" | "medium" | "peak";
type HistogramBandLevel = CrowdBandLevel | "unknown";

interface CrowdBand {
    start: string;
    end: string;
    level: CrowdBandLevel;
}

type ForecastDayWithBands = ForecastDay & {
    crowdBands?: CrowdBand[];
};

interface Props {
    day: ForecastDayWithBands | null;
    comparisonDays?: ForecastDayWithBands[];
    schedule?: FacilityHoursFacilityPayload | null;
    occupancyThresholds?: OccupancyThresholds | null;
    dayOffset: number;
    totalDays: number;
    canPrev: boolean;
    canNext: boolean;
    onPrev: () => void;
    onNext: () => void;
    isLoading: boolean;
    error: string | null;
}

const BAND_LEVEL_ORDER: CrowdBandLevel[] = ["low", "medium", "peak"];
const SWIPE_THRESHOLD_PX = 52;
const SWIPE_VERTICAL_TOLERANCE_PX = 24;
const SWIPE_INTENT_THRESHOLD_PX = 12;
const SWIPE_INTENT_BIAS_PX = 6;

const dayContentVariants = {
    enter: (direction: 1 | -1) => ({
        x: direction > 0 ? 36 : -36,
        opacity: 0,
    }),
    center: {
        x: 0,
        opacity: 1,
    },
    exit: (direction: 1 | -1) => ({
        x: direction > 0 ? -28 : 28,
        opacity: 0,
    }),
};

const formatTime = (value?: string): string => {
    const formatted = formatChicagoTime(value);
    return formatted ?? "N/A";
};

const formatRange = (start?: string, end?: string): string => {
    const formattedStart = formatTime(start);
    const formattedEnd = formatTime(end);
    return `${formattedStart} – ${formattedEnd}`;
};

const BAND_STYLES: Record<CrowdBandLevel, {label: string; color: string; bg: string}> = {
    low: {label: "LOW CROWD", color: OCCUPANCY_MAIN_HEX.success, bg: OCCUPANCY_SOFT_BG.success},
    medium: {label: "MEDIUM CROWD", color: OCCUPANCY_MAIN_HEX.warning, bg: OCCUPANCY_SOFT_BG.warning},
    peak: {label: "PEAK CROWD", color: OCCUPANCY_MAIN_HEX.error, bg: OCCUPANCY_SOFT_BG.error},
};
const UNKNOWN_BAND_STYLE = {
    label: "BAND UNAVAILABLE",
    color: "#64748b",
    bg: "rgba(100, 116, 139, 0.16)",
} as const;
const getHistogramBandStyle = (level: HistogramBandLevel) => (
    level === "unknown" ? UNKNOWN_BAND_STYLE : BAND_STYLES[level]
);

const isHistogramLevelVisible = (
    level: HistogramBandLevel,
    selected: Set<CrowdBandLevel>,
    showDefault: boolean
): boolean => {
    if (showDefault) return true;
    if (level === "unknown") return false;
    return selected.has(level);
};

const getVisibleHistogramSegments = (
    bar: HistogramBar,
    selected: Set<CrowdBandLevel>,
    showDefault: boolean
): [boolean, boolean] => [
    isHistogramLevelVisible(bar.segmentLevels[0], selected, showDefault),
    isHistogramLevelVisible(bar.segmentLevels[1], selected, showDefault),
];

const HISTOGRAM_MARGIN_LEFT = 24;
const HISTOGRAM_MARGIN_RIGHT = 2;
const HISTOGRAM_MARGIN_TOP = 30;
const HISTOGRAM_MARGIN_BOTTOM = 70;
const HISTOGRAM_PLOT_HEIGHT = 146;
const HISTOGRAM_SLOT_WIDTH = 22;
const HISTOGRAM_TICK_TARGET = 5;
const HISTOGRAM_HOUR_LABEL_FONT_SIZE = 9;
const HISTOGRAM_TREND_MIN_NEIGHBOR_TOLERANCE = 18;
const HISTOGRAM_TREND_NEIGHBOR_TOLERANCE_RATIO = 0.18;
const HISTOGRAM_TREND_MIN_DEVIATION = 32;
const HISTOGRAM_TREND_DEVIATION_RATIO = 0.3;
const FORECAST_DISPLAY_SLOT_MINUTES = 30;
const MINUTES_PER_DAY = 24 * 60;
const CROWD_BAND_BRIDGE_MINUTES = 90;
const CROWD_BAND_CONTIGUITY_TOLERANCE_MS = 60 * 1000;
const EMPTY_CROWD_BANDS: CrowdBand[] = [];

interface ForecastDisplaySlot {
    startMinute: number;
    endMinute: number;
    startTs: number;
    endTs: number;
    count: number;
    percent: number | null;
    source: "actual" | "predicted" | "mixed";
    level: HistogramBandLevel;
}

interface HistogramBar {
    startMinute: number;
    endMinute: number;
    axisLabel: string;
    rangeLabel: string;
    showAxisLabel: boolean;
    hasSplit: boolean;
    segmentRangeLabels: [string, string];
    segmentCounts: [number, number];
    segmentLevels: [HistogramBandLevel, HistogramBandLevel];
    x: number;
    centerX: number;
    y: number;
    width: number;
    height: number;
    count: number;
    rawCount: number;
    wasSmoothed: boolean;
    source: "actual" | "predicted" | "mixed";
    level: HistogramBandLevel;
}

interface HistogramTick {
    value: number;
    y: number;
}

interface HistogramModel {
    bars: HistogramBar[];
    yTicks: HistogramTick[];
    yMax: number;
    maxCount: number;
    viewBoxWidth: number;
    viewBoxHeight: number;
    plotLeft: number;
    plotRight: number;
    plotTop: number;
    baselineY: number;
    rotateHourLabels: boolean;
    actualBarCount: number;
    predictedBarCount: number;
    mixedBarCount: number;
    unknownBarCount: number;
    smoothedBarCount: number;
}

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

const sortBands = (bands: CrowdBand[]): CrowdBand[] =>
    bands
        .slice()
        .sort((a, b) => {
            const left = getChicagoTimestampMs(a.start);
            const right = getChicagoTimestampMs(b.start);
            if (left === null && right === null) return 0;
            if (left === null) return 1;
            if (right === null) return -1;
            return left - right;
        });

const mergeAdjacentCrowdBands = (bands: CrowdBand[]): CrowdBand[] => {
    const sorted = sortBands(bands);
    if (sorted.length <= 1) return sorted;

    const merged: CrowdBand[] = [];
    for (const band of sorted) {
        const previous = merged[merged.length - 1];
        if (!previous) {
            merged.push({...band});
            continue;
        }

        const previousEndTs = getChicagoTimestampMs(previous.end);
        const currentStartTs = getChicagoTimestampMs(band.start);
        const isContiguous = previousEndTs !== null
            && currentStartTs !== null
            && Math.abs(currentStartTs - previousEndTs) <= CROWD_BAND_CONTIGUITY_TOLERANCE_MS;

        if (previous.level === band.level && isContiguous) {
            previous.end = band.end;
            continue;
        }

        merged.push({...band});
    }

    return merged;
};

const smoothCrowdBandBridges = (bands: CrowdBand[]): CrowdBand[] => {
    let smoothed = mergeAdjacentCrowdBands(bands);
    if (smoothed.length < 3 || CROWD_BAND_BRIDGE_MINUTES <= 0) {
        return smoothed;
    }

    const maxBridgeDurationMs = CROWD_BAND_BRIDGE_MINUTES * 60 * 1000;

    while (true) {
        let changed = false;
        const nextBands = smoothed.map((band) => ({...band}));

        for (let index = 1; index < smoothed.length - 1; index += 1) {
            const previous = smoothed[index - 1];
            const current = smoothed[index];
            const next = smoothed[index + 1];

            if (previous.level !== next.level || current.level === previous.level) {
                continue;
            }

            const previousEndTs = getChicagoTimestampMs(previous.end);
            const currentStartTs = getChicagoTimestampMs(current.start);
            const currentEndTs = getChicagoTimestampMs(current.end);
            const nextStartTs = getChicagoTimestampMs(next.start);
            if (
                previousEndTs === null
                || currentStartTs === null
                || currentEndTs === null
                || nextStartTs === null
            ) {
                continue;
            }

            const isContiguous = Math.abs(currentStartTs - previousEndTs) <= CROWD_BAND_CONTIGUITY_TOLERANCE_MS
                && Math.abs(nextStartTs - currentEndTs) <= CROWD_BAND_CONTIGUITY_TOLERANCE_MS;
            if (!isContiguous) {
                continue;
            }

            const durationMs = currentEndTs - currentStartTs;
            if (durationMs > maxBridgeDurationMs) {
                continue;
            }

            nextBands[index] = {
                ...current,
                level: previous.level,
            };
            changed = true;
        }

        if (!changed) {
            return smoothed;
        }

        smoothed = mergeAdjacentCrowdBands(nextBands);
    }
};

const occupancyToneToBandLevel = (tone: ReturnType<typeof getOccupancyTone>): CrowdBandLevel | null => {
    if (tone === "success") return "low";
    if (tone === "warning") return "medium";
    if (tone === "error") return "peak";
    return null;
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

const buildCrowdBandsFromDisplaySlots = (slots: ForecastDisplaySlot[]): CrowdBand[] => {
    const bands = slots
        .filter((slot): slot is ForecastDisplaySlot & {level: CrowdBandLevel} => slot.level !== "unknown")
        .map((slot) => ({
            start: new Date(slot.startTs).toISOString(),
            end: new Date(slot.endTs).toISOString(),
            level: slot.level,
        }));

    return smoothCrowdBandBridges(bands);
};

const applyCrowdBandsToDisplaySlots = (
    slots: ForecastDisplaySlot[],
    bands: CrowdBand[]
): ForecastDisplaySlot[] => {
    if (slots.length === 0 || bands.length === 0) {
        return slots;
    }

    const ranges = buildBandTimeRanges(bands);
    if (ranges.length === 0) {
        return slots;
    }

    return slots.map((slot) => {
        const midpointTs = slot.startTs + ((slot.endTs - slot.startTs) / 2);
        const level = getLevelAtTimestamp(midpointTs, ranges) ?? slot.level;
        return {
            ...slot,
            level,
        };
    });
};

const buildForecastDisplaySlots = (
    day: ForecastDayWithBands | null,
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

        const pointSource: HistogramBar["source"] = point.actualCategoryCount <= 0
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

const isCurrentBand = (band: CrowdBand, nowTs: number): boolean => {
    const startTs = getChicagoTimestampMs(band.start);
    const endTs = getChicagoTimestampMs(band.end);
    if (startTs === null || endTs === null) return false;
    return nowTs >= startTs && nowTs < endTs;
};

const renderBands = (
    bands: CrowdBand[],
    nowTs: number,
    isDark: boolean
) => {
    const sorted = sortBands(bands);

    if (sorted.length === 0) {
        return (
            <Typography variant="body2" sx={{fontWeight: 600}} color="text.secondary">
                No matching intervals.
            </Typography>
        );
    }

    return (
        <Stack spacing={0.8}>
            {sorted.map((band, index) => {
                const style = BAND_STYLES[band.level] ?? BAND_STYLES.medium;
                const isCurrent = isCurrentBand(band, nowTs);
                return (
                    <Box
                        key={`${band.start}-${band.end}-${index}`}
                        sx={{
                            ...INNER_SURFACE_SX,
                            p: 1,
                            borderColor: isCurrent
                                ? (isDark ? "rgba(255, 255, 255, 0.72)" : "rgba(15, 23, 42, 0.6)")
                                : "divider",
                            borderWidth: isCurrent ? 1.5 : 1,
                            boxShadow: "none",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 1,
                            flexWrap: "wrap",
                        }}
                    >
                        <Typography variant="body2" sx={{fontWeight: 700, color: "text.primary"}}>
                            {formatRange(band.start, band.end)}
                        </Typography>
                        <Box
                            sx={{
                                px: 1.15,
                                minHeight: 24,
                                minWidth: 132,
                                borderRadius: 999,
                                bgcolor: style.bg,
                                color: style.color,
                                display: "inline-flex",
                                alignItems: "center",
                                justifyContent: "center",
                            }}
                        >
                            <Typography
                                variant="caption"
                                sx={{
                                    fontWeight: 800,
                                    letterSpacing: 0.3,
                                    textTransform: "uppercase",
                                    lineHeight: 1,
                                }}
                            >
                                {style.label}
                            </Typography>
                        </Box>
                    </Box>
                );
            })}
        </Stack>
    );
};

const formatWindow = (window: {start?: string; end?: string}): string => {
    const start = formatTime(window.start);
    const end = formatTime(window.end);
    return `${start} – ${end}`;
};

const parseShortDate = (value?: string | null): string | null => {
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

const getChicagoTimestampForDateMinute = (dateKey: string, minuteOffset: number): number | null => {
    const dayShift = Math.floor(minuteOffset / MINUTES_PER_DAY);
    const minuteInDay = ((minuteOffset % MINUTES_PER_DAY) + MINUTES_PER_DAY) % MINUTES_PER_DAY;
    const shiftedDate = shiftDateKeyByDays(dateKey, dayShift);
    if (!shiftedDate) return null;

    const hour = Math.floor(minuteInDay / 60);
    const minute = minuteInDay % 60;
    return getChicagoTimestampMs(`${shiftedDate}T${pad2(hour)}:${pad2(minute)}:00`);
};

const formatMinuteLabel = (minuteOfDay: number, includeMinutes: boolean): string => {
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

const formatHistogramRangeLabel = (startMinute: number, endMinute: number): string => {
    return `${formatMinuteLabel(startMinute, true)} – ${formatMinuteLabel(endMinute, true)}`;
};

const getChicagoMinuteOfDayFromTimestamp = (timestampMs: number): number | null => {
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

const getChicagoDateKeyFromTimestamp = (timestampMs: number): string | null => {
    const parts = chicagoDateFormatter.formatToParts(new Date(timestampMs));
    const year = parts.find((part) => part.type === "year")?.value;
    const month = parts.find((part) => part.type === "month")?.value;
    const day = parts.find((part) => part.type === "day")?.value;
    if (!year || !month || !day) return null;
    return `${year}-${month}-${day}`;
};

const buildBandTimeRanges = (bands: CrowdBand[]): Array<{startTs: number; endTs: number; level: CrowdBandLevel}> =>
    bands
        .map((band) => {
            const startTs = getChicagoTimestampMs(band.start);
            const endTs = getChicagoTimestampMs(band.end);
            if (startTs === null || endTs === null || endTs <= startTs) return null;
            return {startTs, endTs, level: band.level};
        })
        .filter((range): range is {startTs: number; endTs: number; level: CrowdBandLevel} => Boolean(range))
        .sort((a, b) => a.startTs - b.startTs);

const getLevelAtTimestamp = (
    timestampMs: number,
    ranges: Array<{startTs: number; endTs: number; level: CrowdBandLevel}>
): CrowdBandLevel | null => {
    for (const range of ranges) {
        if (timestampMs >= range.startTs && timestampMs < range.endTs) {
            return range.level;
        }
    }
    return null;
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

const clipBandsToWorkingHours = (
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

const filterWindowsToWorkingHours = (
    windows: Array<{start?: string; end?: string}>,
    openWindows: FacilityOpenWindow[],
    enforceWorkingHours: boolean,
    dateKey: string | null | undefined
) => {
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

const getNiceTickStep = (maxValue: number, targetTicks = HISTOGRAM_TICK_TARGET): number => {
    if (!Number.isFinite(maxValue) || maxValue <= 0) return 1;

    const roughStep = maxValue / Math.max(1, targetTicks);
    const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
    const residual = roughStep / magnitude;

    if (residual <= 1) return magnitude;
    if (residual <= 2) return 2 * magnitude;
    if (residual <= 5) return 5 * magnitude;
    return 10 * magnitude;
};

const inferDisplayMaxCapacity = (slots: ForecastDisplaySlot[]): number | null => {
    const candidates = slots
        .map((slot) => {
            if (
                typeof slot.percent !== "number"
                || !Number.isFinite(slot.percent)
                || slot.percent <= 0
                || slot.count <= 0
            ) {
                return null;
            }
            return slot.count / slot.percent;
        })
        .filter((candidate): candidate is number => (
            candidate !== null
            && Number.isFinite(candidate)
            && candidate > 0
        ))
        .sort((left, right) => left - right);

    if (candidates.length === 0) {
        return null;
    }

    const middleIndex = Math.floor(candidates.length / 2);
    const median = candidates.length % 2 === 0
        ? (candidates[middleIndex - 1] + candidates[middleIndex]) / 2
        : candidates[middleIndex];

    return median > 0 ? median : null;
};

const smoothIsolatedHourlyCounts = (
    counts: number[]
): {counts: number[]; smoothedIndices: Set<number>} => {
    if (counts.length < 3) {
        return {counts, smoothedIndices: new Set<number>()};
    }

    const smoothed = counts.slice();
    const smoothedIndices = new Set<number>();

    for (let index = 1; index < counts.length - 1; index += 1) {
        const previous = counts[index - 1];
        const current = counts[index];
        const next = counts[index + 1];
        const neighborAverage = (previous + next) / 2;
        const neighborDelta = Math.abs(previous - next);
        const neighborTolerance = Math.max(
            HISTOGRAM_TREND_MIN_NEIGHBOR_TOLERANCE,
            neighborAverage * HISTOGRAM_TREND_NEIGHBOR_TOLERANCE_RATIO
        );

        if (neighborDelta > neighborTolerance) {
            continue;
        }

        const isLocalDip = current < previous && current < next;
        const isLocalSpike = current > previous && current > next;
        if (!isLocalDip && !isLocalSpike) {
            continue;
        }

        const deviation = Math.abs(current - neighborAverage);
        const deviationThreshold = Math.max(
            HISTOGRAM_TREND_MIN_DEVIATION,
            neighborAverage * HISTOGRAM_TREND_DEVIATION_RATIO
        );
        if (deviation < deviationThreshold) {
            continue;
        }

        smoothed[index] = Math.max(0, Math.round(neighborAverage));
        smoothedIndices.add(index);
    }

    return {counts: smoothed, smoothedIndices};
};

const buildHistogramModel = (
    slots: ForecastDisplaySlot[],
    scaleMaxCountOverride: number | null = null,
    thresholds: OccupancyThresholds | null | undefined = null
): HistogramModel | null => {
    if (slots.length === 0) return null;

    const byHour = new Map<number, [ForecastDisplaySlot | null, ForecastDisplaySlot | null]>();
    for (const slot of slots) {
        const hourStartMinute = Math.floor(slot.startMinute / 60) * 60;
        const segmentIndex = slot.startMinute % 60 >= FORECAST_DISPLAY_SLOT_MINUTES ? 1 : 0;
        const current = byHour.get(hourStartMinute) ?? [null, null];
        current[segmentIndex] = slot;
        byHour.set(hourStartMinute, current);
    }

    const hourBuckets = [...byHour.entries()].sort((a, b) => a[0] - b[0]);
    const inferredMaxCapacity = inferDisplayMaxCapacity(slots);
    const rawHourlyCounts = hourBuckets.map(([, segments]) => {
        const segmentCounts = segments
            .filter((segment): segment is ForecastDisplaySlot => segment !== null)
            .map((segment) => segment.count);
        if (segmentCounts.length === 0) return 0;
        const hourlyAverage = segmentCounts.reduce((sum, count) => sum + count, 0) / segmentCounts.length;
        return Math.max(0, Math.round(hourlyAverage));
    });
    const {counts: smoothedHourlyCounts, smoothedIndices} = smoothIsolatedHourlyCounts(rawHourlyCounts);
    const maxCount = smoothedHourlyCounts.reduce((max, count) => Math.max(max, count), 0);
    const scaleMaxCount = Math.max(maxCount, scaleMaxCountOverride ?? 0);
    const yTickStep = getNiceTickStep(scaleMaxCount);
    const yMax = Math.max(yTickStep, Math.ceil(scaleMaxCount / yTickStep) * yTickStep);
    const yTicks: HistogramTick[] = [];
    for (let value = 0; value <= yMax; value += yTickStep) {
        const ratio = yMax > 0 ? value / yMax : 0;
        const y = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT - (ratio * HISTOGRAM_PLOT_HEIGHT);
        yTicks.push({value, y});
    }

    const plotWidth = Math.max(220, hourBuckets.length * HISTOGRAM_SLOT_WIDTH);
    const viewBoxWidth = HISTOGRAM_MARGIN_LEFT + plotWidth + HISTOGRAM_MARGIN_RIGHT;
    const viewBoxHeight = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT + HISTOGRAM_MARGIN_BOTTOM;
    const baselineY = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT;
    const barWidth = HISTOGRAM_SLOT_WIDTH;
    const bars: HistogramBar[] = hourBuckets.map(([hourStartMinute, segments], index) => {
        const leftSegment = segments[0] ?? segments[1];
        const rightSegment = segments[1] ?? segments[0];
        const rawCount = rawHourlyCounts[index] ?? 0;
        const count = smoothedHourlyCounts[index] ?? rawCount;
        const height = yMax > 0 ? (count / yMax) * HISTOGRAM_PLOT_HEIGHT : 0;
        const x = HISTOGRAM_MARGIN_LEFT + (index * HISTOGRAM_SLOT_WIDTH);
        const centerX = x + (barWidth / 2);
        const segmentSources = segments
            .filter((segment): segment is ForecastDisplaySlot => segment !== null)
            .map((segment) => segment.source);
        const source: HistogramBar["source"] = segmentSources.length === 0
            ? "predicted"
            : segmentSources.every((segmentSource) => segmentSource === "actual")
                ? "actual"
                : segmentSources.every((segmentSource) => segmentSource === "predicted")
                    ? "predicted"
                    : "mixed";
        const rawSegmentLevels: [HistogramBandLevel, HistogramBandLevel] = [
            leftSegment?.level ?? "unknown",
            rightSegment?.level ?? "unknown",
        ];
        const smoothedLevel = (
            smoothedIndices.has(index)
            && thresholds
            && inferredMaxCapacity
            && inferredMaxCapacity > 0
        )
            ? occupancyToneToBandLevel(
                getOccupancyTone((count / inferredMaxCapacity) * 100, thresholds)
            )
            : null;
        const segmentLevels: [HistogramBandLevel, HistogramBandLevel] = smoothedLevel
            ? [smoothedLevel, smoothedLevel]
            : rawSegmentLevels;
        const hasSplit = Boolean(
            segments[0]
            && segments[1]
            && segmentLevels[0] !== segmentLevels[1]
        );
        return {
            startMinute: hourStartMinute,
            endMinute: hourStartMinute + 60,
            axisLabel: formatMinuteLabel(hourStartMinute, false),
            rangeLabel: formatHistogramRangeLabel(hourStartMinute, hourStartMinute + 60),
            showAxisLabel: true,
            hasSplit,
            segmentRangeLabels: [
                formatHistogramRangeLabel(hourStartMinute, hourStartMinute + FORECAST_DISPLAY_SLOT_MINUTES),
                formatHistogramRangeLabel(hourStartMinute + FORECAST_DISPLAY_SLOT_MINUTES, hourStartMinute + 60),
            ],
            segmentCounts: [
                leftSegment?.count ?? 0,
                rightSegment?.count ?? 0,
            ],
            segmentLevels,
            x,
            centerX,
            y: baselineY - height,
            width: barWidth,
            height,
            count,
            rawCount,
            wasSmoothed: smoothedIndices.has(index),
            source,
            level: segmentLevels[1] !== "unknown" ? segmentLevels[1] : segmentLevels[0],
        };
    });
    const actualBarCount = bars.filter((bar) => bar.source === "actual").length;
    const predictedBarCount = bars.filter((bar) => bar.source === "predicted").length;
    const mixedBarCount = bars.filter((bar) => bar.source === "mixed").length;
    const unknownBarCount = bars.filter((bar) => bar.segmentLevels.includes("unknown")).length;
    const smoothedBarCount = bars.filter((bar) => bar.wasSmoothed).length;

    return {
        bars,
        yTicks,
        yMax,
        maxCount,
        viewBoxWidth,
        viewBoxHeight,
        plotLeft: HISTOGRAM_MARGIN_LEFT,
        plotRight: HISTOGRAM_MARGIN_LEFT + plotWidth,
        plotTop: HISTOGRAM_MARGIN_TOP,
        baselineY,
        rotateHourLabels: bars.length >= 10,
        actualBarCount,
        predictedBarCount,
        mixedBarCount,
        unknownBarCount,
        smoothedBarCount,
    };
};

export default function ForecastWindowsCard({
    day,
    comparisonDays = [],
    schedule = null,
    occupancyThresholds = null,
    dayOffset,
    totalDays,
    canPrev,
    canNext,
    onPrev,
    onNext,
    isLoading,
    error,
}: Props) {
    const theme = useTheme();
    const isDark = theme.palette.mode === "dark";
    const neutralButtonBg = isDark ? alpha(theme.palette.common.white, 0.06) : "#ffffff";
    const neutralButtonBorder = alpha(theme.palette.text.primary, isDark ? 0.32 : 0.22);
    const neutralButtonHoverBg = isDark ? alpha(theme.palette.common.white, 0.1) : "rgba(15, 23, 42, 0.03)";
    const selectedBarStroke = alpha(theme.palette.text.primary, isDark ? 0.96 : 0.86);
    const barSeparatorStroke = alpha(theme.palette.text.primary, isDark ? 0.82 : 0.6);
    const nowMarkerStroke = alpha(theme.palette.text.primary, isDark ? 0.96 : 0.9);
    const nowMarkerText = alpha(theme.palette.text.primary, isDark ? 0.98 : 0.94);
    const axisTickColor = alpha(theme.palette.text.primary, isDark ? 0.5 : 0.3);
    const axisLabelColor = alpha(theme.palette.text.primary, isDark ? 0.88 : 0.75);
    const axisValueColor = alpha(theme.palette.text.primary, isDark ? 0.62 : 0.5);
    const axisBaselineColor = alpha(theme.palette.text.primary, isDark ? 0.45 : 0.25);
    const histogramGridLineColor = alpha(theme.palette.text.primary, isDark ? 0.12 : 0.08);
    const reduceMotion = useReducedMotion();
    const [selectedLevels, setSelectedLevels] = useState<CrowdBandLevel[]>([]);
    const [nowTs, setNowTs] = useState<number>(() => Date.now());
    const [slideDirection, setSlideDirection] = useState<1 | -1>(1);
    const [trendExpanded, setTrendExpanded] = useState(false);
    const [selectedHistogramSelection, setSelectedHistogramSelection] = useState<{
        date: string | null;
        startMinute: number | null;
    }>({date: null, startMinute: null});
    const swipeAreaRef = useRef<HTMLDivElement | null>(null);
    const touchStartRef = useRef<{x: number; y: number} | null>(null);
    const swipeIntentRef = useRef<"pending" | "horizontal" | "vertical">("pending");
    const horizontalLockRef = useRef(false);

    useEffect(() => {
        const timerId = window.setInterval(() => {
            setNowTs(Date.now());
        }, 30000);
        return () => window.clearInterval(timerId);
    }, []);

    const toggleLevel = (level: CrowdBandLevel) => {
        setSelectedLevels((prev) => (
            prev.includes(level)
                ? prev.filter((item) => item !== level)
                : [...prev, level]
        ));
    };

    const openWindows = useMemo(
        () => getFacilityOpenWindowsForDate(schedule, day?.date ?? null),
        [schedule, day?.date]
    );

    const hasWorkingHours = Boolean(schedule && day?.date);
    const fallbackCrowdBands = day?.crowdBands ?? EMPTY_CROWD_BANDS;

    const displaySlots = useMemo(
        () => buildForecastDisplaySlots(
            day,
            openWindows,
            hasWorkingHours,
            occupancyThresholds,
            fallbackCrowdBands,
            nowTs
        ),
        [day, fallbackCrowdBands, hasWorkingHours, nowTs, occupancyThresholds, openWindows]
    );

    const slotDerivedBands = useMemo(
        () => buildCrowdBandsFromDisplaySlots(displaySlots),
        [displaySlots]
    );
    const chartDisplaySlots = useMemo(
        () => applyCrowdBandsToDisplaySlots(displaySlots, slotDerivedBands),
        [displaySlots, slotDerivedBands]
    );
    const histogramScaleMaxCount = useMemo(() => {
        const daysForScale = comparisonDays.length > 0 ? comparisonDays : (day ? [day] : []);
        return daysForScale.reduce((globalMax, comparisonDay) => {
            const comparisonOpenWindows = getFacilityOpenWindowsForDate(schedule, comparisonDay?.date ?? null);
            const comparisonHasWorkingHours = Boolean(schedule && comparisonDay?.date);
            const comparisonFallbackCrowdBands = comparisonDay?.crowdBands ?? EMPTY_CROWD_BANDS;
            const comparisonDisplaySlots = buildForecastDisplaySlots(
                comparisonDay,
                comparisonOpenWindows,
                comparisonHasWorkingHours,
                occupancyThresholds,
                comparisonFallbackCrowdBands,
                nowTs
            );
            const comparisonBands = buildCrowdBandsFromDisplaySlots(comparisonDisplaySlots);
            const comparisonChartSlots = applyCrowdBandsToDisplaySlots(comparisonDisplaySlots, comparisonBands);
            const comparisonHistogram = buildHistogramModel(
                comparisonChartSlots,
                null,
                occupancyThresholds
            );
            return Math.max(globalMax, comparisonHistogram?.maxCount ?? 0);
        }, 0);
    }, [comparisonDays, day, schedule, occupancyThresholds, nowTs]);

    const workingHoursBands = useMemo(() => {
        if (slotDerivedBands.length > 0) {
            return slotDerivedBands;
        }
        return clipBandsToWorkingHours(fallbackCrowdBands, openWindows, hasWorkingHours, day?.date ?? null);
    }, [day?.date, fallbackCrowdBands, hasWorkingHours, openWindows, slotDerivedBands]);

    const displayBands = useMemo(() => {
        const allBands = workingHoursBands;
        const showDefault =
            selectedLevels.length === 0 || selectedLevels.length === BAND_LEVEL_ORDER.length;

        if (showDefault) {
            return sortBands(allBands);
        }

        const selected = new Set(selectedLevels);
        return sortBands(allBands).filter((band) => selected.has(band.level));
    }, [workingHoursBands, selectedLevels]);
    const showAllHistogramLevels =
        selectedLevels.length === 0 || selectedLevels.length === BAND_LEVEL_ORDER.length;
    const selectedLevelSet = useMemo(
        () => new Set(selectedLevels),
        [selectedLevels]
    );

    const histogram = useMemo(
        () => buildHistogramModel(chartDisplaySlots, histogramScaleMaxCount, occupancyThresholds),
        [chartDisplaySlots, histogramScaleMaxCount, occupancyThresholds]
    );

    const filteredBestWindows = useMemo(
        () => filterWindowsToWorkingHours(day?.bestWindows ?? [], openWindows, hasWorkingHours, day?.date ?? null),
        [day?.bestWindows, day?.date, openWindows, hasWorkingHours]
    );

    const filteredAvoidWindows = useMemo(
        () => filterWindowsToWorkingHours(day?.avoidWindows ?? [], openWindows, hasWorkingHours, day?.date ?? null),
        [day?.avoidWindows, day?.date, openWindows, hasWorkingHours]
    );
    const selectedHistogramStartMinute = (
        selectedHistogramSelection.date === (day?.date ?? null)
            ? selectedHistogramSelection.startMinute
            : null
    );
    const selectedHistogramBar = useMemo(
        () => (
            histogram?.bars.find((bar) => {
                if (bar.startMinute !== selectedHistogramStartMinute) {
                    return false;
                }
                const [leftVisible, rightVisible] = getVisibleHistogramSegments(
                    bar,
                    selectedLevelSet,
                    showAllHistogramLevels
                );
                return leftVisible || rightVisible;
            }) ?? null
        ),
        [histogram, selectedHistogramStartMinute, selectedLevelSet, showAllHistogramLevels]
    );
    const currentTimeMarker = useMemo(() => {
        if (!histogram || !day?.date) return null;

        const todayDateKey = getChicagoDateKeyFromTimestamp(nowTs);
        if (!todayDateKey || day.date !== todayDateKey) return null;

        const nowMinuteOfDay = getChicagoMinuteOfDayFromTimestamp(nowTs);
        if (nowMinuteOfDay === null) return null;

        const activeBar = histogram.bars.find((bar) => (
            nowMinuteOfDay >= bar.startMinute && nowMinuteOfDay < bar.endMinute
        ));
        if (!activeBar) return null;
        const [leftVisible, rightVisible] = getVisibleHistogramSegments(
            activeBar,
            selectedLevelSet,
            showAllHistogramLevels
        );
        if (!leftVisible && !rightVisible) return null;

        const minuteInSlot = nowMinuteOfDay - activeBar.startMinute;
        const slotDuration = Math.max(1, activeBar.endMinute - activeBar.startMinute);
        return {
            x: activeBar.x + ((minuteInSlot / slotDuration) * activeBar.width),
            label: "Now",
        };
    }, [day, histogram, nowTs, selectedLevelSet, showAllHistogramLevels]);

    const shortDate = parseShortDate(day?.date);
    const shortWeekday = day?.dayName ? day.dayName.slice(0, 3) : null;
    const slideVariants = useMemo(
        () => (reduceMotion
            ? {
                enter: {x: 0, opacity: 1},
                center: {x: 0, opacity: 1},
                exit: {x: 0, opacity: 1},
            }
            : dayContentVariants),
        [reduceMotion]
    );
    const title = (() => {
        if (dayOffset === 0) return "Forecast Today";
        if (dayOffset === 1) return "Forecast Tomorrow";
        if (shortWeekday && shortDate) return `Forecast ${shortWeekday} ${shortDate}`;
        if (shortWeekday) return `Forecast ${shortWeekday}`;
        if (shortDate) return `Forecast ${shortDate}`;
        return dayOffset === 2 ? "Forecast In 2 Days" : "Forecast In 3 Days";
    })();

    const handlePrevDay = () => {
        if (!canPrev || isLoading) return;
        setSlideDirection(-1);
        onPrev();
    };

    const handleNextDay = () => {
        if (!canNext || isLoading) return;
        setSlideDirection(1);
        onNext();
    };

    const handleTouchStart = (event: TouchEvent<HTMLDivElement>) => {
        if (isLoading || event.touches.length !== 1) return;
        const touch = event.touches[0];
        touchStartRef.current = {x: touch.clientX, y: touch.clientY};
        swipeIntentRef.current = "pending";
        horizontalLockRef.current = false;
        const node = swipeAreaRef.current;
        if (node) node.style.touchAction = "";
    };

    const handleTouchMove = (event: TouchEvent<HTMLDivElement>) => {
        const start = touchStartRef.current;
        if (!start || event.touches.length !== 1) return;

        const touch = event.touches[0];
        const deltaX = touch.clientX - start.x;
        const deltaY = touch.clientY - start.y;
        const absX = Math.abs(deltaX);
        const absY = Math.abs(deltaY);

        if (swipeIntentRef.current === "pending") {
            const movedEnough = absX >= SWIPE_INTENT_THRESHOLD_PX || absY >= SWIPE_INTENT_THRESHOLD_PX;
            if (!movedEnough) return;

            if (absX > absY + SWIPE_INTENT_BIAS_PX) {
                swipeIntentRef.current = "horizontal";
                if (!horizontalLockRef.current) {
                    horizontalLockRef.current = true;
                    const node = swipeAreaRef.current;
                    if (node) node.style.touchAction = "none";
                }
            } else if (absY > absX + SWIPE_INTENT_BIAS_PX) {
                swipeIntentRef.current = "vertical";
                if (horizontalLockRef.current) {
                    horizontalLockRef.current = false;
                    const node = swipeAreaRef.current;
                    if (node) node.style.touchAction = "";
                }
            } else {
                return;
            }
        }

        if (swipeIntentRef.current === "horizontal") {
            event.preventDefault();
        }
    };

    const resetTouchGesture = () => {
        touchStartRef.current = null;
        swipeIntentRef.current = "pending";
        horizontalLockRef.current = false;
        const node = swipeAreaRef.current;
        if (node) node.style.touchAction = "";
    };

    const handleTouchEnd = (event: TouchEvent<HTMLDivElement>) => {
        const start = touchStartRef.current;
        const swipeIntent = swipeIntentRef.current;
        resetTouchGesture();
        if (!start || event.changedTouches.length !== 1) return;
        if (swipeIntent !== "horizontal") return;

        const touch = event.changedTouches[0];
        const deltaX = touch.clientX - start.x;
        const deltaY = touch.clientY - start.y;
        const absX = Math.abs(deltaX);
        const absY = Math.abs(deltaY);

        if (absX < SWIPE_THRESHOLD_PX || absX < absY + SWIPE_VERTICAL_TOLERANCE_PX) {
            return;
        }

        if (deltaX < 0) {
            handleNextDay();
            return;
        }

        handlePrevDay();
    };

    const toggleHistogramSlotSelection = (startMinute: number) => {
        const currentDate = day?.date ?? null;
        setSelectedHistogramSelection((prev) => {
            const previousStartMinuteForCurrentDay = prev.date === currentDate ? prev.startMinute : null;
            return {
                date: currentDate,
                startMinute: previousStartMinuteForCurrentDay === startMinute ? null : startMinute,
            };
        });
    };

    return (
        <ModernCard>
            <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
                <Typography variant="subtitle2" color="text.secondary">
                    {title}
                </Typography>
                <Stack direction="row" spacing={0.25} sx={{display: {xs: "none", sm: "flex"}}}>
                    <IconButton
                        size="medium"
                        onClick={handlePrevDay}
                        disabled={!canPrev || isLoading}
                        aria-label="Previous forecast day"
                        sx={{width: 44, height: 44}}
                    >
                        <ChevronLeftIcon fontSize="small"/>
                    </IconButton>
                    <IconButton
                        size="medium"
                        onClick={handleNextDay}
                        disabled={!canNext || isLoading}
                        aria-label="Next forecast day"
                        sx={{width: 44, height: 44}}
                    >
                        <ChevronRightIcon fontSize="small"/>
                    </IconButton>
                </Stack>
            </Box>

            {isLoading && (
                <Box sx={{display: "flex", alignItems: "center", gap: 1, mt: 1}}>
                    <CircularProgress size={16} thickness={5}/>
                    <Typography variant="body2" color="text.secondary">
                        Loading forecast...
                    </Typography>
                </Box>
            )}

            {!isLoading && error && (
                <Alert severity="info" variant="outlined" sx={{mt: 1}}>
                    {error}
                </Alert>
            )}

            <Box
                ref={swipeAreaRef}
                data-disable-pull-refresh="true"
                onTouchStartCapture={handleTouchStart}
                onTouchMoveCapture={handleTouchMove}
                onTouchEndCapture={handleTouchEnd}
                onTouchCancelCapture={resetTouchGesture}
                sx={{
                    mt: 1,
                    touchAction: {xs: "pan-y", sm: "auto"},
                    overflowX: "hidden",
                    overscrollBehaviorX: "contain",
                    position: "relative",
                }}
            >
                <AnimatePresence mode="wait" custom={slideDirection} initial={false}>
                    {!error && day && (
                        <Box
                            key={day.date || `offset-${dayOffset}`}
                            component={motion.div}
                            custom={slideDirection}
                            variants={slideVariants}
                            initial="enter"
                            animate="center"
                            exit="exit"
                            transition={reduceMotion ? {duration: 0} : {duration: 0.24, ease: [0.22, 1, 0.36, 1]}}
                        >
                            {workingHoursBands.length > 0 ? (
                                <>
                                    <Stack direction="row" spacing={0.75} sx={{mb: 1, flexWrap: "wrap", rowGap: 0.75}}>
                                        {BAND_LEVEL_ORDER.map((level) => {
                                            const active = selectedLevels.includes(level);
                                            const style = BAND_STYLES[level];
                                            return (
                                                <Button
                                                    key={level}
                                                    size="small"
                                                    variant="outlined"
                                                    onClick={() => toggleLevel(level)}
                                                    sx={{
                                                        textTransform: "uppercase",
                                                        fontWeight: 700,
                                                        borderRadius: 999,
                                                        px: 1.15,
                                                        minWidth: 88,
                                                        minHeight: 34,
                                                        bgcolor: active ? style.bg : neutralButtonBg,
                                                        borderColor: active
                                                            ? alpha(style.color, 0.6)
                                                            : neutralButtonBorder,
                                                        color: active ? style.color : "text.secondary",
                                                        fontSize: "0.68rem",
                                                        "@media (hover: hover) and (pointer: fine)": {
                                                            "&:hover": {
                                                                borderColor: active
                                                                    ? alpha(style.color, 0.7)
                                                                    : alpha(theme.palette.text.primary, isDark ? 0.46 : 0.36),
                                                                bgcolor: active
                                                                    ? alpha(style.color, isDark ? 0.2 : 0.16)
                                                                    : neutralButtonHoverBg,
                                                            },
                                                        },
                                                    }}
                                                >
                                                    {style.label.replace(" CROWD", "")}
                                                </Button>
                                            );
                                        })}
                                    </Stack>
                                    {renderBands(displayBands, nowTs, isDark)}
                                </>
                            ) : (
                                <Stack spacing={0.5}>
                                    {filteredBestWindows.length === 0 && filteredAvoidWindows.length === 0 ? (
                                        <Typography variant="body2" color="text.secondary" sx={{fontWeight: 600}}>
                                            Forecast crowd bands are unavailable for this day.
                                        </Typography>
                                    ) : (
                                        <>
                                            {filteredBestWindows.map((window, index) => (
                                                <Typography key={`low-${index}`} variant="body2" sx={{fontWeight: 600}}>
                                                    {formatWindow(window)} (LOW CROWD)
                                                </Typography>
                                            ))}
                                            {filteredAvoidWindows.map((window, index) => (
                                                <Typography key={`peak-${index}`} variant="body2" sx={{fontWeight: 600}}>
                                                    {formatWindow(window)} (PEAK CROWD)
                                                </Typography>
                                            ))}
                                        </>
                                    )}
                                </Stack>
                            )}
                            {histogram && (
                                <Box sx={{mt: 1}}>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        onClick={() => setTrendExpanded((prev) => !prev)}
                                        endIcon={
                                            <ExpandMoreIcon
                                                sx={{
                                                    transform: trendExpanded ? "rotate(180deg)" : "rotate(0deg)",
                                                    transition: "transform 180ms ease",
                                                }}
                                            />
                                        }
                                        sx={{
                                            borderRadius: 999,
                                            textTransform: "none",
                                            fontWeight: 700,
                                            px: 1.2,
                                        }}
                                    >
                                        {trendExpanded ? "Hide hourly chart" : "Show hourly chart"}
                                    </Button>
                                    <Collapse in={trendExpanded} timeout={180} unmountOnExit>
                                        <Box
                                            sx={{
                                                ...INNER_SURFACE_SX,
                                                mt: 1,
                                                p: 1.2,
                                                borderColor: "divider",
                                                boxShadow: "none",
                                            }}
                                        >
                                            <Typography variant="caption" color="text.secondary" sx={{fontWeight: 700}}>
                                                People by hour
                                            </Typography>
                                            <Box sx={{mt: 0.75, overflowX: "hidden", pb: 0.6}}>
                                                <Box
                                                    component="svg"
                                                    viewBox={`0 0 ${histogram.viewBoxWidth} ${histogram.viewBoxHeight}`}
                                                    role="img"
                                                    aria-label="People histogram by hourly forecast bar"
                                                    sx={{
                                                        display: "block",
                                                        width: "100%",
                                                        height: {xs: 230, sm: 250},
                                                    }}
                                                >
                                                    {histogram.yTicks.map((tick, index) => (
                                                        <g key={`y-tick-${tick.value}-${index}`}>
                                                            {tick.value > 0 && (
                                                                <line
                                                                    x1={histogram.plotLeft}
                                                                    x2={histogram.plotRight}
                                                                    y1={tick.y}
                                                                    y2={tick.y}
                                                                    stroke={histogramGridLineColor}
                                                                    strokeWidth={0.45}
                                                                />
                                                            )}
                                                            <text
                                                                x={histogram.plotLeft - 3.5}
                                                                y={tick.y}
                                                                textAnchor="end"
                                                                dominantBaseline="middle"
                                                                fontSize="8"
                                                                fontWeight={700}
                                                                fill={axisValueColor}
                                                            >
                                                                {tick.value}
                                                            </text>
                                                        </g>
                                                    ))}
                                                    {histogram.bars.map((bar, index) => {
                                                        const style = getHistogramBandStyle(bar.level);
                                                        const leftStyle = getHistogramBandStyle(bar.segmentLevels[0]);
                                                        const rightStyle = getHistogramBandStyle(bar.segmentLevels[1]);
                                                        const [leftVisible, rightVisible] = getVisibleHistogramSegments(
                                                            bar,
                                                            selectedLevelSet,
                                                            showAllHistogramLevels
                                                        );
                                                        const hasVisibleSegments = leftVisible || rightVisible;
                                                        const roundedCount = Math.max(0, Math.round(bar.count));
                                                        const isSelected = selectedHistogramStartMinute === bar.startMinute;
                                                        const axisLabelY = histogram.baselineY + 22;
                                                        const halfWidth = bar.width / 2;
                                                        const outlineX = leftVisible
                                                            ? bar.x
                                                            : bar.x + halfWidth;
                                                        const outlineWidth = leftVisible && rightVisible
                                                            ? bar.width
                                                            : leftVisible
                                                                ? halfWidth
                                                                : rightVisible
                                                                    ? bar.width - halfWidth
                                                                    : 0;

                                                        return (
                                                            <g
                                                                key={`bar-${bar.startMinute}-${index}`}
                                                                role={hasVisibleSegments ? "button" : undefined}
                                                                tabIndex={hasVisibleSegments ? 0 : -1}
                                                                onClick={() => {
                                                                    if (!hasVisibleSegments) return;
                                                                    toggleHistogramSlotSelection(bar.startMinute);
                                                                }}
                                                                onKeyDown={(event) => {
                                                                    if (!hasVisibleSegments) return;
                                                                    if (event.key !== "Enter" && event.key !== " ") return;
                                                                    event.preventDefault();
                                                                    toggleHistogramSlotSelection(bar.startMinute);
                                                                }}
                                                                aria-label={`${bar.rangeLabel}, ${roundedCount} people`}
                                                                aria-hidden={!hasVisibleSegments}
                                                                style={{cursor: hasVisibleSegments ? "pointer" : "default"}}
                                                            >
                                                                {bar.hasSplit ? (
                                                                    <>
                                                                        {leftVisible && (
                                                                            <rect
                                                                                x={bar.x}
                                                                                y={bar.y}
                                                                                width={halfWidth}
                                                                                height={bar.height}
                                                                                fill={leftStyle.color}
                                                                                stroke="none"
                                                                                rx={0}
                                                                            />
                                                                        )}
                                                                        {rightVisible && (
                                                                            <rect
                                                                                x={bar.x + halfWidth}
                                                                                y={bar.y}
                                                                                width={bar.width - halfWidth}
                                                                                height={bar.height}
                                                                                fill={rightStyle.color}
                                                                                stroke="none"
                                                                                rx={0}
                                                                            />
                                                                        )}
                                                                    </>
                                                                ) : hasVisibleSegments ? (
                                                                    <rect
                                                                        x={bar.x}
                                                                        y={bar.y}
                                                                        width={bar.width}
                                                                        height={bar.height}
                                                                        fill={style.color}
                                                                        stroke="none"
                                                                        rx={0}
                                                                    />
                                                                ) : null}
                                                                {hasVisibleSegments && (
                                                                    <rect
                                                                        x={outlineX}
                                                                        y={bar.y}
                                                                        width={outlineWidth}
                                                                        height={bar.height}
                                                                        fill="none"
                                                                        stroke={isSelected ? selectedBarStroke : barSeparatorStroke}
                                                                        strokeWidth={isSelected ? 1.1 : 0.35}
                                                                        rx={1.4}
                                                                    />
                                                                )}
                                                                <line
                                                                    x1={bar.centerX}
                                                                    x2={bar.centerX}
                                                                    y1={histogram.baselineY}
                                                                    y2={histogram.baselineY + 2.3}
                                                                    stroke={axisTickColor}
                                                                    strokeWidth={0.55}
                                                                />
                                                                {bar.showAxisLabel && (
                                                                    <text
                                                                        x={bar.centerX}
                                                                        y={histogram.rotateHourLabels ? axisLabelY + 7 : axisLabelY}
                                                                        textAnchor={histogram.rotateHourLabels ? "end" : "middle"}
                                                                        dominantBaseline="hanging"
                                                                        fontSize={HISTOGRAM_HOUR_LABEL_FONT_SIZE}
                                                                        fontWeight={700}
                                                                        fill={axisLabelColor}
                                                                        transform={histogram.rotateHourLabels
                                                                            ? `rotate(-52 ${bar.centerX} ${axisLabelY + 7})`
                                                                            : undefined}
                                                                    >
                                                                        {bar.axisLabel}
                                                                    </text>
                                                                )}
                                                            </g>
                                                        );
                                                    })}
                                                    <line
                                                        x1={histogram.plotLeft}
                                                        x2={histogram.plotRight}
                                                        y1={histogram.baselineY}
                                                        y2={histogram.baselineY}
                                                        stroke={axisBaselineColor}
                                                        strokeWidth={0.95}
                                                    />
                                                    {currentTimeMarker && (
                                                        <>
                                                            <line
                                                                x1={currentTimeMarker.x}
                                                                x2={currentTimeMarker.x}
                                                                y1={12}
                                                                y2={histogram.baselineY}
                                                                stroke={nowMarkerStroke}
                                                                strokeWidth={1.15}
                                                            />
                                                            <circle
                                                                cx={currentTimeMarker.x}
                                                                cy={histogram.baselineY}
                                                                r={1.35}
                                                                fill={nowMarkerStroke}
                                                            />
                                                            <text
                                                                x={Math.max(
                                                                    histogram.plotLeft + 22,
                                                                    Math.min(histogram.plotRight - 22, currentTimeMarker.x)
                                                                )}
                                                                y={1}
                                                                textAnchor="middle"
                                                                dominantBaseline="hanging"
                                                                fontSize="9"
                                                                fontWeight={800}
                                                                fill={nowMarkerText}
                                                            >
                                                                {currentTimeMarker.label}
                                                            </text>
                                                        </>
                                                    )}
                                                </Box>
                                                <Typography variant="body2" color="text.secondary" sx={{display: "block", mt: 0.4, fontWeight: 700}}>
                                                    Max people: {Math.round(histogram.maxCount)}
                                                </Typography>
                                                {histogram.unknownBarCount > 0 && (
                                                    <Typography variant="body2" color="text.secondary" sx={{display: "block", mt: 0.25, fontWeight: 700}}>
                                                        Some half-hour colors are unavailable because occupancy thresholds were missing for one or both halves.
                                                    </Typography>
                                                )}
                                                <Typography variant="body2" color="text.primary" sx={{display: "block", mt: 0.25, fontWeight: 700}}>
                                                    {selectedHistogramBar
                                                        ? `${selectedHistogramBar.rangeLabel}: ${Math.round(selectedHistogramBar.count)}`
                                                        : "Tap a bar to inspect the hourly trend"}
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Collapse>
                                </Box>
                            )}
                        </Box>
                    )}
                </AnimatePresence>
            </Box>

            {totalDays > 1 && (
                <Stack direction="row" spacing={0.75} justifyContent="center" sx={{mt: 1.5}}>
                    {Array.from({length: totalDays}).map((_, index) => {
                        const isActive = index === dayOffset;
                        return (
                            <Box
                                key={`day-dot-${index}`}
                                sx={{
                                    width: isActive ? 16 : 6,
                                    height: 6,
                                    borderRadius: 999,
                                    bgcolor: isActive ? "text.primary" : "divider",
                                    transition: "all 180ms ease",
                                }}
                            />
                        );
                    })}
                </Stack>
            )}
        </ModernCard>
    );
}
