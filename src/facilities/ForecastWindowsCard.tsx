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
    INNER_SURFACE_SX,
    OCCUPANCY_MAIN_HEX,
    OCCUPANCY_SOFT_BG,
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
    schedule?: FacilityHoursFacilityPayload | null;
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

const HISTOGRAM_MARGIN_LEFT = 2;
const HISTOGRAM_MARGIN_RIGHT = 2;
const HISTOGRAM_MARGIN_TOP = 30;
const HISTOGRAM_MARGIN_BOTTOM = 70;
const HISTOGRAM_PLOT_HEIGHT = 138;
const HISTOGRAM_SLOT_WIDTH = 22;
const HISTOGRAM_BAND_SEGMENT_MINUTES = 30;
const HISTOGRAM_TICK_TARGET = 5;
const HISTOGRAM_VALUE_LABEL_FONT_SIZE = 9;
const HISTOGRAM_HOUR_LABEL_FONT_SIZE = 9;
const HISTOGRAM_MIN_ROTATED_INSIDE_LABEL_HEIGHT = 28;
const HISTOGRAM_INSIDE_LABEL_TOP_OFFSET = 13.2;
const MINUTES_PER_DAY = 24 * 60;

interface HistogramBar {
    hour: number;
    hourLabel: string;
    hourRangeLabel: string;
    x: number;
    centerX: number;
    y: number;
    width: number;
    height: number;
    count: number;
    source: "actual" | "predicted";
    level: HistogramBandLevel;
    segmentLevels: [HistogramBandLevel, HistogramBandLevel];
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
    rotateValueLabels: boolean;
    actualBarCount: number;
    predictedBarCount: number;
    unknownBarCount: number;
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
                                px: 1,
                                minHeight: 22,
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

const formatHistogramAxisLabel = (minuteOfDay: number): string =>
    formatMinuteLabel(minuteOfDay, minuteOfDay % 60 !== 0);

const formatHistogramRangeLabel = (startMinute: number): string => {
    const endMinute = startMinute + 60;
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

const isMinuteWithinOpenWindows = (minute: number, windows: FacilityOpenWindow[]): boolean =>
    windows.some((window) => minute >= window.startMinutes && minute < window.endMinutes);

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

const buildWorkingHourSlots = (windows: FacilityOpenWindow[]): number[] => {
    const hours: number[] = [];
    for (let hour = 0; hour < 24; hour += 1) {
        const minute = hour * 60;
        if (isMinuteWithinOpenWindows(minute, windows)) {
            hours.push(hour);
        }
    }
    return hours;
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

const buildHistogramModel = (
    day: ForecastDayWithBands | null,
    openWindows: FacilityOpenWindow[],
    enforceWorkingHours: boolean,
    bands: CrowdBand[],
    nowTs: number
): HistogramModel | null => {
    if (!day?.categories || day.categories.length === 0) return null;
    if (enforceWorkingHours && openWindows.length === 0) return null;

    const byTimestamp = new Map<number, {
        ts: number;
        minuteOfDay: number;
        predictedCount: number;
        actualCount: number;
        actualSources: number;
    }>();
    for (const category of day.categories) {
        for (const hour of category.hours ?? []) {
            const timestampMs = getChicagoTimestampMs(hour.hourStart);
            if (timestampMs === null) continue;

            const minuteOfDay = getChicagoMinuteOfDay(hour.hourStart);
            if (minuteOfDay === null) continue;
            if (enforceWorkingHours && !isMinuteWithinOpenWindows(minuteOfDay, openWindows)) {
                continue;
            }

            const expectedCount = typeof hour.expectedCount === "number" ? hour.expectedCount : 0;
            const actualCount = typeof hour.actualCount === "number" ? hour.actualCount : null;
            const current = byTimestamp.get(timestampMs);
            byTimestamp.set(timestampMs, {
                ts: timestampMs,
                minuteOfDay,
                predictedCount: (current?.predictedCount ?? 0) + Math.max(0, expectedCount),
                actualCount: (current?.actualCount ?? 0) + (actualCount === null ? 0 : Math.max(0, actualCount)),
                actualSources: (current?.actualSources ?? 0) + (actualCount === null ? 0 : 1),
            });
        }
    }

    const points = [...byTimestamp.values()].sort((a, b) => a.ts - b.ts);
    if (points.length === 0) return null;

    const byHour = new Map<number, {
        predictedSum: number;
        predictedSamples: number;
        actualSum: number;
        actualSamples: number;
        firstTs: number;
        firstSegmentTs: number | null;
        secondSegmentTs: number | null;
    }>();
    for (const point of points) {
        const hourOfDay = Math.floor(point.minuteOfDay / 60);
        const minuteInHour = point.minuteOfDay % 60;
        const hasObservedActual = point.ts <= nowTs && point.actualSources > 0;
        const current = byHour.get(hourOfDay);
        byHour.set(hourOfDay, {
            predictedSum: (current?.predictedSum ?? 0) + point.predictedCount,
            predictedSamples: (current?.predictedSamples ?? 0) + 1,
            actualSum: (current?.actualSum ?? 0) + (hasObservedActual ? point.actualCount : 0),
            actualSamples: (current?.actualSamples ?? 0) + (hasObservedActual ? 1 : 0),
            firstTs: current?.firstTs ?? point.ts,
            firstSegmentTs: minuteInHour < HISTOGRAM_BAND_SEGMENT_MINUTES
                ? (current?.firstSegmentTs ?? point.ts)
                : (current?.firstSegmentTs ?? null),
            secondSegmentTs: minuteInHour >= HISTOGRAM_BAND_SEGMENT_MINUTES
                ? (current?.secondSegmentTs ?? point.ts)
                : (current?.secondSegmentTs ?? null),
        });
    }

    const hours = enforceWorkingHours
        ? buildWorkingHourSlots(openWindows)
        : [...new Set(points.map((point) => Math.floor(point.minuteOfDay / 60)))].sort((a, b) => a - b);
    if (hours.length === 0) return null;

    const hourSeries = hours.map((hour) => {
        const bucket = byHour.get(hour);
        const predictedAverage = bucket ? bucket.predictedSum / Math.max(1, bucket.predictedSamples) : 0;
        const hasObservedActual = Boolean(bucket && bucket.actualSamples > 0);
        const actualAverage = hasObservedActual && bucket
            ? bucket.actualSum / Math.max(1, bucket.actualSamples)
            : null;
        const source: HistogramBar["source"] = actualAverage === null ? "predicted" : "actual";
        const count = actualAverage ?? predictedAverage;
        return {
            hour,
            count: Math.max(0, Math.round(count)),
            source,
            representativeTs: bucket?.firstTs ?? null,
            firstSegmentTs: bucket?.firstSegmentTs ?? null,
            secondSegmentTs: bucket?.secondSegmentTs ?? null,
        };
    });

    const maxCount = hourSeries.reduce((max, item) => Math.max(max, item.count), 0);
    const yTickStep = getNiceTickStep(maxCount);
    const yMax = Math.max(yTickStep, Math.ceil(maxCount / yTickStep) * yTickStep);
    const yTicks: HistogramTick[] = [];
    for (let value = 0; value <= yMax; value += yTickStep) {
        const ratio = yMax > 0 ? value / yMax : 0;
        const y = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT - (ratio * HISTOGRAM_PLOT_HEIGHT);
        yTicks.push({value, y});
    }

    const plotWidth = Math.max(220, hourSeries.length * HISTOGRAM_SLOT_WIDTH);
    const viewBoxWidth = HISTOGRAM_MARGIN_LEFT + plotWidth + HISTOGRAM_MARGIN_RIGHT;
    const viewBoxHeight = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT + HISTOGRAM_MARGIN_BOTTOM;
    const baselineY = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT;
    const barWidth = HISTOGRAM_SLOT_WIDTH;
    const bandRanges = buildBandTimeRanges(bands);

    const bars: HistogramBar[] = hourSeries.map((point, index) => {
        const height = yMax > 0 ? (point.count / yMax) * HISTOGRAM_PLOT_HEIGHT : 0;
        const x = HISTOGRAM_MARGIN_LEFT + (index * HISTOGRAM_SLOT_WIDTH);
        const centerX = x + (barWidth / 2);
        const bandLevel = point.representativeTs === null
            ? null
            : getLevelAtTimestamp(point.representativeTs, bandRanges);
        const firstSegmentLevel = point.firstSegmentTs === null
            ? null
            : getLevelAtTimestamp(point.firstSegmentTs, bandRanges);
        const secondSegmentLevel = point.secondSegmentTs === null
            ? null
            : getLevelAtTimestamp(point.secondSegmentTs, bandRanges);
        const segmentLevels: [HistogramBandLevel, HistogramBandLevel] = [
            firstSegmentLevel ?? secondSegmentLevel ?? "unknown",
            secondSegmentLevel ?? firstSegmentLevel ?? "unknown",
        ];
        const resolvedLevel: HistogramBandLevel = bandLevel ?? segmentLevels[0] ?? segmentLevels[1] ?? "unknown";
        return {
            hour: point.hour,
            hourLabel: formatHistogramAxisLabel(point.hour * 60),
            hourRangeLabel: formatHistogramRangeLabel(point.hour * 60),
            x,
            centerX,
            y: baselineY - height,
            width: barWidth,
            height,
            count: point.count,
            source: point.source,
            level: resolvedLevel,
            segmentLevels,
        };
    });
    const actualBarCount = bars.filter((bar) => bar.source === "actual").length;
    const unknownBarCount = bars.filter((bar) => bar.level === "unknown").length;

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
        rotateHourLabels: hourSeries.length >= 10,
        rotateValueLabels: hourSeries.length >= 10,
        actualBarCount,
        predictedBarCount: bars.length - actualBarCount,
        unknownBarCount,
    };
};

export default function ForecastWindowsCard({
    day,
    schedule = null,
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
    const axisBaselineColor = alpha(theme.palette.text.primary, isDark ? 0.45 : 0.25);
    const reduceMotion = useReducedMotion();
    const [selectedLevels, setSelectedLevels] = useState<CrowdBandLevel[]>([]);
    const [nowTs, setNowTs] = useState<number>(() => Date.now());
    const [slideDirection, setSlideDirection] = useState<1 | -1>(1);
    const [trendExpanded, setTrendExpanded] = useState(false);
    const [selectedHistogramSelection, setSelectedHistogramSelection] = useState<{
        date: string | null;
        hour: number | null;
    }>({date: null, hour: null});
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

    const workingHoursBands = useMemo(() => {
        const allBands = day?.crowdBands ?? [];
        return sortBands(
            clipBandsToWorkingHours(allBands, openWindows, hasWorkingHours, day?.date ?? null)
        );
    }, [day?.crowdBands, day?.date, hasWorkingHours, openWindows]);

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

    const histogram = useMemo(
        () => buildHistogramModel(day, openWindows, hasWorkingHours, workingHoursBands, nowTs),
        [day, openWindows, hasWorkingHours, workingHoursBands, nowTs]
    );

    const filteredBestWindows = useMemo(
        () => filterWindowsToWorkingHours(day?.bestWindows ?? [], openWindows, hasWorkingHours, day?.date ?? null),
        [day?.bestWindows, day?.date, openWindows, hasWorkingHours]
    );

    const filteredAvoidWindows = useMemo(
        () => filterWindowsToWorkingHours(day?.avoidWindows ?? [], openWindows, hasWorkingHours, day?.date ?? null),
        [day?.avoidWindows, day?.date, openWindows, hasWorkingHours]
    );
    const selectedHistogramHour = (
        selectedHistogramSelection.date === (day?.date ?? null)
            ? selectedHistogramSelection.hour
            : null
    );
    const selectedHistogramBar = useMemo(
        () => histogram?.bars.find((bar) => bar.hour === selectedHistogramHour) ?? null,
        [histogram, selectedHistogramHour]
    );
    const currentTimeMarker = useMemo(() => {
        if (!histogram || !day?.date) return null;

        const todayDateKey = getChicagoDateKeyFromTimestamp(nowTs);
        if (!todayDateKey || day.date !== todayDateKey) return null;

        const nowMinuteOfDay = getChicagoMinuteOfDayFromTimestamp(nowTs);
        if (nowMinuteOfDay === null) return null;

        const nowHour = Math.floor(nowMinuteOfDay / 60);
        const minuteInHour = nowMinuteOfDay % 60;
        const activeBar = histogram.bars.find((bar) => bar.hour === nowHour);
        if (!activeBar) return null;

        return {
            x: activeBar.x + ((minuteInHour / 60) * activeBar.width),
            label: "Now",
        };
    }, [day, histogram, nowTs]);

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

    const toggleHistogramHourSelection = (hour: number) => {
        const currentDate = day?.date ?? null;
        setSelectedHistogramSelection((prev) => {
            const previousHourForCurrentDay = prev.date === currentDate ? prev.hour : null;
            return {
                date: currentDate,
                hour: previousHourForCurrentDay === hour ? null : hour,
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
                    {!isLoading && !error && day && (
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
                            {day.crowdBands && day.crowdBands.length > 0 ? (
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
                                                        px: 1.05,
                                                        minWidth: 0,
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
                                        {trendExpanded ? "Hide hourly graph" : "Show hourly graph"}
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
                                                    aria-label="People histogram by working hour"
                                                    sx={{
                                                        display: "block",
                                                        width: "100%",
                                                        height: {xs: 230, sm: 250},
                                                    }}
                                                >
                                                    {histogram.bars.map((bar, index) => {
                                                        const style = getHistogramBandStyle(bar.level);
                                                        const leftStyle = getHistogramBandStyle(bar.segmentLevels[0]);
                                                        const rightStyle = getHistogramBandStyle(bar.segmentLevels[1]);
                                                        const roundedCount = Math.max(0, Math.round(bar.count));
                                                        const isSelected = selectedHistogramHour === bar.hour;
                                                        const showInside = histogram.rotateValueLabels
                                                            ? bar.height >= HISTOGRAM_MIN_ROTATED_INSIDE_LABEL_HEIGHT
                                                            : bar.height >= 34;
                                                        const shouldRotateValueLabel = histogram.rotateValueLabels && showInside;
                                                        const valueLabelY = bar.y + HISTOGRAM_INSIDE_LABEL_TOP_OFFSET;
                                                        const hourLabelY = histogram.baselineY + 22;
                                                        const halfWidth = bar.width / 2;
                                                        const isSplitColor = bar.segmentLevels[0] !== bar.segmentLevels[1];

                                                        return (
                                                            <g
                                                                key={`bar-${bar.hour}-${index}`}
                                                                role="button"
                                                                tabIndex={0}
                                                                onClick={() => toggleHistogramHourSelection(bar.hour)}
                                                                onKeyDown={(event) => {
                                                                    if (event.key !== "Enter" && event.key !== " ") return;
                                                                    event.preventDefault();
                                                                    toggleHistogramHourSelection(bar.hour);
                                                                }}
                                                                aria-label={`${bar.hourRangeLabel}, ${roundedCount} people`}
                                                                style={{cursor: "pointer"}}
                                                            >
                                                                {isSplitColor ? (
                                                                    <>
                                                                        <rect
                                                                            x={bar.x}
                                                                            y={bar.y}
                                                                            width={halfWidth}
                                                                            height={bar.height}
                                                                            fill={leftStyle.color}
                                                                            stroke="none"
                                                                            rx={0}
                                                                        />
                                                                        <rect
                                                                            x={bar.x + halfWidth}
                                                                            y={bar.y}
                                                                            width={bar.width - halfWidth}
                                                                            height={bar.height}
                                                                            fill={rightStyle.color}
                                                                            stroke="none"
                                                                            rx={0}
                                                                        />
                                                                        <rect
                                                                            x={bar.x}
                                                                            y={bar.y}
                                                                            width={bar.width}
                                                                            height={bar.height}
                                                                            fill="none"
                                                                            stroke={isSelected ? selectedBarStroke : barSeparatorStroke}
                                                                            strokeWidth={isSelected ? 1.1 : 0.35}
                                                                            rx={0}
                                                                        />
                                                                    </>
                                                                ) : (
                                                                    <rect
                                                                        x={bar.x}
                                                                        y={bar.y}
                                                                        width={bar.width}
                                                                        height={bar.height}
                                                                        fill={style.color}
                                                                        stroke={isSelected ? selectedBarStroke : barSeparatorStroke}
                                                                        strokeWidth={isSelected ? 1.1 : 0.35}
                                                                        rx={0}
                                                                    />
                                                                )}
                                                                {showInside && (
                                                                    <text
                                                                        x={bar.centerX}
                                                                        y={valueLabelY}
                                                                        textAnchor="middle"
                                                                        dominantBaseline={shouldRotateValueLabel ? "middle" : "alphabetic"}
                                                                        fontSize={HISTOGRAM_VALUE_LABEL_FONT_SIZE}
                                                                        fontWeight={800}
                                                                        fill="#ffffff"
                                                                        transform={shouldRotateValueLabel
                                                                            ? `rotate(-90 ${bar.centerX} ${valueLabelY})`
                                                                            : undefined}
                                                                    >
                                                                        {roundedCount}
                                                                    </text>
                                                                )}
                                                                <line
                                                                    x1={bar.centerX}
                                                                    x2={bar.centerX}
                                                                    y1={histogram.baselineY}
                                                                    y2={histogram.baselineY + 2.3}
                                                                    stroke={axisTickColor}
                                                                    strokeWidth={0.55}
                                                                />
                                                                <text
                                                                    x={bar.centerX}
                                                                    y={histogram.rotateHourLabels ? hourLabelY + 7 : hourLabelY}
                                                                    textAnchor={histogram.rotateHourLabels ? "end" : "middle"}
                                                                    dominantBaseline="hanging"
                                                                    fontSize={HISTOGRAM_HOUR_LABEL_FONT_SIZE}
                                                                    fontWeight={700}
                                                                    fill={axisLabelColor}
                                                                    transform={histogram.rotateHourLabels
                                                                        ? `rotate(-52 ${bar.centerX} ${hourLabelY + 7})`
                                                                        : undefined}
                                                                >
                                                                    {bar.hourLabel}
                                                                </text>
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
                                                        Some hourly colors are unavailable because forecast bands were not emitted for those slots.
                                                    </Typography>
                                                )}
                                                <Typography variant="body2" color="text.primary" sx={{display: "block", mt: 0.25, fontWeight: 700}}>
                                                    {selectedHistogramBar
                                                        ? `${selectedHistogramBar.hourRangeLabel}: ${Math.round(selectedHistogramBar.count)} people`
                                                        : "Tap a bar to see exact hour and people"}
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
