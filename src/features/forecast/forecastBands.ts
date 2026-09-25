import type {ForecastBand, ForecastDay} from "../../lib/types/forecast";
import {getChicagoTimestampMs} from "../../shared/utils/chicagoTime";
import {
    getOccupancyTone,
    OCCUPANCY_MAIN_HEX,
    OCCUPANCY_SOFT_BG,
} from "../../shared/utils/styles";
import type {ForecastDisplaySlot} from "./forecastHistogram";

export type CrowdBandLevel = ForecastBand["level"];
export type HistogramBandLevel = CrowdBandLevel | "unknown";
export type CrowdBand = ForecastBand;
export type ForecastDayWithBands = ForecastDay;

export const BAND_LEVEL_ORDER: CrowdBandLevel[] = ["low", "medium", "peak"];
export const BAND_STYLES: Record<CrowdBandLevel, {label: string; color: string; bg: string}> = {
    low: {label: "LOW CROWD", color: OCCUPANCY_MAIN_HEX.success, bg: OCCUPANCY_SOFT_BG.success},
    medium: {label: "MEDIUM CROWD", color: OCCUPANCY_MAIN_HEX.warning, bg: OCCUPANCY_SOFT_BG.warning},
    peak: {label: "PEAK CROWD", color: OCCUPANCY_MAIN_HEX.error, bg: OCCUPANCY_SOFT_BG.error},
};
// Caption foregrounds must remain readable on the matching translucent fill.
// Bar/occupancy identity colors stay unchanged.
export const getBandCaptionStyle = (level: CrowdBandLevel, isDark: boolean) => ({
    ...BAND_STYLES[level],
    color: isDark
        ? {low: "#86efac", medium: "#fde047", peak: "#fca5a5"}[level]
        : {low: "#166534", medium: "#854d0e", peak: "#991b1b"}[level],
});
export const UNKNOWN_BAND_STYLE = {
    label: "BAND UNAVAILABLE",
    color: "#64748b",
    bg: "rgba(100, 116, 139, 0.16)",
} as const;

const CROWD_BAND_CONTIGUITY_TOLERANCE_MS = 60 * 1000;
export const EMPTY_CROWD_BANDS: CrowdBand[] = [];

export const getHistogramBandStyle = (level: HistogramBandLevel) => (
    level === "unknown" ? UNKNOWN_BAND_STYLE : BAND_STYLES[level]
);

export const isHistogramLevelVisible = (
    level: HistogramBandLevel,
    selected: Set<CrowdBandLevel>,
    showDefault: boolean
): boolean => {
    if (showDefault) return true;
    if (level === "unknown") return false;
    return selected.has(level);
};

export const sortBands = (bands: CrowdBand[]): CrowdBand[] =>
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

export const occupancyToneToBandLevel = (
    tone: ReturnType<typeof getOccupancyTone>
): CrowdBandLevel | null => {
    if (tone === "success") return "low";
    if (tone === "warning") return "medium";
    if (tone === "error") return "peak";
    return null;
};

export const buildBandTimeRanges = (
    bands: CrowdBand[]
): Array<{startTs: number; endTs: number; level: CrowdBandLevel}> =>
    bands
        .map((band) => {
            const startTs = getChicagoTimestampMs(band.start);
            const endTs = getChicagoTimestampMs(band.end);
            if (startTs === null || endTs === null || endTs <= startTs) return null;
            return {startTs, endTs, level: band.level};
        })
        .filter((range): range is {startTs: number; endTs: number; level: CrowdBandLevel} => Boolean(range))
        .sort((a, b) => a.startTs - b.startTs);

export const getLevelAtTimestamp = (
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

export const buildCrowdBandsFromDisplaySlots = (slots: ForecastDisplaySlot[]): CrowdBand[] => {
    const bands = slots
        .filter((slot): slot is ForecastDisplaySlot & {level: CrowdBandLevel} => slot.level !== "unknown")
        .map((slot) => ({
            start: new Date(slot.startTs).toISOString(),
            end: new Date(slot.endTs).toISOString(),
            level: slot.level,
        }));

    return mergeAdjacentCrowdBands(bands);
};

export const isCurrentBand = (band: CrowdBand, nowTs: number): boolean => {
    const startTs = getChicagoTimestampMs(band.start);
    const endTs = getChicagoTimestampMs(band.end);
    if (startTs === null || endTs === null) return false;
    return nowTs >= startTs && nowTs < endTs;
};
