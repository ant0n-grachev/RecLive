import {getOccupancyTone, type OccupancyThresholds} from "../../shared/utils/styles";
import {occupancyToneToBandLevel, type HistogramBandLevel} from "./forecastBands";
import {
    FORECAST_DISPLAY_SLOT_MINUTES,
    formatHistogramRangeLabel,
    formatMinuteLabel,
} from "./forecastTime";

const HISTOGRAM_MARGIN_LEFT = 24;
const HISTOGRAM_MARGIN_RIGHT = 2;
const HISTOGRAM_MARGIN_TOP = 30;
const HISTOGRAM_MARGIN_BOTTOM = 70;
const HISTOGRAM_PLOT_HEIGHT = 146;
const HISTOGRAM_SLOT_WIDTH = 22;
const HISTOGRAM_TICK_TARGET = 5;
export const HISTOGRAM_HOUR_LABEL_FONT_SIZE = 9;
const HISTOGRAM_TREND_MIN_NEIGHBOR_TOLERANCE = 18;
const HISTOGRAM_TREND_NEIGHBOR_TOLERANCE_RATIO = 0.18;
const HISTOGRAM_TREND_MIN_DEVIATION = 32;
const HISTOGRAM_TREND_DEVIATION_RATIO = 0.3;

export interface ForecastDisplaySlot {
    startMinute: number;
    endMinute: number;
    startTs: number;
    endTs: number;
    count: number;
    percent: number | null;
    source: "actual" | "predicted" | "mixed";
    level: HistogramBandLevel;
}

export interface HistogramBar {
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

export interface HistogramTick {
    value: number;
    y: number;
}

export interface HistogramModel {
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

export const buildHistogramModel = (
    slots: readonly ForecastDisplaySlot[],
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
    const inferredMaxCapacity = inferDisplayMaxCapacity([...slots]);
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
