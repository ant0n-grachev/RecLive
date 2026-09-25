import type {HistogramBandLevel} from "./forecastBands";
import {FORECAST_DISPLAY_SLOT_MINUTES, formatHistogramRangeLabel, formatMinuteLabel} from "./forecastTime";

const HISTOGRAM_MARGIN_LEFT = 24;
const HISTOGRAM_MARGIN_RIGHT = 2;
const HISTOGRAM_MARGIN_TOP = 30;
const HISTOGRAM_MARGIN_BOTTOM = 70;
const HISTOGRAM_PLOT_HEIGHT = 146;
const HISTOGRAM_SLOT_WIDTH = 22;
const HISTOGRAM_TICK_TARGET = 5;
export const HISTOGRAM_HOUR_LABEL_FONT_SIZE = 9;

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

export interface HistogramBar extends ForecastDisplaySlot {
    axisLabel: string;
    rangeLabel: string;
    showAxisLabel: boolean;
    x: number;
    centerX: number;
    y: number;
    width: number;
    height: number;
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

export const buildHistogramModel = (
    slots: readonly ForecastDisplaySlot[],
    scaleMaxCountOverride: number | null = null
): HistogramModel | null => {
    if (slots.length === 0) return null;

    const sortedSlots = [...slots].sort((a, b) => a.startTs - b.startTs);
    const maxCount = sortedSlots.reduce((max, slot) => Math.max(max, slot.count), 0);
    const scaleMaxCount = Math.max(maxCount, scaleMaxCountOverride ?? 0);
    const yTickStep = getNiceTickStep(scaleMaxCount);
    const yMax = Math.max(yTickStep, Math.ceil(scaleMaxCount / yTickStep) * yTickStep);
    const yTicks: HistogramTick[] = [];
    for (let value = 0; value <= yMax; value += yTickStep) {
        const ratio = value / yMax;
        const y = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT - (ratio * HISTOGRAM_PLOT_HEIGHT);
        yTicks.push({value, y});
    }

    const firstStartTs = sortedSlots[0].startTs;
    const slotDurationMs = FORECAST_DISPLAY_SLOT_MINUTES * 60 * 1000;
    const slotSpan = (sortedSlots[sortedSlots.length - 1].endTs - firstStartTs) / slotDurationMs;
    const plotWidth = Math.max(220, slotSpan * HISTOGRAM_SLOT_WIDTH);
    const viewBoxWidth = HISTOGRAM_MARGIN_LEFT + plotWidth + HISTOGRAM_MARGIN_RIGHT;
    const viewBoxHeight = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT + HISTOGRAM_MARGIN_BOTTOM;
    const baselineY = HISTOGRAM_MARGIN_TOP + HISTOGRAM_PLOT_HEIGHT;
    const labelStepMinutes = slotSpan > 12 ? 120 : 60;
    const bars: HistogramBar[] = sortedSlots.map((slot, index) => {
        const height = (slot.count / yMax) * HISTOGRAM_PLOT_HEIGHT;
        const x = HISTOGRAM_MARGIN_LEFT + ((slot.startTs - firstStartTs) / slotDurationMs) * HISTOGRAM_SLOT_WIDTH;
        const width = ((slot.endTs - slot.startTs) / slotDurationMs) * HISTOGRAM_SLOT_WIDTH;
        return {
            ...slot,
            axisLabel: formatMinuteLabel(slot.startMinute, slot.startMinute % 60 !== 0),
            rangeLabel: formatHistogramRangeLabel(slot.startMinute, slot.endMinute),
            showAxisLabel: index === 0 || slot.startMinute % labelStepMinutes === 0,
            x,
            centerX: x + (width / 2),
            y: baselineY - height,
            width,
            height,
        };
    });

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
        rotateHourLabels: bars.filter((bar) => bar.showAxisLabel).length > 8,
        actualBarCount: bars.filter((bar) => bar.source === "actual").length,
        predictedBarCount: bars.filter((bar) => bar.source === "predicted").length,
        mixedBarCount: bars.filter((bar) => bar.source === "mixed").length,
        unknownBarCount: bars.filter((bar) => bar.level === "unknown").length,
    };
};
