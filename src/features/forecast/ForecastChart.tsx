import {Box} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import {
    getHistogramBandStyle,
    isHistogramLevelVisible,
    type CrowdBandLevel,
} from "./forecastBands";
import {
    FORECAST_SOURCE_LABELS,
    HISTOGRAM_HOUR_LABEL_FONT_SIZE,
    type HistogramModel,
} from "./forecastHistogram";

interface Props {
    histogram: HistogramModel;
    selectedLevelSet: Set<CrowdBandLevel>;
    showAllHistogramLevels: boolean;
    selectedStartTs: number | null;
    onToggleBar: (startTs: number) => void;
    currentTimeMarker: {x: number; label: string} | null;
}

export default function ForecastChart({
    histogram,
    selectedLevelSet,
    showAllHistogramLevels,
    selectedStartTs,
    onToggleBar,
    currentTimeMarker,
}: Props) {
    const theme = useTheme();
    const isDark = theme.palette.mode === "dark";
    const selectedBarStroke = alpha(theme.palette.text.primary, isDark ? 0.96 : 0.86);
    const barSeparatorStroke = alpha(theme.palette.text.primary, isDark ? 0.82 : 0.6);
    const nowMarkerStroke = alpha(theme.palette.text.primary, isDark ? 0.96 : 0.9);
    const nowMarkerText = alpha(theme.palette.text.primary, isDark ? 0.98 : 0.94);
    const axisTickColor = alpha(theme.palette.text.primary, isDark ? 0.5 : 0.3);
    const axisLabelColor = alpha(theme.palette.text.primary, isDark ? 0.88 : 0.75);
    const axisValueColor = alpha(theme.palette.text.primary, isDark ? 0.62 : 0.5);
    const axisBaselineColor = alpha(theme.palette.text.primary, isDark ? 0.45 : 0.25);
    const histogramGridLineColor = alpha(theme.palette.text.primary, isDark ? 0.12 : 0.08);

    return (
        <Box
            component="svg"
            viewBox={`0 0 ${histogram.viewBoxWidth} ${histogram.viewBoxHeight}`}
            role="img"
            aria-label="People by half hour"
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
                const isVisible = isHistogramLevelVisible(bar.level, selectedLevelSet, showAllHistogramLevels);
                const roundedCount = Math.max(0, Math.round(bar.count));
                const isSelected = selectedStartTs === bar.startTs;
                const axisLabelY = histogram.baselineY + 22;

                return (
                    <g
                        key={`bar-${bar.startTs}-${index}`}
                        role={isVisible ? "button" : undefined}
                        tabIndex={isVisible ? 0 : -1}
                        onClick={() => {
                            if (!isVisible) return;
                            onToggleBar(bar.startTs);
                        }}
                        onKeyDown={(event) => {
                            if (!isVisible) return;
                            if (event.key !== "Enter" && event.key !== " ") return;
                            event.preventDefault();
                            onToggleBar(bar.startTs);
                        }}
                        aria-label={`${bar.rangeLabel}, ${FORECAST_SOURCE_LABELS[bar.source]}, ${roundedCount} people`}
                        aria-hidden={!isVisible}
                        style={{cursor: isVisible ? "pointer" : "default"}}
                    >
                        {isVisible && (
                            <rect
                                x={bar.x}
                                y={bar.y}
                                width={bar.width}
                                height={bar.height}
                                fill={style.color}
                                stroke="none"
                                rx={0}
                            />
                        )}
                        {isVisible && (
                            <rect
                                x={bar.x}
                                y={bar.y}
                                width={bar.width}
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
    );
}
