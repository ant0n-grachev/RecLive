import {Box} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import {
    getHistogramBandStyle,
    getVisibleHistogramSegments,
    type CrowdBandLevel,
} from "./forecastBands";
import {
    HISTOGRAM_HOUR_LABEL_FONT_SIZE,
    type HistogramModel,
} from "./forecastHistogram";

interface Props {
    histogram: HistogramModel;
    selectedLevelSet: Set<CrowdBandLevel>;
    showAllHistogramLevels: boolean;
    selectedStartMinute: number | null;
    onToggleBar: (startMinute: number) => void;
    currentTimeMarker: {x: number; label: string} | null;
}

export default function ForecastChart({
    histogram,
    selectedLevelSet,
    showAllHistogramLevels,
    selectedStartMinute,
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
                const isSelected = selectedStartMinute === bar.startMinute;
                const axisLabelY = histogram.baselineY + 22;
                const halfWidth = bar.width / 2;
                const outlineX = leftVisible ? bar.x : bar.x + halfWidth;
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
                            onToggleBar(bar.startMinute);
                        }}
                        onKeyDown={(event) => {
                            if (!hasVisibleSegments) return;
                            if (event.key !== "Enter" && event.key !== " ") return;
                            event.preventDefault();
                            onToggleBar(bar.startMinute);
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
    );
}
