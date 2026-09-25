import {Box, Stack, Typography} from "@mui/material";
import type {ForecastWindow} from "../../lib/types/forecast";
import {INNER_SURFACE_SX} from "../../shared/utils/styles";
import {
    getBandCaptionStyle,
    isCurrentBand,
    sortBands,
    type CrowdBand,
} from "./forecastBands";
import {formatRange, formatWindow} from "./forecastTime";

interface Props {
    workingHoursBands: CrowdBand[];
    displayBands: CrowdBand[];
    filteredBestWindows: ForecastWindow[];
    filteredAvoidWindows: ForecastWindow[];
    nowTs: number;
    isDark: boolean;
}

function ForecastBands({bands, nowTs, isDark}: {
    bands: CrowdBand[];
    nowTs: number;
    isDark: boolean;
}) {
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
                const style = getBandCaptionStyle(band.level, isDark);
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
}

export default function ForecastWindowsList({
    workingHoursBands,
    displayBands,
    filteredBestWindows,
    filteredAvoidWindows,
    nowTs,
    isDark,
}: Props) {
    if (workingHoursBands.length > 0) {
        return <ForecastBands bands={displayBands} nowTs={nowTs} isDark={isDark}/>;
    }

    if (filteredBestWindows.length === 0 && filteredAvoidWindows.length === 0) {
        return null;
    }

    return (
        <Stack spacing={0.5}>
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
        </Stack>
    );
}
