import {useEffect, useMemo, useState} from "react";
import {Alert, Box, Button, CircularProgress, IconButton, Stack, Typography} from "@mui/material";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import ModernCard from "../shared/components/ModernCard";
import type {ForecastDay} from "../lib/types/forecast";
import {formatChicagoTime, getChicagoTimestampMs} from "../shared/utils/chicagoTime";

type CrowdBandLevel = "low" | "medium" | "peak";

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
    dayOffset: number;
    canPrev: boolean;
    canNext: boolean;
    onPrev: () => void;
    onNext: () => void;
    isLoading: boolean;
    error: string | null;
}

const BAND_LEVEL_ORDER: CrowdBandLevel[] = ["low", "medium", "peak"];

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
    low: {label: "LOW CROWD", color: "#1b5e20", bg: "rgba(46, 125, 50, 0.12)"},
    medium: {label: "MEDIUM CROWD", color: "#8a6d00", bg: "rgba(245, 158, 11, 0.18)"},
    peak: {label: "PEAK CROWD", color: "#b71c1c", bg: "rgba(211, 47, 47, 0.12)"},
};

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

const renderBands = (bands: CrowdBand[], nowTs: number) => {
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
                            p: 1,
                            borderRadius: 1.5,
                            border: "1px solid",
                            borderColor: isCurrent ? "text.primary" : "divider",
                            boxShadow: isCurrent ? "0 0 0 2px rgba(0, 0, 0, 0.08)" : "none",
                            bgcolor: "background.default",
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
                                py: 0.25,
                                borderRadius: 999,
                                bgcolor: style.bg,
                                color: style.color,
                            }}
                        >
                            <Typography
                                variant="caption"
                                sx={{fontWeight: 800, letterSpacing: 0.3, textTransform: "uppercase"}}
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

export default function ForecastWindowsCard({
    day,
    dayOffset,
    canPrev,
    canNext,
    onPrev,
    onNext,
    isLoading,
    error,
}: Props) {
    const [selectedLevels, setSelectedLevels] = useState<CrowdBandLevel[]>([]);
    const [nowTs, setNowTs] = useState<number>(() => Date.now());

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

    const displayBands = useMemo(() => {
        const allBands = day?.crowdBands ?? [];
        const showDefault =
            selectedLevels.length === 0 || selectedLevels.length === BAND_LEVEL_ORDER.length;

        if (showDefault) {
            return sortBands(allBands);
        }

        const selected = new Set(selectedLevels);
        return sortBands(allBands).filter((band) => selected.has(band.level));
    }, [day?.crowdBands, selectedLevels]);

    const dateLabel = day?.date ? ` (${day.date})` : "";
    const title =
        dayOffset === 0
            ? `Forecast Today${dateLabel}`
            : dayOffset === 1
              ? `Forecast Tomorrow${dateLabel}`
              : dayOffset === 2
                ? `Forecast In 2 Days${dateLabel}`
                : `Forecast In 3 Days${dateLabel}`;

    return (
        <ModernCard>
            <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
                <Typography variant="subtitle2" color="text.secondary">
                    {title}
                </Typography>
                <Stack direction="row" spacing={0.25}>
                    <IconButton
                        size="small"
                        onClick={onPrev}
                        disabled={!canPrev || isLoading}
                        aria-label="Previous forecast day"
                    >
                        <ChevronLeftIcon fontSize="small"/>
                    </IconButton>
                    <IconButton
                        size="small"
                        onClick={onNext}
                        disabled={!canNext || isLoading}
                        aria-label="Next forecast day"
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

            {!isLoading && !error && day && (
                <Box sx={{mt: 1}}>
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
                                            variant={active ? "contained" : "outlined"}
                                            onClick={() => toggleLevel(level)}
                                            sx={{
                                                textTransform: "uppercase",
                                                fontWeight: 700,
                                                borderRadius: 999,
                                                px: 1.25,
                                                minWidth: 0,
                                                borderColor: style.color,
                                                color: active ? style.color : "text.primary",
                                                bgcolor: active ? style.bg : "transparent",
                                                "@media (hover: hover) and (pointer: fine)": {
                                                    "&:hover": {
                                                        borderColor: style.color,
                                                        bgcolor: style.bg,
                                                    },
                                                },
                                            }}
                                        >
                                            {style.label.replace(" CROWD", "")}
                                        </Button>
                                    );
                                })}
                            </Stack>
                            {renderBands(displayBands, nowTs)}
                        </>
                    ) : (
                        <Stack spacing={0.5}>
                            {(day.bestWindows ?? []).map((window, index) => (
                                <Typography key={`low-${index}`} variant="body2" sx={{fontWeight: 600}}>
                                    {formatWindow(window)} (LOW CROWD)
                                </Typography>
                            ))}
                            {(day.avoidWindows ?? []).map((window, index) => (
                                <Typography key={`peak-${index}`} variant="body2" sx={{fontWeight: 600}}>
                                    {formatWindow(window)} (PEAK CROWD)
                                </Typography>
                            ))}
                        </Stack>
                    )}
                </Box>
            )}
        </ModernCard>
    );
}
