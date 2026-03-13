import {useEffect, useRef, useState} from "react";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import EastRoundedIcon from "@mui/icons-material/EastRounded";
import SouthRoundedIcon from "@mui/icons-material/SouthRounded";
import NorthRoundedIcon from "@mui/icons-material/NorthRounded";
import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Box,
    Chip,
    Stack,
    Typography,
} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import {useReducedMotion} from "framer-motion";
import type {Location} from "../lib/types/facility";
import type {ForecastHour} from "../lib/types/forecast";
import {
    CARD_SHELL_SX,
    CARD_TITLE_SX,
    clampPercent,
    getOccupancyColor,
    INNER_SURFACE_SX,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {getSectionVisual} from "./sectionIcons";

interface Props {
    title: string;
    ids: number[];
    locations: Location[];
    forecast?: ForecastHour[];
    occupancyThresholds?: OccupancyThresholds | null;
    locationOccupancyThresholds?: Partial<Record<number, OccupancyThresholds>>;
}

export default function SectionCommandCenter({
    title,
    ids,
    locations,
    forecast,
    occupancyThresholds = null,
    locationOccupancyThresholds = {},
}: Props) {
    const theme = useTheme();
    const metricColumnSx = {textAlign: "right", flexShrink: 0, width: {xs: 118, sm: 126}} as const;
    const metricRailSx = {display: "flex", alignItems: "center", justifyContent: "flex-end", gap: 0.35, flexShrink: 0} as const;
    const metricStackSx = {
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "flex-end",
        minHeight: 44,
    } as const;
    const reduceMotion = useReducedMotion();
    const list = locations
        .filter((l) => ids.includes(l.locationId))
        .sort((a, b) => ids.indexOf(a.locationId) - ids.indexOf(b.locationId));
    const isSingleLocation = list.length <= 1;

    const total = list.reduce((sum, loc) => sum + (loc.currentCapacity ?? 0), 0);
    const max = list.reduce((sum, loc) => sum + (loc.maxCapacity ?? 0), 0);
    const [animatedTotal, setAnimatedTotal] = useState(0);
    const [animatedPercent, setAnimatedPercent] = useState(0);
    const animatedTotalRef = useRef(0);
    const animatedPercentRef = useRef(0);
    const targetPercent = clampPercent(max ? (total / max) * 100 : 0);

    useEffect(() => {
        if (reduceMotion) return;

        const fromTotal = animatedTotalRef.current;
        const fromPercent = animatedPercentRef.current;
        const toTotal = total;
        const toPercent = targetPercent;
        if (fromTotal === toTotal && fromPercent === toPercent) return;

        let frameId = 0;
        let startAt = 0;
        const durationMs = 920;

        const tick = (timestamp: number) => {
            if (startAt === 0) startAt = timestamp;
            const progress = Math.min(1, (timestamp - startAt) / durationMs);
            const eased = 1 - Math.pow(1 - progress, 3);
            const nextTotal = fromTotal + (toTotal - fromTotal) * eased;
            const nextPercent = fromPercent + (toPercent - fromPercent) * eased;

            animatedTotalRef.current = nextTotal;
            animatedPercentRef.current = nextPercent;
            setAnimatedTotal(nextTotal);
            setAnimatedPercent(nextPercent);

            if (progress < 1) {
                frameId = window.requestAnimationFrame(tick);
            } else {
                animatedTotalRef.current = toTotal;
                animatedPercentRef.current = toPercent;
                setAnimatedTotal(toTotal);
                setAnimatedPercent(toPercent);
            }
        };

        frameId = window.requestAnimationFrame(tick);
        return () => window.cancelAnimationFrame(frameId);
    }, [reduceMotion, targetPercent, total]);

    const displayTotal = reduceMotion ? Math.max(0, total) : Math.max(0, Math.round(animatedTotal));
    const displayPercent = reduceMotion ? targetPercent : clampPercent(animatedPercent);
    const closedCount = list.filter((loc) => loc.isClosed === true).length;
    const allClosed = list.length > 0 && closedCount === list.length;
    const titleVisual = getSectionVisual(title);
    const singleLoc = list[0] ?? null;
    const isSingleClosed = singleLoc?.isClosed === true;
    const singleLocationThresholds = singleLoc
        ? (locationOccupancyThresholds[singleLoc.locationId] ?? occupancyThresholds)
        : occupancyThresholds;
    const percentColor = getOccupancyColor(
        displayPercent,
        isSingleLocation ? singleLocationThresholds : occupancyThresholds
    );
    const forecastChipBorderColor = alpha(theme.palette.text.primary, theme.palette.mode === "dark" ? 0.28 : 0.2);
    const cardShellSx = {
        ...CARD_SHELL_SX,
        overflow: "hidden",
    };

    const forecastStrip = !!forecast?.length && (
        <Stack direction="row" spacing={0.75} sx={{mb: 1.25, flexWrap: "wrap", rowGap: 0.75}}>
            {forecast.slice(0, 3).map((point, index) => (
                <Chip
                    key={`${point.hourStart}-${index}`}
                    label={(
                        <Stack direction="row" spacing={0.6} alignItems="center">
                            <Typography
                                component="span"
                                sx={{fontSize: "0.72rem", fontWeight: 800, color: "text.secondary"}}
                            >
                                +{index + 1}h
                            </Typography>
                            <Stack direction="row" spacing={0.2} alignItems="center">
                                <Typography
                                    component="span"
                                    sx={{fontSize: "0.92rem", fontWeight: 800, color: "text.primary", lineHeight: 1}}
                                >
                                    {Math.max(0, Math.round(point.expectedCount))}
                                </Typography>
                                {(() => {
                                    const currentValue = Math.max(0, Math.round(point.expectedCount));
                                    const previousValue = index === 0
                                        ? Math.max(0, total)
                                        : Math.max(0, Math.round(forecast[index - 1]?.expectedCount ?? 0));
                                    const delta = currentValue - previousValue;

                                    if (delta > 0) {
                                        return <NorthRoundedIcon sx={{fontSize: 14, color: "error.main"}}/>;
                                    }

                                    if (delta < 0) {
                                        return <SouthRoundedIcon sx={{fontSize: 14, color: "success.main"}}/>;
                                    }

                                    return <EastRoundedIcon sx={{fontSize: 13, color: "text.disabled"}}/>;
                                })()}
                            </Stack>
                        </Stack>
                    )}
                    size="small"
                    variant="outlined"
                    sx={{
                        height: 32,
                        px: 0.2,
                        borderRadius: 999,
                        borderColor: forecastChipBorderColor,
                        bgcolor: "background.paper",
                        boxShadow: "none",
                        "& .MuiChip-label": {
                            px: 0.75,
                            py: 0,
                        },
                    }}
                />
            ))}
        </Stack>
    );

    const locationRows = (
        <Stack spacing={0.75}>
            {list.map((loc) => {
                const isClosed = loc.isClosed === true;
                const current = loc.currentCapacity ?? 0;
                const capacity = loc.maxCapacity ?? 0;
                const locPct = capacity ? clampPercent((current / capacity) * 100) : 0;
                const locationThresholds = locationOccupancyThresholds[loc.locationId] ?? occupancyThresholds;

                return (
                    <Box
                        key={loc.locationId}
                        sx={{
                            ...INNER_SURFACE_SX,
                            px: 1.25,
                            py: 0.9,
                            display: "flex",
                            justifyContent: "space-between",
                            alignItems: "center",
                            gap: 1,
                        }}
                    >
                        <Typography variant="body2" sx={{fontWeight: 600}}>
                            {loc.locationName}
                        </Typography>
                        {isClosed ? (
                            <Typography variant="body2" sx={{fontWeight: 700, color: "error.main"}}>
                                CLOSED
                            </Typography>
                        ) : (
                            <Typography variant="body2" sx={{fontWeight: 700, color: "text.primary", whiteSpace: "nowrap"}}>
                                {current} / {capacity}{" "}
                                <Box component="span" sx={{color: getOccupancyColor(locPct, locationThresholds)}}>
                                    ({locPct}%)
                                </Box>
                            </Typography>
                        )}
                    </Box>
                );
            })}
        </Stack>
    );

    if (isSingleLocation) {
        return (
            <Box
                sx={{
                    ...cardShellSx,
                    px: 2,
                    pt: 1.3,
                    pb: 1.7,
                }}
            >
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{width: "100%", gap: 1}}>
                    <Box sx={{minWidth: 0}}>
                        <Stack direction="row" spacing={0.75} alignItems="center">
                            {titleVisual && (
                                <Box
                                    sx={{
                                        width: 28,
                                        height: 28,
                                        borderRadius: 2,
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        color: titleVisual.color,
                                        bgcolor: titleVisual.bg,
                                        flexShrink: 0,
                                    }}
                                >
                                    {titleVisual.icon}
                                </Box>
                            )}
                            <Typography variant="h6" sx={CARD_TITLE_SX}>
                                {title}
                            </Typography>
                        </Stack>
                    </Box>
                    <Box sx={metricRailSx}>
                        {isSingleClosed ? (
                            <Box sx={{...metricColumnSx, ...metricStackSx}}>
                                <Typography
                                    variant="body1"
                                    sx={{fontWeight: 900, color: "error.main", letterSpacing: 0.4}}
                                >
                                    CLOSED
                                </Typography>
                            </Box>
                        ) : (
                            <Box sx={{...metricColumnSx, ...metricStackSx}}>
                                <Typography variant="body1" sx={{fontWeight: 800, fontVariantNumeric: "tabular-nums"}}>
                                    {displayTotal} / {max}
                                </Typography>
                                <Typography variant="body2" sx={{fontWeight: 700, color: percentColor, fontVariantNumeric: "tabular-nums"}}>
                                    {displayPercent}% full
                                </Typography>
                            </Box>
                        )}
                        <Box sx={{width: 22, display: "flex", justifyContent: "center", opacity: 0}}>
                            <ExpandMoreIcon fontSize="small"/>
                        </Box>
                    </Box>
                </Stack>

                {forecastStrip && <Box sx={{pt: 1.25}}>{forecastStrip}</Box>}
            </Box>
        );
    }

    return (
        <Box sx={cardShellSx}>
            <Accordion
                disableGutters
                elevation={0}
                sx={{
                    bgcolor: "transparent",
                    boxShadow: "none",
                    borderRadius: 0,
                    "&:before": {display: "none"},
                }}
            >
                <AccordionSummary
                    expandIcon={<ExpandMoreIcon/>}
                    sx={{
                        px: 2,
                        py: 0.5,
                        "& .MuiAccordionSummary-content": {
                            my: 0.8,
                        },
                        "& .MuiAccordionSummary-expandIconWrapper": {
                            width: 22,
                            ml: 0.35,
                            mr: 0,
                            justifyContent: "center",
                            color: "text.secondary",
                        },
                    }}
                >
                    <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{width: "100%", gap: 1}}>
                        <Box sx={{minWidth: 0}}>
                            <Stack direction="row" spacing={0.75} alignItems="center">
                                {titleVisual && (
                                    <Box
                                        sx={{
                                            width: 28,
                                            height: 28,
                                            borderRadius: 2,
                                            display: "flex",
                                            alignItems: "center",
                                            justifyContent: "center",
                                            color: titleVisual.color,
                                            bgcolor: titleVisual.bg,
                                            flexShrink: 0,
                                        }}
                                    >
                                        {titleVisual.icon}
                                    </Box>
                                )}
                                <Typography variant="h6" sx={CARD_TITLE_SX}>
                                    {title}
                                </Typography>
                            </Stack>
                            {closedCount > 0 && (
                                <Typography variant="caption" sx={{fontWeight: 700, color: "error.main"}}>
                                    {allClosed ? "All locations closed" : `${closedCount} closed right now`}
                                </Typography>
                            )}
                        </Box>
                        {allClosed ? (
                            <Box sx={{...metricColumnSx, ...metricStackSx}}>
                                <Typography
                                    variant="body1"
                                    sx={{fontWeight: 900, color: "error.main", letterSpacing: 0.4}}
                                >
                                    CLOSED
                                </Typography>
                            </Box>
                        ) : (
                            <Box sx={{...metricColumnSx, ...metricStackSx}}>
                                <Typography variant="body1" sx={{fontWeight: 800, fontVariantNumeric: "tabular-nums"}}>
                                    {displayTotal} / {max}
                                </Typography>
                                <Typography variant="body2" sx={{fontWeight: 700, color: percentColor, fontVariantNumeric: "tabular-nums"}}>
                                    {displayPercent}% full
                                </Typography>
                            </Box>
                        )}
                    </Stack>
                </AccordionSummary>

                <AccordionDetails sx={{pt: 0, px: 2, pb: 2}}>
                    {forecastStrip}
                    {locationRows}
                </AccordionDetails>
            </Accordion>
        </Box>
    );
}
