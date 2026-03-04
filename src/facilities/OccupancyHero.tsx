import {useCallback, useEffect, useRef, useState} from "react";
import {Box, Button, LinearProgress, Link, Stack, Typography} from "@mui/material";
import {alpha, useTheme, type PaletteMode} from "@mui/material/styles";
import {animate, useMotionValue, useReducedMotion} from "framer-motion";
import DarkModeRoundedIcon from "@mui/icons-material/DarkModeRounded";
import LightModeRoundedIcon from "@mui/icons-material/LightModeRounded";
import type {FacilityId} from "../lib/types/facility";
import {
    CARD_SHELL_SX,
    clampPercent,
    getOccupancyColor,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {formatChicagoUpdatedRelative} from "../shared/utils/chicagoTime";
import {EXTERNAL_LINK_REL, openExternalInBrowser} from "../shared/utils/externalLink";

interface Props {
    total: number;
    max: number;
    lastUpdated?: string | null;
    facilityId: FacilityId;
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
    occupancyThresholds?: OccupancyThresholds | null;
    onOpenAlerts?: () => void;
}

const FACILITY_LINKS: Record<FacilityId, {label: string; href: string}> = {
    1186: {label: "Nick", href: "https://recwell.wisc.edu/locations/nick/"},
    1656: {label: "Bakke", href: "https://recwell.wisc.edu/locations/bakke/"},
};

export default function OccupancyHero({
    total,
    max,
    lastUpdated,
    facilityId,
    themeMode,
    onThemeModeChange,
    occupancyThresholds = null,
    onOpenAlerts,
}: Props) {
    const theme = useTheme();
    const reduceMotion = useReducedMotion();
    const progressMotion = useMotionValue(0);
    const [animatedTotal, setAnimatedTotal] = useState(0);
    const [animatedPercent, setAnimatedPercent] = useState(0);
    const [animatedBarPercent, setAnimatedBarPercent] = useState(0);
    const animatedTotalRef = useRef(0);
    const animatedPercentRef = useRef(0);
    const animatedBarPercentRef = useRef(0);
    const animationRangeRef = useRef({
        fromTotal: 0,
        toTotal: 0,
        fromPercent: 0,
        toPercent: 0,
    });

    const sampleAnimationRange = (progressValue: number) => {
        const clampedProgress = Math.max(0, Math.min(1, progressValue));
        const {fromTotal, toTotal, fromPercent, toPercent} = animationRangeRef.current;
        return {
            total: fromTotal + (toTotal - fromTotal) * clampedProgress,
            percent: fromPercent + (toPercent - fromPercent) * clampedProgress,
            barPercent: fromPercent + (toPercent - fromPercent) * clampedProgress,
        };
    };

    useEffect(() => {
        const unsubscribe = progressMotion.on("change", (progress) => {
            const sampled = sampleAnimationRange(progress);
            const nextTotal = sampled.total;
            const nextPercent = sampled.percent;
            const nextBarPercent = sampled.barPercent;

            animatedTotalRef.current = nextTotal;
            animatedPercentRef.current = nextPercent;
            animatedBarPercentRef.current = nextBarPercent;
            setAnimatedTotal(nextTotal);
            setAnimatedPercent(nextPercent);
            setAnimatedBarPercent(nextBarPercent);
        });

        return () => {
            unsubscribe();
        };
    }, [progressMotion]);

    useEffect(() => {
        const nextPercent = max ? (total / max) * 100 : 0;
        const sampledCurrent = sampleAnimationRange(progressMotion.get());
        const fromTotal = sampledCurrent.total;
        const fromPercent = sampledCurrent.percent;
        const fromBarPercent = sampledCurrent.barPercent;

        animatedTotalRef.current = fromTotal;
        animatedPercentRef.current = fromPercent;
        animatedBarPercentRef.current = fromBarPercent;

        const unchanged = fromTotal === total && fromPercent === nextPercent;
        if (unchanged) return;

        if (reduceMotion) {
            animatedTotalRef.current = total;
            animatedPercentRef.current = nextPercent;
            animatedBarPercentRef.current = nextPercent;
            return;
        }

        animationRangeRef.current = {
            fromTotal,
            toTotal: total,
            fromPercent,
            toPercent: nextPercent,
        };
        progressMotion.set(0);

        const timeline = animate(progressMotion, 1, {
            duration: 0.95,
            ease: [0.22, 1, 0.36, 1],
            onComplete: () => {
                animatedTotalRef.current = total;
                animatedPercentRef.current = nextPercent;
                animatedBarPercentRef.current = nextPercent;
                setAnimatedTotal(total);
                setAnimatedPercent(nextPercent);
                setAnimatedBarPercent(nextPercent);
            },
        });

        return () => {
            timeline.stop();
        };
    }, [max, progressMotion, reduceMotion, total]);

    const immediatePercent = max ? (total / max) * 100 : 0;
    const displayTotal = reduceMotion ? Math.max(0, total) : Math.max(0, Math.round(animatedTotal));
    const percent = reduceMotion ? clampPercent(immediatePercent) : clampPercent(animatedPercent);
    const barPercent = reduceMotion ? percent : Math.max(0, Math.min(100, animatedBarPercent));
    const occupancyColor = getOccupancyColor(percent, occupancyThresholds);
    const progressTrackBg = alpha(theme.palette.text.primary, theme.palette.mode === "dark" ? 0.24 : 0.1);
    const facilityLink = FACILITY_LINKS[facilityId];
    const relativeUpdatedText = formatChicagoUpdatedRelative(lastUpdated);
    const hoverThemeButtonBg = alpha(theme.palette.text.primary, theme.palette.mode === "dark" ? 0.14 : 0.06);
    const handleThemeToggle = useCallback(
        () => onThemeModeChange(themeMode === "dark" ? "light" : "dark"),
        [onThemeModeChange, themeMode]
    );
    const activeThemeLabel = themeMode === "dark" ? "Dark" : "Light";
    const activeThemeIcon = themeMode === "dark"
        ? <DarkModeRoundedIcon sx={{fontSize: 16, mr: 0.5}}/>
        : <LightModeRoundedIcon sx={{fontSize: 16, mr: 0.5}}/>;

    return (
        <Box
            sx={{
                ...CARD_SHELL_SX,
                p: {xs: 2, sm: 2.25},
            }}
        >
            <Stack spacing={1.25}>
                <Stack direction="row" alignItems="center" justifyContent="space-between" spacing={1}>
                    <Typography variant="subtitle2" color="text.secondary">
                        Live Occupancy
                    </Typography>
                    <Button
                        size="small"
                        onClick={handleThemeToggle}
                        aria-label={`Switch to ${themeMode === "dark" ? "light" : "dark"} theme`}
                        sx={{
                            minHeight: 36,
                            px: 1.2,
                            border: 0,
                            borderRadius: 999,
                            textTransform: "none",
                            fontWeight: 700,
                            color: "text.primary",
                            bgcolor: "action.selected",
                            "&:hover": {
                                border: 0,
                                bgcolor: hoverThemeButtonBg,
                            },
                        }}
                    >
                        {activeThemeIcon}
                        {activeThemeLabel}
                    </Button>
                </Stack>

                <Typography variant="h3" sx={{fontWeight: 800, letterSpacing: -0.5}}>
                    {displayTotal}
                    <Typography
                        component="span"
                        sx={{fontSize: "0.58em", fontWeight: 700, color: "text.secondary", ml: 0.75}}
                    >
                        / {max}
                    </Typography>
                </Typography>

                <Stack direction="row" spacing={1.5} alignItems="center">
                    <Typography sx={{color: occupancyColor, fontWeight: 700, whiteSpace: "nowrap"}}>
                        {percent}% full
                    </Typography>
                    <LinearProgress
                        variant="determinate"
                        value={barPercent}
                        sx={{
                            flexGrow: 1,
                            height: 8,
                            borderRadius: 999,
                            bgcolor: progressTrackBg,
                            "& .MuiLinearProgress-bar": {
                                borderRadius: 999,
                                backgroundColor: occupancyColor,
                                transition: "none",
                            },
                        }}
                    />
                </Stack>

                <Stack direction="row" justifyContent="space-between" alignItems="flex-end" spacing={1.25}>
                    <Stack spacing={0.4} sx={{minWidth: 0, flexGrow: 1}}>
                        {relativeUpdatedText && (
                            <Typography variant="body2" color="text.secondary">
                                {relativeUpdatedText}
                            </Typography>
                        )}
                        <Typography variant="body2" color="text.secondary">
                            Official website:
                            {" "}
                            <Link
                                href={facilityLink.href}
                                target="_blank"
                                rel={EXTERNAL_LINK_REL}
                                underline="hover"
                                onClick={(event) => {
                                    event.preventDefault();
                                    openExternalInBrowser(facilityLink.href);
                                }}
                            >
                                {facilityLink.label}
                            </Link>
                        </Typography>
                    </Stack>

                    {onOpenAlerts && (
                        <Button
                            size="small"
                            variant="outlined"
                            onClick={onOpenAlerts}
                            sx={{
                                minWidth: 72,
                                minHeight: 40,
                                px: 1.3,
                                borderRadius: 999,
                                textTransform: "none",
                                fontWeight: 700,
                                color: "text.secondary",
                                borderColor: "divider",
                                bgcolor: "background.paper",
                                flexShrink: 0,
                            }}
                        >
                            Alerts
                        </Button>
                    )}
                </Stack>
            </Stack>
        </Box>
    );
}
