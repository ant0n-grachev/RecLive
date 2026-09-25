import type {ReactNode} from "react";
import {Box, Button, LinearProgress, Link, Stack, Typography} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import type {FacilityId} from "../lib/types/facility";
import {
    CARD_SHELL_SX,
    clampPercent,
    getOccupancyColor,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {formatChicagoUpdatedRelative} from "../shared/utils/chicagoTime";
import {EXTERNAL_LINK_REL, openExternalInBrowser} from "../shared/utils/externalLink";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";

interface Props {
    summary: OccupancySummary;
    nowTs: number;
    facilityId: FacilityId;
    headerAction?: ReactNode;
    occupancyThresholds?: OccupancyThresholds | null;
    onOpenAlerts?: () => void;
}

const FACILITY_LINKS: Record<FacilityId, {label: string; href: string}> = {
    1186: {label: "Nick", href: "https://recwell.wisc.edu/locations/nick/"},
    1656: {label: "Bakke", href: "https://recwell.wisc.edu/locations/bakke/"},
};

export default function OccupancyHero({
    summary,
    nowTs,
    facilityId,
    headerAction = null,
    occupancyThresholds = null,
    onOpenAlerts,
}: Props) {
    const theme = useTheme();
    const hasObservedOccupancy = (
        summary.status === "live"
        && summary.count !== null
        && summary.percent !== null
    );
    const total = hasObservedOccupancy && summary.count !== null ? summary.count : 0;
    const max = hasObservedOccupancy ? summary.observedCapacity : 0;
    const targetPercent = hasObservedOccupancy && summary.percent !== null ? summary.percent : 0;
    const displayTotal = Math.max(0, Math.round(total));
    const percent = clampPercent(targetPercent);
    const barPercent = percent;
    const occupancyColor = getOccupancyColor(percent, occupancyThresholds);
    const progressTrackBg = alpha(theme.palette.text.primary, theme.palette.mode === "dark" ? 0.24 : 0.1);
    const facilityLink = FACILITY_LINKS[facilityId];
    const relativeUpdatedText = formatChicagoUpdatedRelative(summary.latestFetchedAt, new Date(nowTs));
    if (!hasObservedOccupancy && summary.status !== "closed") return null;

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
                    {headerAction}
                </Stack>

                {summary.status === "closed" ? (
                    <Typography variant="h4" sx={{fontWeight: 900, color: "error.main", letterSpacing: 0.4}}>
                        CLOSED
                    </Typography>
                ) : hasObservedOccupancy ? (
                    <>
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
                                aria-label="Current occupancy percentage"
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
                    </>
                ) : null}

                <Stack direction="row" justifyContent="space-between" alignItems="flex-end" spacing={1.25}>
                    <Stack spacing={0.4} sx={{minWidth: 0, flexGrow: 1}}>
                        {hasObservedOccupancy && relativeUpdatedText && (
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
                                underline="always"
                                color="text.primary"
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
                                minHeight: 44,
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
