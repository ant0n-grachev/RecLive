import {Box, Stack, Typography} from "@mui/material";
import type {Location} from "../lib/types/facility";
import ModernCard from "../shared/components/ModernCard";
import {
    CARD_TITLE_SX,
    clampPercent,
    combineOccupancyThresholds,
    getOccupancyColor,
    INNER_SURFACE_SX,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {getSectionVisual} from "./sectionIcons";
import {computeOccupancySummary} from "../shared/occupancy/computeOccupancySummary";

interface Props {
    title: string;
    exclude: number[];
    locations: Location[];
    nowTs: number;
    occupancyThresholds?: OccupancyThresholds | null;
    locationOccupancyThresholds?: Partial<Record<number, OccupancyThresholds>>;
}

export default function SectionSummaryOther({
    title,
    exclude,
    locations,
    nowTs,
    occupancyThresholds = null,
    locationOccupancyThresholds = {},
}: Props) {
    const list = locations
        .filter((l) => !exclude.includes(l.locationId))
        .sort((a, b) => a.locationName.localeCompare(b.locationName));

    if (list.length === 0) {
        return null;
    }

    const summary = computeOccupancySummary(list, {nowMs: nowTs});
    const locationModels = list.map((location) => ({
        location,
        summary: computeOccupancySummary([location], {nowMs: nowTs}),
    }));
    const hasObservedOccupancy = (
        (summary.status === "live" || summary.status === "partial")
        && summary.count !== null
        && summary.percent !== null
    );
    const percent = summary.percent === null ? 0 : clampPercent(summary.percent);
    const summaryThresholds = combineOccupancyThresholds(
        list.map((loc) => ({
            thresholds: locationOccupancyThresholds[loc.locationId],
            weight: typeof loc.maxCapacity === "number" && Number.isFinite(loc.maxCapacity)
                ? Math.max(0, loc.maxCapacity)
                : 0,
        }))
    ) ?? occupancyThresholds;
    const color = getOccupancyColor(percent, summaryThresholds);
    const allClosed = summary.status === "closed";
    const titleVisual = getSectionVisual(title);

    return (
        <ModernCard>
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
                <Typography variant="h6" sx={CARD_TITLE_SX}>{title}</Typography>
            </Stack>

            {allClosed ? (
                <Typography variant="h5" sx={{fontWeight: 900, color: "error.main", letterSpacing: 0.4}}>
                    CLOSED
                </Typography>
            ) : hasObservedOccupancy && summary.count !== null ? (
                <Typography variant="h5">{summary.count} / {summary.observedCapacity}</Typography>
            ) : (
                <Typography variant="h6" color="text.secondary" sx={{fontWeight: 800}}>
                    Live occupancy unavailable
                </Typography>
            )}

            {hasObservedOccupancy && (
                <Typography sx={{color, fontWeight: 600}}>{percent}% full</Typography>
            )}
            {summary.status === "partial" && (
                <Typography variant="body2" color="text.secondary" sx={{fontWeight: 600}}>
                    Coverage: {Math.round(summary.coverage * 100)}% of open capacity observed
                </Typography>
            )}

            <Stack spacing={1} sx={{mt: 1}}>
                {locationModels.map(({location: loc, summary: locationSummary}) => {
                    const isClosed = locationSummary.status === "closed";

                    if (isClosed) {
                        return (
                            <Box
                                key={loc.locationId}
                                sx={{
                                    ...INNER_SURFACE_SX,
                                    px: 1.25,
                                    py: 0.9,
                                    display: "flex",
                                    flexDirection: {xs: "column", sm: "row"},
                                    justifyContent: "space-between",
                                    alignItems: {xs: "flex-start", sm: "center"},
                                    gap: {xs: 0.5, sm: 1},
                                    width: "100%",
                                }}
                            >
                                <Typography variant="body2" fontWeight={600}>
                                    {loc.locationName}
                                </Typography>
                                <Typography variant="body2" color="error.main" fontWeight={700}>
                                    CLOSED
                                </Typography>
                            </Box>
                        );
                    }

                    const isObserved = locationSummary.status === "live"
                        && locationSummary.count !== null
                        && locationSummary.percent !== null;
                    const locationThresholds = locationOccupancyThresholds[loc.locationId] ?? occupancyThresholds;

                    return (
                        <Box
                            key={loc.locationId}
                            sx={{
                                ...INNER_SURFACE_SX,
                                px: 1.25,
                                py: 0.9,
                                display: "flex",
                                flexDirection: {xs: "column", sm: "row"},
                                justifyContent: "space-between",
                                alignItems: {xs: "flex-start", sm: "center"},
                                gap: {xs: 0.5, sm: 2},
                                width: "100%",
                            }}
                        >
                            <Typography variant="body2" fontWeight={600}>
                                {loc.locationName}
                            </Typography>

                            <Stack
                                direction={{xs: "column", sm: "row"}}
                                spacing={{xs: 0.5, sm: 1}}
                                alignItems={{xs: "flex-start", sm: "center"}}
                                sx={{textAlign: {xs: "left", sm: "inherit"}}}
                            >
                                {isObserved && locationSummary.count !== null && locationSummary.percent !== null ? (
                                    <>
                                        <Typography variant="body2" fontWeight={600}>
                                            {locationSummary.count} / {locationSummary.observedCapacity}
                                        </Typography>
                                        <Typography variant="body2" sx={{color: getOccupancyColor(locationSummary.percent, locationThresholds)}}>
                                            ({Math.round(locationSummary.percent)}%)
                                        </Typography>
                                    </>
                                ) : (
                                    <Typography variant="body2" color="text.secondary" fontWeight={700}>
                                        Live occupancy unavailable
                                    </Typography>
                                )}
                            </Stack>
                        </Box>
                    );
                })}
            </Stack>
        </ModernCard>
    );
}
