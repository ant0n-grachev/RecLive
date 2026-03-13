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

interface Props {
    title: string;
    exclude: number[];
    locations: Location[];
    occupancyThresholds?: OccupancyThresholds | null;
    locationOccupancyThresholds?: Partial<Record<number, OccupancyThresholds>>;
}

export default function SectionSummaryOther({
    title,
    exclude,
    locations,
    occupancyThresholds = null,
    locationOccupancyThresholds = {},
}: Props) {
    const list = locations
        .filter((l) => !exclude.includes(l.locationId))
        .sort((a, b) => a.locationName.localeCompare(b.locationName));

    if (list.length === 0) {
        return null;
    }

    const total = list.reduce((s, l) => s + (l.currentCapacity ?? 0), 0);
    const max = list.reduce((s, l) => s + (l.maxCapacity ?? 0), 0);
    const percent = clampPercent(max ? (total / max) * 100 : 0);
    const summaryThresholds = combineOccupancyThresholds(
        list.map((loc) => ({
            thresholds: locationOccupancyThresholds[loc.locationId],
            weight: loc.maxCapacity ?? 0,
        }))
    ) ?? occupancyThresholds;
    const color = getOccupancyColor(percent, summaryThresholds);
    const closedCount = list.filter((loc) => loc.isClosed === true).length;
    const allClosed = list.length > 0 && closedCount === list.length;
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
            ) : (
                <Typography variant="h5">{total} / {max}</Typography>
            )}

            {!allClosed && <Typography sx={{color, fontWeight: 600}}>{percent}% full</Typography>}

            <Stack spacing={1} sx={{mt: 1}}>
                {list.map((loc) => {
                    const isClosed = loc.isClosed === true;

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

                    const p = loc.maxCapacity
                        ? clampPercent(
                            ((loc.currentCapacity ?? 0) / loc.maxCapacity) * 100
                        )
                        : 0;
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
                                <Typography variant="body2" fontWeight={600}>
                                    {loc.currentCapacity ?? 0} / {loc.maxCapacity ?? 0}
                                </Typography>
                                <Typography variant="body2" sx={{color: getOccupancyColor(p, locationThresholds)}}>
                                    ({p}%)
                                </Typography>
                            </Stack>
                        </Box>
                    );
                })}
            </Stack>
        </ModernCard>
    );
}
