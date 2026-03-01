import {Box, Stack, Typography} from "@mui/material";
import type {Location} from "../lib/types/facility";
import ModernCard from "../shared/components/ModernCard";
import {clampPercent, getOccupancyColor} from "../shared/utils/styles";

interface Props {
    title: string;
    exclude: number[];
    locations: Location[];
}

export default function SectionSummaryOther({title, exclude, locations}: Props) {
    const list = locations
        .filter((l) => !exclude.includes(l.locationId))
        .sort((a, b) => a.locationName.localeCompare(b.locationName));

    if (list.length === 0) {
        return null;
    }

    const total = list.reduce((s, l) => s + (l.currentCapacity ?? 0), 0);
    const max = list.reduce((s, l) => s + (l.maxCapacity ?? 0), 0);
    const percent = clampPercent(max ? (total / max) * 100 : 0);
    const color = getOccupancyColor(percent);

    return (
        <ModernCard>
            <Typography variant="h6">{title}</Typography>

            <Typography variant="h5">{total} / {max}</Typography>

            <Typography sx={{color, fontWeight: 600}}>{percent}% full</Typography>

            <Stack spacing={1} sx={{mt: 1}}>
                {list.map((loc) => {
                    const isClosed = loc.isClosed === true;

                    if (isClosed) {
                        return (
                            <Box
                                key={loc.locationId}
                                sx={{
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

                    return (
                        <Box
                            key={loc.locationId}
                            sx={{
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
                                <Typography variant="body2" sx={{color: getOccupancyColor(p)}}>
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
