import {Box, Link, Typography} from "@mui/material";
import type {FacilityId} from "../lib/types/facility";
import ModernCard from "../shared/components/ModernCard";
import {clampPercent, getOccupancyColor} from "../shared/utils/styles";
import {formatChicagoTime, getChicagoDayAge} from "../shared/utils/chicagoTime";

interface Props {
    title: string;
    total: number;
    max: number;
    lastUpdated?: string | null;
    facilityId: FacilityId;
    isLoading: boolean;
}

const FACILITY_LINKS: Record<FacilityId, {label: string; href: string}> = {
    1186: {label: "Nick", href: "https://recwell.wisc.edu/locations/nick/"},
    1656: {label: "Bakke", href: "https://recwell.wisc.edu/locations/bakke/"},
};

export default function OccupancyCard({title, total, max, lastUpdated, facilityId, isLoading}: Props) {
    const percent = clampPercent(max ? (total / max) * 100 : 0);
    const color = getOccupancyColor(percent);
    const facilityLink = FACILITY_LINKS[facilityId];
    const daysAgo = getChicagoDayAge(lastUpdated);

    let formatted: string | null = null;

    const time = formatChicagoTime(lastUpdated);
    if (time) {
        let descriptor: string;

        if (daysAgo === 0) {
            descriptor = "Updated today at";
        } else if (daysAgo === 1) {
            descriptor = "Updated yesterday at";
        } else if (typeof daysAgo === "number") {
            descriptor = `Updated ${daysAgo} days ago at`;
        } else {
            descriptor = "Updated at";
        }

        formatted = `${descriptor} ${time}`;
    }

    const showStaleNotice = !isLoading && typeof daysAgo === "number" && daysAgo >= 1;

    return (
        <ModernCard>
            <Typography variant="subtitle2" color="text.secondary">
                {title}
            </Typography>

            <Typography variant="h4" fontWeight={700}>
                {total} / {max}
            </Typography>

            <Typography sx={{color, fontWeight: 600}}>{percent}% full</Typography>

            <Box sx={{mt: 1, display: "flex", flexDirection: "column", gap: 0.5}}>
                {showStaleNotice && (
                    <Typography variant="body2" sx={{color: "warning.main", fontWeight: 600}}>
                        {facilityLink.label} might be closed.
                        Check the Official Website below for Hours of Operation.
                    </Typography>
                )}

                {formatted && (
                    <Typography variant="body2" color="text.secondary">
                        {formatted}
                    </Typography>
                )}

                <Typography variant="body2" color="text.secondary">
                    Official Website:
                    {" "}
                    <Link href={facilityLink.href} target="_blank" rel="noopener noreferrer" underline="hover">
                        {facilityLink.label}
                    </Link>
                </Typography>
            </Box>
        </ModernCard>
    );
}
