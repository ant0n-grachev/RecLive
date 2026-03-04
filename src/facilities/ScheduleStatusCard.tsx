import {Box, Chip, Stack, Typography} from "@mui/material";
import CheckCircleOutlineRoundedIcon from "@mui/icons-material/CheckCircleOutlineRounded";
import CloseRoundedIcon from "@mui/icons-material/CloseRounded";
import HelpOutlineRoundedIcon from "@mui/icons-material/HelpOutlineRounded";
import {CARD_SHELL_SX} from "../shared/utils/styles";
import type {FacilityOpenStatus} from "../shared/utils/facilityScheduleStatus";

interface ScheduleStatusCardProps {
    status: FacilityOpenStatus;
}

const STATUS_TEXT: Record<FacilityOpenStatus["state"], string> = {
    open: "Open now according to the official schedule",
    closed: "Closed now according to the official schedule",
    unknown: "Schedule status is temporarily unavailable",
};

const STATUS_COLOR: Record<FacilityOpenStatus["state"], "success" | "error" | "default"> = {
    open: "success",
    closed: "error",
    unknown: "default",
};

const statusIcon = (state: FacilityOpenStatus["state"]) => {
    if (state === "open") {
        return <CheckCircleOutlineRoundedIcon fontSize="small"/>;
    }
    if (state === "closed") {
        return <CloseRoundedIcon fontSize="small"/>;
    }
    return <HelpOutlineRoundedIcon fontSize="small"/>;
};

export default function ScheduleStatusCard({status}: ScheduleStatusCardProps) {
    const detailText = status.matchedRule
        ? `${status.matchedRule.label}: ${status.matchedRule.hours}`
        : null;

    return (
        <Box
            sx={{
                ...CARD_SHELL_SX,
                p: {xs: 1.6, sm: 1.8},
            }}
        >
            <Stack spacing={1}>
                <Stack direction="row" alignItems="center" justifyContent="space-between" gap={1}>
                    <Typography variant="subtitle2" color="text.secondary">
                        Schedule Status
                    </Typography>
                    <Chip
                        size="small"
                        color={STATUS_COLOR[status.state]}
                        icon={statusIcon(status.state)}
                        label={status.state.toUpperCase()}
                        sx={{fontWeight: 700}}
                    />
                </Stack>
                <Typography variant="body1" sx={{fontWeight: 700}}>
                    {STATUS_TEXT[status.state]}
                </Typography>
                {detailText && (
                    <Typography variant="body2" color="text.secondary">
                        {detailText}
                    </Typography>
                )}
            </Stack>
        </Box>
    );
}
