import {
    Accordion,
    AccordionDetails,
    AccordionSummary,
    Box,
    CircularProgress,
    Divider,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableRow,
    Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {CARD_SHELL_SX, CARD_TITLE_SX} from "../shared/utils/styles";
import type {FacilityHoursFacilityPayload} from "../lib/types/facilitySchedule";

interface FacilityHoursBlockProps {
    facilityName: string;
    schedule: FacilityHoursFacilityPayload | null;
    isLoading: boolean;
    error: string | null;
}

const getFirstColumnLabel = (rows: Array<{label: string; hours: string}>): string => {
    const normalized = rows.map((row) => row.label.toLowerCase());
    if (normalized.some((label) => /(monday|tuesday|wednesday|thursday|friday|saturday|sunday|weekdays|weekends)/.test(label))) {
        return "Day / Date";
    }
    return "Date";
};

export default function FacilityHoursBlock({
    facilityName,
    schedule,
    isLoading,
    error,
}: FacilityHoursBlockProps) {
    return (
        <Box
            sx={{
                ...CARD_SHELL_SX,
                overflow: "hidden",
            }}
        >
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
                        py: 0.4,
                        "& .MuiAccordionSummary-content": {
                            my: 0.85,
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
                    <Box sx={{minWidth: 0}}>
                        <Typography variant="h6" sx={CARD_TITLE_SX}>
                            Official Hours, Closures & Notices
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                            {facilityName}
                        </Typography>
                    </Box>
                </AccordionSummary>
                <AccordionDetails sx={{pt: 0, px: 2, pb: 2}}>
                    <Stack spacing={1.5}>
                        {isLoading && !schedule && (
                            <Box sx={{display: "flex", alignItems: "center", gap: 1}}>
                                <CircularProgress size={16} thickness={6}/>
                                <Typography variant="body2" color="text.secondary">
                                    Loading schedule...
                                </Typography>
                            </Box>
                        )}

                        {!isLoading && error && !schedule && (
                            <Typography variant="body2" color="text.secondary">
                                Schedule is unavailable right now.
                            </Typography>
                        )}

                        {schedule && (
                            <Stack spacing={2}>
                                {schedule.sections.length === 0 && (
                                    <Typography variant="body2" color="text.secondary">
                                        No schedule rows are available for this facility yet.
                                    </Typography>
                                )}

                                {schedule.sections.map((section, index) => (
                                    <Box key={`${section.title}-${index}`}>
                                        {index > 0 && <Divider sx={{mb: 1.75}}/>}
                                        <Stack spacing={1.1}>
                                            <Typography variant="subtitle1" sx={{fontWeight: 800}}>
                                                {section.title}
                                            </Typography>

                                            {section.note && (
                                                <Typography variant="body2" color="text.secondary">
                                                    {section.note}
                                                </Typography>
                                            )}

                                            {section.rows.length > 0 && (
                                                <Box sx={{overflowX: "auto", WebkitOverflowScrolling: "touch"}}>
                                                    <Table
                                                        size="small"
                                                        sx={{
                                                            minWidth: 320,
                                                            border: "1px solid",
                                                            borderColor: "divider",
                                                            "& .MuiTableCell-root": {
                                                                borderColor: "divider",
                                                            },
                                                        }}
                                                    >
                                                        <TableHead>
                                                            <TableRow>
                                                                <TableCell sx={{fontWeight: 700}}>
                                                                    {getFirstColumnLabel(section.rows)}
                                                                </TableCell>
                                                                <TableCell sx={{fontWeight: 700}}>
                                                                    Hours
                                                                </TableCell>
                                                            </TableRow>
                                                        </TableHead>
                                                        <TableBody>
                                                            {section.rows.map((row) => (
                                                                <TableRow key={`${section.title}-${row.label}-${row.hours}`}>
                                                                    <TableCell>{row.label}</TableCell>
                                                                    <TableCell>{row.hours}</TableCell>
                                                                </TableRow>
                                                            ))}
                                                        </TableBody>
                                                    </Table>
                                                </Box>
                                            )}
                                        </Stack>
                                    </Box>
                                ))}
                            </Stack>
                        )}
                    </Stack>
                </AccordionDetails>
            </Accordion>
        </Box>
    );
}
