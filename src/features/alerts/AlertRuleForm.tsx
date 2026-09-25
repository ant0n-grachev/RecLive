import {useId, useMemo, useState} from "react";
import {Alert, Box, Button, CircularProgress, MenuItem, Stack, TextField, Typography} from "@mui/material";
import type {FacilityId} from "../../lib/types/facility";
import {FACILITY_SHORT_NAMES} from "../../lib/config/facilitySections";
import type {AlertSectionOption} from "./alertTypes";
import {getThresholdUpperBound, hasUsableSummary, normalizePercentInt, resolveDefaultThresholdInput, resolveInitialSectionKey} from "./alertSubscriptionStorage";

export interface AlertRuleSubmission {
    readonly sectionKey: string;
    readonly threshold: number;
}

export interface SuccessfulAlertDraft extends AlertRuleSubmission {
    readonly facility: FacilityId;
}

interface Props {
    facility: FacilityId;
    sections: AlertSectionOption[];
    requireStandalonePwaForAlerts: boolean;
    isAvailabilityChecking: boolean;
    alertsUnavailableText: string | null;
    creationErrorText: string | null;
    creationBlocked: boolean;
    isSubmitting: boolean;
    successfulDraft: SuccessfulAlertDraft | null;
    onSubmit: (request: AlertRuleSubmission) => void;
    resetSubscribeError: () => void;
}

const SELECTED_BORDER_COLOR = "rgba(15, 23, 42, 0.85)";
const SELECTED_FOCUS_BORDER_COLOR = "rgba(15, 23, 42, 0.95)";
const UNSELECTED_BORDER_COLOR = "rgba(0, 0, 0, 0.23)";

export default function AlertRuleForm({
    facility, sections, requireStandalonePwaForAlerts, isAvailabilityChecking,
    alertsUnavailableText, creationErrorText, creationBlocked, isSubmitting,
    successfulDraft, onSubmit, resetSubscribeError,
}: Props) {
    const idPrefix = useId().replace(/:/g, "");
    const availabilityErrorId = idPrefix + "-availability-error";
    const standaloneInfoId = idPrefix + "-standalone-info";
    const subscribeErrorId = idPrefix + "-subscribe-error";

    const orderedSections = useMemo(() => {
        const visibleSections = sections.filter((section) => hasUsableSummary(section) || section.summary.status === "closed");
        const overall = visibleSections.find((section) => section.key === "overall");
        const rest = visibleSections.filter((section) => section.key !== "overall");
        return overall ? [overall, ...rest] : visibleSections;
    }, [sections]);
    const initialSectionKey = useMemo(
        () => resolveInitialSectionKey(facility, orderedSections),
        [facility, orderedSections]
    );

    const [sectionKeyByFacility, setSectionKeyByFacility] = useState<Record<number, string>>({});
    const [thresholdOverrides, setThresholdOverrides] = useState<Record<string, string>>({});
    const [sectionTouched, setSectionTouched] = useState(false);
    const [thresholdTouched, setThresholdTouched] = useState(false);
    const [appliedSuccessfulDraft, setAppliedSuccessfulDraft] = useState<SuccessfulAlertDraft | null>(null);
    // The coordinator emits this only after its captured operation is still current.
    // Apply it to that captured facility/section, even if the visible draft changed.
    if (successfulDraft && successfulDraft !== appliedSuccessfulDraft) {
        setAppliedSuccessfulDraft(successfulDraft);
        setThresholdOverrides((previous) => ({
            ...previous,
            [successfulDraft.facility + ":" + successfulDraft.sectionKey]: String(successfulDraft.threshold),
        }));
    }

    const requestedSectionKey = sectionKeyByFacility[facility] ?? initialSectionKey;
    const sectionKey = orderedSections.some(
        (section) => section.key === requestedSectionKey && hasUsableSummary(section)
    ) ? requestedSectionKey : initialSectionKey;

    if (sectionKeyByFacility[facility] !== sectionKey) {
        setSectionKeyByFacility((previous) => ({...previous, [facility]: sectionKey}));
    }

    const facilityName = FACILITY_SHORT_NAMES[facility];
    const selectedSection = orderedSections.find(
        (section) => section.key === sectionKey && hasUsableSummary(section)
    ) ?? null;
    const thresholdContextKey = `${facility}:${sectionKey}`;
    const defaultThresholdInput = useMemo(
        () => resolveDefaultThresholdInput(facility, selectedSection),
        [facility, selectedSection]
    );
    const thresholdInput = thresholdOverrides[thresholdContextKey] ?? defaultThresholdInput;
    const parsedThreshold = Number(thresholdInput);
    const currentOccupancyPercent = selectedSection?.summary.percent !== null
        && selectedSection?.summary.percent !== undefined
        ? normalizePercentInt(selectedSection.summary.percent)
        : 0;
    const thresholdUpperBound = getThresholdUpperBound(selectedSection);
    const hasValidThresholdRange = thresholdUpperBound >= 1;
    const isThresholdValid = Number.isInteger(parsedThreshold)
        && parsedThreshold >= 1
        && parsedThreshold <= thresholdUpperBound;
    const canSubscribe = (
        Boolean(selectedSection)
        && hasValidThresholdRange
        && isThresholdValid
        && !creationBlocked
    );
    const isSectionChosen = sectionTouched && sectionKey.length > 0;
    const isThresholdChosen = thresholdTouched && thresholdInput.length > 0;
    const introText = useMemo(() => {
        if (!selectedSection) {
            return "Select a gym area to see current occupancy and set your alert range.";
        }

        const currentCount = selectedSection.summary.count;
        const base = currentCount === null
            ? `${facilityName}'s ${selectedSection.label} current occupancy is ${currentOccupancyPercent}% full.`
            : `${facilityName}'s ${selectedSection.label} current occupancy is ${currentCount} people (${currentOccupancyPercent}% full).`;
        if (!hasValidThresholdRange) return `${base}\nThere is no lower threshold available yet.`;
        return `${base}\nChoose a threshold between 1-${thresholdUpperBound}.`;
    }, [
        selectedSection,
        facilityName,
        currentOccupancyPercent,
        hasValidThresholdRange,
        thresholdUpperBound,
    ]);

    const subscribeDescriptionIds = [
        requireStandalonePwaForAlerts ? standaloneInfoId : null,
        alertsUnavailableText ? availabilityErrorId : null,
        creationErrorText ? subscribeErrorId : null,
    ].filter((value): value is string => Boolean(value)).join(" ") || undefined;
    return (
        <>
            <Typography variant="body2" color="text.secondary" sx={{whiteSpace: "pre-line"}}>
                {introText}
            </Typography>
            <Typography variant="caption" color="text.secondary">
                You'll get <Box component="span" sx={{fontWeight: 700}}>one-time</Box> notification when occupancy reaches your threshold or lower.
            </Typography>
            <Box sx={{height: 6}}/>
            {requireStandalonePwaForAlerts && (
                <Alert id={standaloneInfoId} severity="info" variant="outlined" sx={{borderRadius: 2}}>
                    Install RecLive as an app (Add to Home Screen) to enable alerts on mobile.
                    The tutorial is at the bottom of the page.
                </Alert>
            )}
            {isAvailabilityChecking && (
                <Typography variant="caption" color="text.secondary">
                    Checking alerts service...
                </Typography>
            )}
            {alertsUnavailableText && (
                <Alert id={availabilityErrorId} severity="warning" variant="outlined" sx={{borderRadius: 2}}>
                    {alertsUnavailableText}
                </Alert>
            )}

            <TextField
                label="Gym area"
                value={sectionKey}
                onChange={(event) => {
                    resetSubscribeError();
                    const nextSectionKey = event.target.value;
                    const nextSection = orderedSections.find(
                        (section) => section.key === nextSectionKey && hasUsableSummary(section)
                    ) ?? null;
                    const nextContextKey = `${facility}:${nextSectionKey}`;
                    const nextDefaultThreshold = resolveDefaultThresholdInput(facility, nextSection);

                    setSectionTouched(true);
                    setSectionKeyByFacility((previous) => ({...previous, [facility]: nextSectionKey}));
                    setThresholdTouched(false);
                    setThresholdOverrides((previous) => ({
                        ...previous,
                        [nextContextKey]: nextDefaultThreshold,
                    }));
                }}
                size="small"
                fullWidth
                select
                disabled={requireStandalonePwaForAlerts}
                sx={{
                    "& .MuiOutlinedInput-root .MuiOutlinedInput-notchedOutline": {
                        borderColor: isSectionChosen ? SELECTED_BORDER_COLOR : UNSELECTED_BORDER_COLOR,
                        borderWidth: isSectionChosen ? 1 : undefined,
                    },
                    "& .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline": {
                        borderColor: isSectionChosen ? SELECTED_FOCUS_BORDER_COLOR : UNSELECTED_BORDER_COLOR,
                        borderWidth: isSectionChosen ? 1 : undefined,
                    },
                    "& .MuiOutlinedInput-root.Mui-error .MuiOutlinedInput-notchedOutline": {
                        borderColor: "error.main",
                    },
                }}
            >
                {orderedSections.map((section) => (
                    <MenuItem key={section.key} value={section.key} disabled={!hasUsableSummary(section)}>
                        {section.label}
                    </MenuItem>
                ))}
            </TextField>

            <TextField
                label="Alert threshold (%)"
                type="number"
                value={thresholdInput}
                onChange={(event) => {
                    resetSubscribeError();
                    setThresholdTouched(true);
                    setThresholdOverrides((previous) => ({
                        ...previous,
                        [thresholdContextKey]: event.target.value,
                    }));
                }}
                onBlur={() => {
                    if (!hasValidThresholdRange) {
                        setThresholdOverrides((previous) => ({...previous, [thresholdContextKey]: ""}));
                        return;
                    }
                    const value = Number(thresholdInput);
                    if (!Number.isFinite(value)) {
                        setThresholdOverrides((previous) => ({
                            ...previous,
                            [thresholdContextKey]: String(Math.min(40, thresholdUpperBound)),
                        }));
                        return;
                    }
                    const clamped = Math.max(1, Math.min(thresholdUpperBound, Math.round(value)));
                    setThresholdOverrides((previous) => ({
                        ...previous,
                        [thresholdContextKey]: String(clamped),
                    }));
                }}
                inputProps={{min: 1, max: Math.max(1, thresholdUpperBound), step: 1, inputMode: "numeric"}}
                size="small"
                fullWidth
                disabled={requireStandalonePwaForAlerts || !selectedSection || !hasValidThresholdRange}
                error={Boolean(selectedSection) && ((thresholdInput.length > 0 && !isThresholdValid) || !hasValidThresholdRange)}
                helperText={
                    requireStandalonePwaForAlerts
                        ? "Install the PWA on mobile to enable alerts."
                        : !selectedSection
                            ? "Current occupancy unavailable."
                            : !hasValidThresholdRange
                                ? `Current occupancy is ${currentOccupancyPercent}%, so there is no lower threshold to set yet.`
                                : (thresholdInput.length > 0 && !isThresholdValid
                                    ? `Enter a number between 1 and ${thresholdUpperBound}.`
                                    : `Choose 1-${thresholdUpperBound}.`)
                }
                sx={{
                    "& input[type=number]": {MozAppearance: "textfield"},
                    "& input[type=number]::-webkit-outer-spin-button, & input[type=number]::-webkit-inner-spin-button": {
                        WebkitAppearance: "none",
                        margin: 0,
                    },
                    "& .MuiOutlinedInput-root .MuiOutlinedInput-notchedOutline": {
                        borderColor: isThresholdChosen ? SELECTED_BORDER_COLOR : UNSELECTED_BORDER_COLOR,
                        borderWidth: isThresholdChosen ? 1 : undefined,
                    },
                    "& .MuiOutlinedInput-root.Mui-focused .MuiOutlinedInput-notchedOutline": {
                        borderColor: isThresholdChosen ? SELECTED_FOCUS_BORDER_COLOR : UNSELECTED_BORDER_COLOR,
                        borderWidth: isThresholdChosen ? 1 : undefined,
                    },
                    "& .MuiOutlinedInput-root.Mui-error .MuiOutlinedInput-notchedOutline": {
                        borderColor: "error.main",
                    },
                }}
            />

            {creationErrorText && !requireStandalonePwaForAlerts && (
                <Typography id={subscribeErrorId} variant="caption" color="error.main" role="alert">
                    {creationErrorText}
                </Typography>
            )}

            <Stack direction="row" justifyContent="flex-end" sx={{pt: 0.25}}>
                <Button
                    size="small"
                    variant="contained"
                    aria-label={isSubmitting ? "Setting alert" : "Set alert"}
                    aria-describedby={subscribeDescriptionIds}
                    onClick={() => {
                        if (canSubscribe && selectedSection && !isSubmitting) {
                            onSubmit({sectionKey: selectedSection.key, threshold: parsedThreshold});
                        }
                    }}
                    disabled={!canSubscribe || isSubmitting}
                    sx={{
                        borderRadius: 999,
                        textTransform: "none",
                        fontWeight: 700,
                        fontSize: "1rem",
                        width: 142,
                        minWidth: 142,
                        minHeight: 44,
                    }}
                >
                    {isSubmitting
                        ? <CircularProgress size={15} thickness={6} color="inherit" aria-hidden="true"/>
                        : "Set alert"}
                </Button>
            </Stack>

        </>
    );
}
