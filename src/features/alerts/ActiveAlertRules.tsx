import {useId} from "react";
import {Alert, Box, Button, CircularProgress, Stack, Typography} from "@mui/material";
import type {FacilityId} from "../../lib/types/facility";
import type {PushRule} from "../../lib/api/pushNotifications";
import {FACILITY_SHARED_CONFIG, FACILITY_SHORT_NAMES} from "../../lib/config/facilitySections";
import type {AlertSectionOption} from "./alertTypes";

export type ManagementLoadStatus = "idle" | "loading" | "no-subscription" | "success" | "error";

interface Props {
    facility: FacilityId;
    sections: AlertSectionOption[];
    managedRules: PushRule[];
    managementLoadStatus: ManagementLoadStatus;
    managementErrorText: string | null;
    cancellingRuleIds: Set<number>;
    cancelRuleErrors: Record<number, string>;
    isCancellingAll: boolean;
    cancelAllErrorText: string | null;
    isSubmitting: boolean;
    onCancelRule: (rule: PushRule) => void;
    onCancelAll: () => void;
}

const expiryFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
    timeZoneName: "short",
});

const formatExpiry = (expiresAt: string): string => {
    const date = new Date(expiresAt);
    if (!Number.isFinite(date.getTime())) return "Expiry unavailable";
    const parts = expiryFormatter.formatToParts(date);
    const value = (type: Intl.DateTimeFormatPartTypes): string => (
        parts.find((part) => part.type === type)?.value ?? ""
    );
    return `${value("month")} ${value("day")}, ${value("year")} at ${value("hour")}:${value("minute")} ${value("dayPeriod")} ${value("timeZoneName")}`;
};

const resolveRuleSectionLabel = (
    rule: PushRule,
    facility: FacilityId,
    sections: AlertSectionOption[]
): string => {
    if (rule.sectionKey === "overall") return "Entire Facility";
    if (rule.facilityId === facility) {
        const supplied = sections.find((section) => section.key === rule.sectionKey);
        if (supplied) return supplied.label;
    }
    const configured = FACILITY_SHARED_CONFIG[rule.facilityId].sections.find(
        (section) => section.key === rule.sectionKey
    );
    return configured?.title ?? `Area: ${rule.sectionKey}`;
};

export default function ActiveAlertRules({
    facility, sections, managedRules, managementLoadStatus, managementErrorText,
    cancellingRuleIds, cancelRuleErrors, isCancellingAll, cancelAllErrorText,
    isSubmitting, onCancelRule, onCancelAll,
}: Props) {
    const idPrefix = useId().replace(/:/g, "");
    const managementHeadingId = idPrefix + "-manage-alerts-heading";
    const managementErrorId = idPrefix + "-manage-alerts-error";
    const cancelAllErrorId = idPrefix + "-cancel-all-error";
    return (
            <Box
                component="section"
                role="region"
                aria-labelledby={managementHeadingId}
                aria-describedby={managementErrorText ? managementErrorId : undefined}
                sx={{pt: 1}}
            >
                <Stack spacing={1}>
                    <Typography id={managementHeadingId} variant="subtitle2">
                        Manage alerts
                    </Typography>
                    {managementLoadStatus === "loading" && (
                        <Typography variant="caption" color="text.secondary">
                            {managedRules.length > 0 ? "Refreshing active alerts..." : "Loading active alerts..."}
                        </Typography>
                    )}
                    {managementLoadStatus === "no-subscription" && (
                        <Typography variant="caption" color="text.secondary">
                            No active browser subscription was found.
                        </Typography>
                    )}
                    {managementLoadStatus === "success" && managedRules.length === 0 && (
                        <Typography variant="caption" color="text.secondary">
                            No active alerts for this browser.
                        </Typography>
                    )}
                    {managementErrorText && (
                        <Alert id={managementErrorId} severity="error" variant="outlined" sx={{borderRadius: 2}}>
                            {managementErrorText}
                        </Alert>
                    )}
                    {managedRules.length > 0 && (
                        <Box component="ul" aria-label="Active alerts" sx={{m: 0, p: 0, listStyle: "none"}}>
                            {managedRules.map((rule) => {
                                const sectionLabel = resolveRuleSectionLabel(rule, facility, sections);
                                const ruleLabel = `${FACILITY_SHORT_NAMES[rule.facilityId]} ${sectionLabel} at ${rule.threshold}%`;
                                const expiry = formatExpiry(rule.expiresAt);
                                const isCancelling = cancellingRuleIds.has(rule.id);
                                const cancelError = cancelRuleErrors[rule.id];
                                const cancelErrorId = `${idPrefix}-cancel-${rule.id}-error`;
                                return (
                                    <Box
                                        component="li"
                                        key={rule.id}
                                        aria-label={`Alert for ${ruleLabel}, expires ${expiry}`}
                                        sx={{py: 1, borderBottom: "1px solid", borderColor: "divider"}}
                                    >
                                        <Stack direction={{xs: "column", sm: "row"}} spacing={1} justifyContent="space-between">
                                            <Box>
                                                <Typography variant="body2">
                                                    {`${FACILITY_SHORT_NAMES[rule.facilityId]} — ${sectionLabel} — ${rule.threshold}%`}
                                                </Typography>
                                                <Typography
                                                    component="time"
                                                    dateTime={rule.expiresAt}
                                                    variant="caption"
                                                    color="text.secondary"
                                                >
                                                    {`Expires ${expiry}`}
                                                </Typography>
                                            </Box>
                                            <Button
                                                size="small"
                                                variant="outlined"
                                                color="error"
                                                aria-label={`${isCancelling ? "Cancelling" : "Cancel"} alert for ${ruleLabel}`}
                                                aria-describedby={cancelError ? cancelErrorId : undefined}
                                                disabled={
                                                    isCancellingAll
                                                    || isSubmitting
                                                    || cancellingRuleIds.size > 0
                                                    || managementLoadStatus === "loading"
                                                }
                                                onClick={() => void onCancelRule(rule)}
                                                sx={{alignSelf: {xs: "flex-start", sm: "center"}, textTransform: "none"}}
                                            >
                                                {isCancelling
                                                    ? <CircularProgress size={14} thickness={6} color="inherit" aria-hidden="true"/>
                                                    : "Cancel"}
                                            </Button>
                                        </Stack>
                                        {cancelError && (
                                            <Typography id={cancelErrorId} variant="caption" color="error.main" role="alert">
                                                {cancelError}
                                            </Typography>
                                        )}
                                    </Box>
                                );
                            })}
                        </Box>
                    )}
                    {managedRules.length > 0 && (
                        <Box>
                            <Button
                                size="small"
                                color="error"
                                aria-label={isCancellingAll ? "Cancelling all alerts" : "Cancel all alerts"}
                                aria-describedby={cancelAllErrorText ? cancelAllErrorId : undefined}
                                disabled={
                                    isCancellingAll
                                    || isSubmitting
                                    || cancellingRuleIds.size > 0
                                    || managementLoadStatus === "loading"
                                }
                                onClick={() => void onCancelAll()}
                                sx={{textTransform: "none"}}
                            >
                                {isCancellingAll
                                    ? <CircularProgress size={14} thickness={6} color="inherit" aria-hidden="true"/>
                                    : "Cancel all alerts"}
                            </Button>
                            {cancelAllErrorText && (
                                <Typography id={cancelAllErrorId} variant="caption" color="error.main" role="alert">
                                    {cancelAllErrorText}
                                </Typography>
                            )}
                        </Box>
                    )}
                </Stack>
            </Box>

    );
}
