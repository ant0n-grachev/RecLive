import {useEffect, useMemo, useRef, useState} from "react";
import {
    Alert,
    Box,
    Button,
    CircularProgress,
    MenuItem,
    Stack,
    TextField,
    Typography,
} from "@mui/material";
import {AnimatePresence, motion} from "framer-motion";
import CheckRoundedIcon from "@mui/icons-material/CheckRounded";
import type {FacilityId} from "../lib/types/facility";
import {
    ensurePushSubscription,
    getExistingPushSubscription,
    getPushAvailability,
    hasMatchingPushRule,
    isWebPushSupported,
    upsertPushRule,
} from "../lib/api/pushNotifications";
import {FACILITY_SHORT_NAMES} from "../lib/config/facilitySections";

export interface AlertSectionOption {
    key: string;
    label: string;
    percent: number;
    total: number;
    max: number;
}

interface Props {
    onClose: () => void;
    facility: FacilityId;
    sections: AlertSectionOption[];
    isOpen: boolean;
    requireStandalonePwaForAlerts?: boolean;
}

interface StoredSubscription {
    sectionKey: string;
    threshold: number;
}

type StoredSubscriptions = Record<number, StoredSubscription>;
type PushStatus = "unsupported" | "idle" | "loading" | "ready" | "blocked" | "error";

const STORAGE_KEY = "reclive:crowd-alert-subscriptions";
const SELECTED_BORDER_COLOR = "rgba(15, 23, 42, 0.85)";
const SELECTED_FOCUS_BORDER_COLOR = "rgba(15, 23, 42, 0.95)";
const UNSELECTED_BORDER_COLOR = "rgba(0, 0, 0, 0.23)";

const readStoredSubscriptions = (): StoredSubscriptions => {
    if (typeof window === "undefined") return {};
    try {
        const text = window.localStorage.getItem(STORAGE_KEY);
        if (!text) return {};
        const parsed = JSON.parse(text) as StoredSubscriptions;
        return parsed && typeof parsed === "object" ? parsed : {};
    } catch {
        return {};
    }
};

const writeStoredSubscriptions = (value: StoredSubscriptions): void => {
    if (typeof window === "undefined") return;
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    } catch {
        // Ignore storage write failures (private mode/quota exceeded).
    }
};

const normalizePercentInt = (value: number): number => Math.max(0, Math.round(value));

const getThresholdUpperBound = (section: AlertSectionOption | null): number => {
    if (!section) return 99;
    return Math.max(0, normalizePercentInt(section.percent) - 1);
};

const resolveInitialSectionKey = (
    facility: FacilityId,
    sections: AlertSectionOption[]
): string => {
    const stored = readStoredSubscriptions()[facility];
    const fallbackSectionKey = sections[0]?.key ?? "";
    return (
        (stored?.sectionKey && sections.some((section) => section.key === stored.sectionKey))
            ? stored.sectionKey
            : fallbackSectionKey
    );
};

const resolveDefaultThresholdInput = (
    facility: FacilityId,
    selectedSection: AlertSectionOption | null
): string => {
    const thresholdUpperBound = getThresholdUpperBound(selectedSection);
    if (thresholdUpperBound < 1) {
        return "";
    }

    const stored = readStoredSubscriptions()[facility];
    const storedThreshold = stored?.threshold;
    const fallbackThreshold = Math.min(40, thresholdUpperBound);
    const preferredThreshold =
        stored?.sectionKey === selectedSection?.key
        && typeof storedThreshold === "number"
        && Number.isFinite(storedThreshold)
            ? Math.round(storedThreshold)
            : fallbackThreshold;
    const clampedThreshold = Math.max(1, Math.min(thresholdUpperBound, preferredThreshold));

    return String(clampedThreshold);
};

export default function CrowdAlertSubscriptionCard({
    onClose,
    facility,
    sections,
    isOpen,
    requireStandalonePwaForAlerts = false,
}: Props) {
    const orderedSections = useMemo(() => {
        const overall = sections.find((section) => section.key === "overall");
        const rest = sections.filter((section) => section.key !== "overall");
        return overall ? [overall, ...rest] : sections;
    }, [sections]);

    const initialSectionKey = useMemo(
        () => resolveInitialSectionKey(facility, orderedSections),
        [facility, orderedSections]
    );
    const [sectionKeyByFacility, setSectionKeyByFacility] = useState<Record<number, string>>({});
    const [thresholdOverrides, setThresholdOverrides] = useState<Record<string, string>>({});
    const [sectionTouched, setSectionTouched] = useState(false);
    const [thresholdTouched, setThresholdTouched] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [isSubmitComplete, setIsSubmitComplete] = useState(false);
    const [pushStatus, setPushStatus] = useState<PushStatus>(() => {
        if (!isWebPushSupported()) return "unsupported";
        if (typeof Notification !== "undefined" && Notification.permission === "denied") return "blocked";
        return "idle";
    });
    const [pushSubscriptionJson, setPushSubscriptionJson] = useState<PushSubscriptionJSON | null>(null);
    const [pushErrorText, setPushErrorText] = useState<string | null>(null);
    const [alertsUnavailableText, setAlertsUnavailableText] = useState<string | null>(null);
    const [isAvailabilityChecking, setIsAvailabilityChecking] = useState(false);
    const submitTimerRef = useRef<number | null>(null);
    const closeTimerRef = useRef<number | null>(null);

    const sectionKey = sectionKeyByFacility[facility] ?? initialSectionKey;
    const facilityName = FACILITY_SHORT_NAMES[facility];
    const selectedSection = orderedSections.find((section) => section.key === sectionKey) ?? null;
    const thresholdContextKey = `${facility}:${sectionKey}`;
    const defaultThresholdInput = useMemo(
        () => resolveDefaultThresholdInput(facility, selectedSection),
        [facility, selectedSection]
    );
    const thresholdInput = thresholdOverrides[thresholdContextKey] ?? defaultThresholdInput;
    const parsedThreshold = Number(thresholdInput);
    const currentOccupancyPercent = selectedSection ? normalizePercentInt(selectedSection.percent) : 100;
    const thresholdUpperBound = getThresholdUpperBound(selectedSection);
    const hasValidThresholdRange = thresholdUpperBound >= 1;
    const isThresholdValid = Number.isInteger(parsedThreshold) && parsedThreshold >= 1 && parsedThreshold <= thresholdUpperBound;
    const canSubscribe = (
        Boolean(selectedSection)
        && hasValidThresholdRange
        && isThresholdValid
        && !isAvailabilityChecking
        && !requireStandalonePwaForAlerts
        && !alertsUnavailableText
        && pushStatus !== "unsupported"
        && pushStatus !== "blocked"
    );
    const isSectionChosen = sectionTouched && sectionKey.length > 0;
    const isThresholdChosen = thresholdTouched && thresholdInput.length > 0;
    const introText = useMemo(() => {
        if (!selectedSection) {
            return "Select a gym area to see current occupancy and set your alert range.";
        }

        const base = `${facilityName}'s ${selectedSection.label} current occupancy is ${selectedSection.total} people (${currentOccupancyPercent}%).`;
        if (!hasValidThresholdRange) {
            return `${base}\nThere is no lower threshold available yet.`;
        }
        return `${base}\nChoose a threshold between 1-${thresholdUpperBound}.`;
    }, [
        selectedSection,
        facilityName,
        currentOccupancyPercent,
        hasValidThresholdRange,
        thresholdUpperBound,
    ]);

    const cancelAutoClose = () => {
        if (closeTimerRef.current !== null) {
            window.clearTimeout(closeTimerRef.current);
            closeTimerRef.current = null;
        }
    };

    const cancelSubmitProgress = () => {
        if (submitTimerRef.current !== null) {
            window.clearTimeout(submitTimerRef.current);
            submitTimerRef.current = null;
        }
    };

    const unlockSubscriptionState = () => {
        if (!isSubmitComplete && !isSubmitting) return;
        setIsSubmitting(false);
        setIsSubmitComplete(false);
        cancelSubmitProgress();
        cancelAutoClose();
    };

    useEffect(() => {
        return () => {
            cancelSubmitProgress();
            cancelAutoClose();
        };
    }, []);

    useEffect(() => {
        if (!isWebPushSupported()) return;

        let isCancelled = false;
        void getExistingPushSubscription()
            .then((subscription) => {
                if (isCancelled) return;
                if (typeof Notification !== "undefined" && Notification.permission === "denied") {
                    setPushStatus("blocked");
                    return;
                }
                if (!subscription) return;
                setPushSubscriptionJson(subscription.toJSON());
                setPushStatus("ready");
            })
            .catch((error) => {
                if (isCancelled) return;
                console.error("Failed to read push subscription", error);
            });

        return () => {
            isCancelled = true;
        };
    }, []);

    useEffect(() => {
        if (!isOpen) return;
        if (requireStandalonePwaForAlerts) {
            setIsAvailabilityChecking(false);
            setAlertsUnavailableText(null);
            return;
        }

        let isCancelled = false;
        const loadAvailability = async () => {
            setIsAvailabilityChecking(true);
            setAlertsUnavailableText(null);

            try {
                const availability = await getPushAvailability();
                if (isCancelled) return;
                if (!availability.alertsAvailable) {
                    if (availability.reason === "push_rules_db_unavailable") {
                        setAlertsUnavailableText("Alerts are temporarily unavailable because the data service is down.");
                    } else if (availability.reason === "push_vapid_unconfigured") {
                        setAlertsUnavailableText("Alerts are temporarily unavailable while notification keys are being configured.");
                    } else {
                        setAlertsUnavailableText("Alerts are temporarily unavailable right now. Please try again shortly.");
                    }
                } else {
                    setAlertsUnavailableText(null);
                }
            } catch {
                if (isCancelled) return;
                setAlertsUnavailableText("Alerts are temporarily unavailable right now. Please try again shortly.");
            } finally {
                if (!isCancelled) {
                    setIsAvailabilityChecking(false);
                }
            }
        };

        void loadAvailability();
        return () => {
            isCancelled = true;
        };
    }, [isOpen, requireStandalonePwaForAlerts]);

    const handleSubscribe = async () => {
        if (!canSubscribe || !selectedSection || isSubmitting || isSubmitComplete) return;
        if (requireStandalonePwaForAlerts) {
            setPushStatus("error");
            setPushErrorText("Install this app to your Home Screen to enable alerts.");
            return;
        }
        if (!isWebPushSupported()) {
            setPushStatus("unsupported");
            setPushErrorText("Push notifications are not supported in this browser.");
            return;
        }

        const normalizedThreshold = Math.max(1, Math.min(thresholdUpperBound, Math.round(parsedThreshold)));
        setPushErrorText(null);
        setPushStatus("loading");
        setIsSubmitting(true);
        setIsSubmitComplete(false);
        cancelSubmitProgress();
        cancelAutoClose();

        try {
            let subscriptionJson = pushSubscriptionJson;
            if (!subscriptionJson) {
                const subscription = await ensurePushSubscription();
                subscriptionJson = subscription.toJSON();
                setPushSubscriptionJson(subscriptionJson);
            }
            if (!subscriptionJson) {
                throw new Error("Push subscription unavailable");
            }
            setPushStatus("ready");
            const existingRule = await hasMatchingPushRule({
                subscription: subscriptionJson,
                facilityId: facility,
                sectionKey: selectedSection.key,
                threshold: normalizedThreshold,
            });
            if (existingRule) {
                setIsSubmitting(false);
                setIsSubmitComplete(false);
                setPushStatus("ready");
                setPushErrorText("This notification is already set for this gym area and threshold.");
                return;
            }

            await upsertPushRule({
                subscription: subscriptionJson,
                facilityId: facility,
                sectionKey: selectedSection.key,
                threshold: normalizedThreshold,
            });

            const stored = readStoredSubscriptions();
            stored[facility] = {
                sectionKey: selectedSection.key,
                threshold: normalizedThreshold,
            };
            writeStoredSubscriptions(stored);
            setThresholdOverrides((prev) => ({
                ...prev,
                [thresholdContextKey]: String(normalizedThreshold),
            }));

            submitTimerRef.current = window.setTimeout(() => {
                setIsSubmitting(false);
                setIsSubmitComplete(true);
                closeTimerRef.current = window.setTimeout(() => {
                    onClose();
                }, 3000);
            }, 900);
        } catch (error) {
            console.error("Failed to save push rule", error);
            const denied = typeof Notification !== "undefined" && Notification.permission === "denied";
            const reason = error instanceof Error ? error.message : "Unknown error";
            setIsSubmitting(false);
            setIsSubmitComplete(false);
            setPushStatus(denied ? "blocked" : "error");
            setPushErrorText(
                denied
                    ? "Notifications are blocked in browser settings."
                    : `Could not save this notification right now. ${reason}`
            );
        }
    };

    return (
        <Stack spacing={1.25}>
            <Typography variant="body2" color="text.secondary" sx={{whiteSpace: "pre-line"}}>
                {introText}
            </Typography>
            <Typography variant="caption" color="text.secondary">
                You'll get <Box component="span" sx={{fontWeight: 700}}>one-time</Box> notification when occupancy reaches your threshold or lower.
            </Typography>
            <Box sx={{height: 6}}/>
            {requireStandalonePwaForAlerts && (
                <Alert severity="info" variant="outlined" sx={{borderRadius: 2}}>
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
                <Alert severity="warning" variant="outlined" sx={{borderRadius: 2}}>
                    {alertsUnavailableText}
                </Alert>
            )}

            <TextField
                label="Gym area"
                value={sectionKey}
                onChange={(event) => {
                    unlockSubscriptionState();
                    setPushErrorText(null);
                    const nextSectionKey = event.target.value;
                    const nextSection = orderedSections.find((section) => section.key === nextSectionKey) ?? null;
                    const nextContextKey = `${facility}:${nextSectionKey}`;
                    const nextDefaultThreshold = resolveDefaultThresholdInput(facility, nextSection);

                    setSectionTouched(true);
                    setSectionKeyByFacility((prev) => ({
                        ...prev,
                        [facility]: nextSectionKey,
                    }));
                    setThresholdTouched(false);
                    setThresholdOverrides((prev) => ({
                        ...prev,
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
                    <MenuItem key={section.key} value={section.key}>
                        {section.label}
                    </MenuItem>
                ))}
            </TextField>

            <TextField
                label="Alert threshold (%)"
                type="number"
                value={thresholdInput}
                onChange={(event) => {
                    unlockSubscriptionState();
                    setPushErrorText(null);
                    setThresholdTouched(true);
                    const next = event.target.value;
                    setThresholdOverrides((prev) => ({
                        ...prev,
                        [thresholdContextKey]: next,
                    }));
                }}
                onBlur={() => {
                    if (!hasValidThresholdRange) {
                        setThresholdOverrides((prev) => ({
                            ...prev,
                            [thresholdContextKey]: "",
                        }));
                        return;
                    }
                    const value = Number(thresholdInput);
                    if (!Number.isFinite(value)) {
                        const fallback = String(Math.min(40, thresholdUpperBound));
                        setThresholdOverrides((prev) => ({
                            ...prev,
                            [thresholdContextKey]: fallback,
                        }));
                        return;
                    }
                    const clamped = Math.max(1, Math.min(thresholdUpperBound, Math.round(value)));
                    setThresholdOverrides((prev) => ({
                        ...prev,
                        [thresholdContextKey]: String(clamped),
                    }));
                }}
                inputProps={{min: 1, max: Math.max(1, thresholdUpperBound), step: 1, inputMode: "numeric"}}
                size="small"
                fullWidth
                disabled={requireStandalonePwaForAlerts || !hasValidThresholdRange}
                error={(thresholdInput.length > 0 && !isThresholdValid) || !hasValidThresholdRange}
                helperText={
                    requireStandalonePwaForAlerts
                        ? "Install the PWA on mobile to enable alerts."
                        : !hasValidThresholdRange
                        ? `Current occupancy is ${currentOccupancyPercent}%, so there is no lower threshold to set yet.`
                        : (thresholdInput.length > 0 && !isThresholdValid
                            ? `Enter a number between 1 and ${thresholdUpperBound}.`
                            : `Choose 1-${thresholdUpperBound}.`)
                }
                sx={{
                    "& input[type=number]": {
                        MozAppearance: "textfield",
                    },
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
            {!requireStandalonePwaForAlerts
                && (pushStatus === "unsupported" || pushStatus === "blocked" || pushStatus === "error" || Boolean(pushErrorText)) && (
                <Typography
                    variant="caption"
                    color={(pushStatus === "unsupported" || pushStatus === "blocked" || pushStatus === "error") ? "error.main" : "text.secondary"}
                >
                    {pushStatus === "unsupported" && "Push notifications are not supported in this browser."}
                    {pushStatus === "blocked" && "Notifications are blocked in browser settings."}
                    {pushStatus === "error" && (pushErrorText ?? "Could not set this notification right now.")}
                    {pushStatus !== "unsupported" && pushStatus !== "blocked" && pushStatus !== "error" && pushErrorText}
                </Typography>
            )}

            <Stack direction="row" justifyContent="flex-end" sx={{pt: 0.25}}>
                <Button
                    size="small"
                    variant="contained"
                    onClick={() => {
                        void handleSubscribe();
                    }}
                    disabled={!canSubscribe || isSubmitting || isSubmitComplete}
                    sx={{
                        borderRadius: 999,
                        textTransform: "none",
                        fontWeight: 700,
                        fontSize: "1rem",
                        width: 142,
                        minWidth: 142,
                        minHeight: 44,
                        bgcolor: isSubmitComplete ? "success.main" : undefined,
                        "&:hover": {
                            bgcolor: isSubmitComplete ? "success.dark" : undefined,
                        },
                        "&.Mui-disabled": isSubmitComplete
                            ? {
                                bgcolor: "success.main",
                                color: "#ffffff",
                                opacity: 1,
                            }
                            : undefined,
                    }}
                >
                    <Box sx={{display: "inline-flex", alignItems: "center", justifyContent: "center", width: "100%"}}>
                        <AnimatePresence mode="wait" initial={false}>
                            {isSubmitComplete ? (
                                <Box
                                    key="check"
                                    component={motion.span}
                                    initial={{opacity: 0, scale: 0.7, y: 3}}
                                    animate={{opacity: 1, scale: 1, y: 0}}
                                    exit={{opacity: 0, scale: 0.7, y: -3}}
                                    transition={{duration: 0.2, ease: [0.22, 1, 0.36, 1]}}
                                    sx={{display: "inline-flex"}}
                                >
                                    <CheckRoundedIcon fontSize="small"/>
                                </Box>
                            ) : isSubmitting ? (
                                <Box
                                    key="loading"
                                    component={motion.span}
                                    initial={{opacity: 0, scale: 0.85}}
                                    animate={{opacity: 1, scale: 1}}
                                    exit={{opacity: 0, scale: 0.85}}
                                    transition={{duration: 0.2, ease: [0.22, 1, 0.36, 1]}}
                                    sx={{display: "inline-flex"}}
                                >
                                    <CircularProgress size={15} thickness={6} color="inherit"/>
                                </Box>
                            ) : (
                                <Box
                                    key="label"
                                    component={motion.span}
                                    initial={{opacity: 0, y: 3}}
                                    animate={{opacity: 1, y: 0}}
                                    exit={{opacity: 0, y: -3}}
                                    transition={{duration: 0.2, ease: [0.22, 1, 0.36, 1]}}
                                >
                                    Set alert
                                </Box>
                            )}
                        </AnimatePresence>
                    </Box>
                </Button>
            </Stack>
        </Stack>
    );
}
