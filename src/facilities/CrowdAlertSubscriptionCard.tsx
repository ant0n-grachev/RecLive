import {useEffect, useId, useMemo, useRef, useState} from "react";
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
import type {FacilityId} from "../lib/types/facility";
import {
    cancelAllPushRules,
    cancelPushRule,
    ensurePushSubscription,
    getExistingPushSubscription,
    getPushAvailability,
    isWebPushSupported,
    listPushRules,
    subscribePushRule,
    type PushRule,
} from "../lib/api/pushNotifications";
import {FACILITY_SHARED_CONFIG, FACILITY_SHORT_NAMES} from "../lib/config/facilitySections";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import {LiveStatusAnnouncer, type LiveStatus} from "./LiveStatusAnnouncer";

export interface AlertSectionOption {
    key: string;
    label: string;
    summary: OccupancySummary;
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

type StoredSubscriptions = Partial<Record<FacilityId, StoredSubscription>>;
type PushStatus = "unsupported" | "idle" | "ready" | "blocked" | "error";
type ManagementLoadStatus = "idle" | "loading" | "no-subscription" | "success" | "error";

const STORAGE_KEY = "reclive:crowd-alert-subscriptions";
const SELECTED_BORDER_COLOR = "rgba(15, 23, 42, 0.85)";
const SELECTED_FOCUS_BORDER_COLOR = "rgba(15, 23, 42, 0.95)";
const UNSELECTED_BORDER_COLOR = "rgba(0, 0, 0, 0.23)";
const LOOKUP_ERROR_TEXT = "Could not access this browser's alerts right now.";
const LIST_ERROR_TEXT = "Could not load alerts right now.";
const SUBSCRIBE_ERROR_TEXT = "Could not save this alert right now.";
const CANCEL_ERROR_TEXT = "Could not cancel this alert right now.";
const CANCEL_ALL_ERROR_TEXT = "Could not cancel alerts right now.";

const isStoredSubscription = (value: unknown): value is StoredSubscription => {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    const candidate = value as Record<string, unknown>;
    return typeof candidate.sectionKey === "string"
        && candidate.sectionKey.length > 0
        && typeof candidate.threshold === "number"
        && Number.isFinite(candidate.threshold);
};

const readStoredSubscriptions = (): StoredSubscriptions => {
    if (typeof window === "undefined") return {};
    try {
        const text = window.localStorage.getItem(STORAGE_KEY);
        if (!text) return {};
        const parsed = JSON.parse(text) as unknown;
        if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};

        const record = parsed as Record<string, unknown>;
        const result: StoredSubscriptions = {};
        const nick = record["1186"];
        const bakke = record["1656"];
        if (isStoredSubscription(nick)) {
            result[1186] = {sectionKey: nick.sectionKey, threshold: nick.threshold};
        }
        if (isStoredSubscription(bakke)) {
            result[1656] = {sectionKey: bakke.sectionKey, threshold: bakke.threshold};
        }
        return result;
    } catch {
        return {};
    }
};

const writeStoredSubscriptions = (value: StoredSubscriptions): void => {
    if (typeof window === "undefined") return;
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    } catch {
        // Storage is only a convenience default; server state remains authoritative.
    }
};

const removeMatchingStoredDefaults = (rules: PushRule[]): void => {
    const stored = readStoredSubscriptions();
    let changed = false;

    for (const rule of rules) {
        const saved = stored[rule.facilityId];
        if (saved?.sectionKey === rule.sectionKey && saved.threshold === rule.threshold) {
            delete stored[rule.facilityId];
            changed = true;
        }
    }

    if (changed) writeStoredSubscriptions(stored);
};

const normalizePercentInt = (value: number): number => Math.max(0, Math.round(value));

const hasUsableSummary = (section: AlertSectionOption): boolean => (
    (section.summary.status === "live" || section.summary.status === "partial")
    && section.summary.percent !== null
    && Number.isFinite(section.summary.percent)
);

const getThresholdUpperBound = (section: AlertSectionOption | null): number => {
    if (!section || !hasUsableSummary(section) || section.summary.percent === null) return 0;
    return Math.min(100, Math.max(0, normalizePercentInt(section.summary.percent) - 1));
};

// Exported here so the saved-selection contract is tested without rendering push UI.
// eslint-disable-next-line react-refresh/only-export-components
export const resolveInitialSectionKey = (
    facility: FacilityId,
    sections: AlertSectionOption[]
): string => {
    const stored = readStoredSubscriptions()[facility];
    const validSections = sections.filter(hasUsableSummary);
    const fallbackSectionKey = validSections[0]?.key ?? "";
    return (
        (stored?.sectionKey && validSections.some((section) => section.key === stored.sectionKey))
            ? stored.sectionKey
            : fallbackSectionKey
    );
};

const resolveDefaultThresholdInput = (
    facility: FacilityId,
    selectedSection: AlertSectionOption | null
): string => {
    const thresholdUpperBound = getThresholdUpperBound(selectedSection);
    if (thresholdUpperBound < 1) return "";

    const stored = readStoredSubscriptions()[facility];
    const storedThreshold = stored?.threshold;
    const fallbackThreshold = Math.min(40, thresholdUpperBound);
    const preferredThreshold =
        stored?.sectionKey === selectedSection?.key
        && typeof storedThreshold === "number"
        && Number.isFinite(storedThreshold)
            ? Math.round(storedThreshold)
            : fallbackThreshold;
    return String(Math.max(1, Math.min(thresholdUpperBound, preferredThreshold)));
};

const samePushSubscription = (
    left: PushSubscriptionJSON | null,
    right: PushSubscriptionJSON
): boolean => {
    if (!left) return false;
    return left.endpoint === right.endpoint
        && left.keys?.p256dh === right.keys?.p256dh
        && left.keys?.auth === right.keys?.auth;
};

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

const upsertManagedRule = (rules: PushRule[], returnedRule: PushRule): PushRule[] => {
    const index = rules.findIndex((rule) => rule.id === returnedRule.id);
    if (index < 0) return [...rules, returnedRule];
    return rules.map((rule) => rule.id === returnedRule.id ? returnedRule : rule);
};

export default function CrowdAlertSubscriptionCard({
    facility,
    sections,
    isOpen,
    requireStandalonePwaForAlerts = false,
}: Props) {
    const idPrefix = useId().replace(/:/g, "");
    const managementHeadingId = `${idPrefix}-manage-alerts-heading`;
    const managementErrorId = `${idPrefix}-manage-alerts-error`;
    const availabilityErrorId = `${idPrefix}-availability-error`;
    const standaloneInfoId = `${idPrefix}-standalone-info`;
    const subscribeErrorId = `${idPrefix}-subscribe-error`;
    const cancelAllErrorId = `${idPrefix}-cancel-all-error`;

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
    const [pushStatus, setPushStatus] = useState<PushStatus>(() => {
        if (!isWebPushSupported()) return "unsupported";
        if (typeof Notification !== "undefined" && Notification.permission === "denied") return "blocked";
        return "idle";
    });
    const [pushSubscriptionJson, setPushSubscriptionJson] = useState<PushSubscriptionJSON | null>(null);
    const [pushErrorText, setPushErrorText] = useState<string | null>(null);
    const [alertsUnavailableText, setAlertsUnavailableText] = useState<string | null>(null);
    const [isAvailabilityChecking, setIsAvailabilityChecking] = useState(false);
    const [managedRules, setManagedRules] = useState<PushRule[]>([]);
    const [managementLoadStatus, setManagementLoadStatus] = useState<ManagementLoadStatus>("idle");
    const [managementErrorText, setManagementErrorText] = useState<string | null>(null);
    const [cancellingRuleIds, setCancellingRuleIds] = useState<Set<number>>(() => new Set());
    const [cancelRuleErrors, setCancelRuleErrors] = useState<Record<number, string>>({});
    const [isCancellingAll, setIsCancellingAll] = useState(false);
    const [cancelAllErrorText, setCancelAllErrorText] = useState<string | null>(null);
    const [successMessage, setSuccessMessage] = useState("");
    const [liveStatus, setLiveStatus] = useState<LiveStatus>("idle");
    const [managementReloadVersion, setManagementReloadVersion] = useState(0);

    const listedSubscriptionRef = useRef<PushSubscriptionJSON | null>(null);
    const currentSubscriptionRef = useRef<PushSubscriptionJSON | null>(null);
    const openGenerationRef = useRef(0);
    const submitInFlightRef = useRef(false);
    const cancellingRuleIdsRef = useRef<Set<number>>(new Set());
    const cancelAllInFlightRef = useRef(false);
    const managementLoadInFlightRef = useRef(false);
    const managementLoadRequestRef = useRef(0);
    const deferredManagementLoadRef = useRef(false);

    useEffect(() => {
        openGenerationRef.current += 1;
    }, [isOpen]);

    const requestedSectionKey = sectionKeyByFacility[facility] ?? initialSectionKey;
    const sectionKey = orderedSections.some(
        (section) => section.key === requestedSectionKey && hasUsableSummary(section)
    ) ? requestedSectionKey : initialSectionKey;

    useEffect(() => {
        setSectionKeyByFacility((previous) => (
            previous[facility] === sectionKey
                ? previous
                : {...previous, [facility]: sectionKey}
        ));
    }, [facility, sectionKey]);

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
        && !isAvailabilityChecking
        && !isCancellingAll
        && cancellingRuleIds.size === 0
        && managementLoadStatus !== "loading"
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

        const currentCount = selectedSection.summary.count;
        const base = currentCount === null
            ? `${facilityName}'s ${selectedSection.label} current occupancy is ${currentOccupancyPercent}% of observed capacity.`
            : `${facilityName}'s ${selectedSection.label} current occupancy is ${currentCount} people (${currentOccupancyPercent}% of observed capacity).`;
        const coverage = selectedSection.summary.status === "partial"
            ? ` Coverage: ${Math.round(selectedSection.summary.coverage * 100)}% of open capacity observed.`
            : "";
        if (!hasValidThresholdRange) return `${base}${coverage}\nThere is no lower threshold available yet.`;
        return `${base}${coverage}\nChoose a threshold between 1-${thresholdUpperBound}.`;
    }, [
        selectedSection,
        facilityName,
        currentOccupancyPercent,
        hasValidThresholdRange,
        thresholdUpperBound,
    ]);

    useEffect(() => {
        if (!isOpen) return;
        let active = true;
        setIsAvailabilityChecking(true);
        setAlertsUnavailableText(null);

        void getPushAvailability()
            .then((availability) => {
                if (!active) return;
                if (availability.alertsAvailable) {
                    setAlertsUnavailableText(null);
                } else if (availability.reason === "push_rules_db_unavailable") {
                    setAlertsUnavailableText("Alerts are temporarily unavailable because the data service is down.");
                } else if (availability.reason === "push_vapid_unconfigured") {
                    setAlertsUnavailableText("Alerts are temporarily unavailable while notification keys are being configured.");
                } else {
                    setAlertsUnavailableText("Alerts are temporarily unavailable right now. Please try again shortly.");
                }
            })
            .catch(() => {
                if (active) {
                    setAlertsUnavailableText("Alerts are temporarily unavailable right now. Please try again shortly.");
                }
            })
            .finally(() => {
                if (active) setIsAvailabilityChecking(false);
            });

        return () => {
            active = false;
        };
    }, [isOpen]);

    useEffect(() => {
        if (!isOpen) {
            managementLoadInFlightRef.current = false;
            deferredManagementLoadRef.current = false;
            return;
        }
        let active = true;
        const loadRequest = managementLoadRequestRef.current + 1;
        managementLoadRequestRef.current = loadRequest;
        managementLoadInFlightRef.current = true;
        setManagementLoadStatus("loading");
        setManagementErrorText(null);
        setCancelRuleErrors({});
        setCancelAllErrorText(null);
        setPushErrorText(null);
        if (!isWebPushSupported()) {
            setPushStatus("unsupported");
        } else if (typeof Notification !== "undefined" && Notification.permission === "denied") {
            setPushStatus("blocked");
        } else {
            setPushStatus("idle");
        }

        if (
            submitInFlightRef.current
            || cancelAllInFlightRef.current
            || cancellingRuleIdsRef.current.size > 0
        ) {
            deferredManagementLoadRef.current = true;
            return () => {
                active = false;
                if (managementLoadRequestRef.current === loadRequest) {
                    managementLoadInFlightRef.current = false;
                }
            };
        }
        deferredManagementLoadRef.current = false;

        const failLookup = () => {
            listedSubscriptionRef.current = null;
            currentSubscriptionRef.current = null;
            setPushSubscriptionJson(null);
            setManagedRules([]);
            setManagementLoadStatus("error");
            setManagementErrorText(LOOKUP_ERROR_TEXT);
        };

        const loadManagement = async () => {
            let existing: PushSubscription | null;
            try {
                existing = await getExistingPushSubscription();
            } catch {
                if (active) failLookup();
                return;
            }
            if (!active) return;
            if (!existing) {
                listedSubscriptionRef.current = null;
                currentSubscriptionRef.current = null;
                setPushSubscriptionJson(null);
                setManagedRules([]);
                setManagementLoadStatus("no-subscription");
                return;
            }

            let subscriptionJson: PushSubscriptionJSON;
            try {
                subscriptionJson = existing.toJSON();
            } catch {
                if (active) failLookup();
                return;
            }

            const isListedOwner = samePushSubscription(listedSubscriptionRef.current, subscriptionJson);
            currentSubscriptionRef.current = subscriptionJson;
            setPushSubscriptionJson(subscriptionJson);
            if (!isListedOwner) {
                listedSubscriptionRef.current = null;
                setManagedRules([]);
            }

            try {
                const rules = await listPushRules(subscriptionJson);
                if (!active) return;
                listedSubscriptionRef.current = subscriptionJson;
                currentSubscriptionRef.current = subscriptionJson;
                setPushSubscriptionJson(subscriptionJson);
                setManagedRules(rules);
                setManagementLoadStatus("success");
                setManagementErrorText(null);
                if (typeof Notification === "undefined" || Notification.permission !== "denied") {
                    setPushStatus("ready");
                }
            } catch {
                if (!active) return;
                setManagementLoadStatus("error");
                setManagementErrorText(LIST_ERROR_TEXT);
            }
        };

        void loadManagement().finally(() => {
            if (active && managementLoadRequestRef.current === loadRequest) {
                managementLoadInFlightRef.current = false;
            }
        });
        return () => {
            active = false;
            if (managementLoadRequestRef.current === loadRequest) {
                managementLoadInFlightRef.current = false;
            }
        };
    }, [isOpen, managementReloadVersion]);

    const requestDeferredManagementLoad = () => {
        if (!deferredManagementLoadRef.current) return;
        deferredManagementLoadRef.current = false;
        setManagementReloadVersion((previous) => previous + 1);
    };

    const resetSubscribeError = () => {
        setPushErrorText(null);
        if (pushStatus === "error") setPushStatus("idle");
    };

    const handleSubscribe = async () => {
        if (
            !canSubscribe
            || !selectedSection
            || submitInFlightRef.current
            || cancelAllInFlightRef.current
            || cancellingRuleIdsRef.current.size > 0
            || managementLoadInFlightRef.current
        ) return;
        submitInFlightRef.current = true;
        const operationGeneration = openGenerationRef.current;
        const requiresAuthoritativeList = managementLoadStatus === "error";
        const normalizedThreshold = Math.max(1, Math.min(thresholdUpperBound, Math.round(parsedThreshold)));
        setPushErrorText(null);
        setSuccessMessage("");
        setLiveStatus("idle");
        setIsSubmitting(true);

        try {
            let subscriptionJson = pushSubscriptionJson;
            if (!subscriptionJson) {
                const subscription = await ensurePushSubscription();
                subscriptionJson = subscription.toJSON();
                if (openGenerationRef.current === operationGeneration) {
                    currentSubscriptionRef.current = subscriptionJson;
                    setPushSubscriptionJson(subscriptionJson);
                }
            }

            const result = await subscribePushRule({
                subscription: subscriptionJson,
                facilityId: facility,
                sectionKey: selectedSection.key,
                threshold: normalizedThreshold,
            });

            const isCurrentOperation = () => (
                openGenerationRef.current === operationGeneration
                && samePushSubscription(currentSubscriptionRef.current, subscriptionJson)
            );
            if (isCurrentOperation()) {
                const stored = readStoredSubscriptions();
                stored[facility] = {
                    sectionKey: selectedSection.key,
                    threshold: normalizedThreshold,
                };
                writeStoredSubscriptions(stored);
                listedSubscriptionRef.current = subscriptionJson;
                setManagedRules((previous) => upsertManagedRule(previous, result.rule));
                setPushStatus("ready");
                setThresholdOverrides((previous) => ({
                    ...previous,
                    [thresholdContextKey]: String(normalizedThreshold),
                }));
                setSuccessMessage(result.created
                    ? "Alert set successfully."
                    : "This alert was already active.");
                setLiveStatus("alert-created");

                if (!requiresAuthoritativeList) {
                    setManagementLoadStatus("success");
                    setManagementErrorText(null);
                } else {
                    listedSubscriptionRef.current = null;
                    setManagementLoadStatus("loading");
                    try {
                        const rules = await listPushRules(subscriptionJson);
                        if (isCurrentOperation()) {
                            listedSubscriptionRef.current = subscriptionJson;
                            setManagedRules(rules);
                            setManagementLoadStatus("success");
                            setManagementErrorText(null);
                        }
                    } catch {
                        if (isCurrentOperation()) {
                            setManagementLoadStatus("error");
                            setManagementErrorText(LIST_ERROR_TEXT);
                        }
                    }
                }
            }
        } catch {
            if (openGenerationRef.current === operationGeneration) {
                const denied = typeof Notification !== "undefined" && Notification.permission === "denied";
                setPushStatus(denied ? "blocked" : "error");
                setPushErrorText(denied
                    ? "Notifications are blocked in browser settings."
                    : SUBSCRIBE_ERROR_TEXT);
            }
        } finally {
            submitInFlightRef.current = false;
            setIsSubmitting(false);
            requestDeferredManagementLoad();
        }
    };

    const handleCancelRule = async (rule: PushRule) => {
        const subscriptionJson = pushSubscriptionJson;
        if (
            !subscriptionJson
            || managementLoadInFlightRef.current
            || submitInFlightRef.current
            || cancelAllInFlightRef.current
            || cancellingRuleIdsRef.current.size > 0
        ) return;
        cancellingRuleIdsRef.current.add(rule.id);
        const operationGeneration = openGenerationRef.current;
        setCancellingRuleIds((previous) => new Set(previous).add(rule.id));
        setCancelRuleErrors((previous) => {
            const next = {...previous};
            delete next[rule.id];
            return next;
        });
        setSuccessMessage("");
        setLiveStatus("idle");

        try {
            await cancelPushRule(rule.id, subscriptionJson);
            if (
                openGenerationRef.current === operationGeneration
                && samePushSubscription(currentSubscriptionRef.current, subscriptionJson)
            ) {
                removeMatchingStoredDefaults([rule]);
                setManagedRules((previous) => previous.filter((candidate) => candidate.id !== rule.id));
                setSuccessMessage("Alert cancelled.");
            }
        } catch {
            if (
                openGenerationRef.current === operationGeneration
                && samePushSubscription(currentSubscriptionRef.current, subscriptionJson)
            ) {
                setCancelRuleErrors((previous) => ({...previous, [rule.id]: CANCEL_ERROR_TEXT}));
            }
        } finally {
            cancellingRuleIdsRef.current.delete(rule.id);
            setCancellingRuleIds((previous) => {
                const next = new Set(previous);
                next.delete(rule.id);
                return next;
            });
            requestDeferredManagementLoad();
        }
    };

    const handleCancelAll = async () => {
        const subscriptionJson = pushSubscriptionJson;
        if (
            !subscriptionJson
            || managementLoadInFlightRef.current
            || cancelAllInFlightRef.current
            || cancellingRuleIdsRef.current.size > 0
            || submitInFlightRef.current
        ) return;
        cancelAllInFlightRef.current = true;
        const operationGeneration = openGenerationRef.current;
        const rulesToCancel = managedRules;
        setIsCancellingAll(true);
        setCancelAllErrorText(null);
        setSuccessMessage("");
        setLiveStatus("idle");

        try {
            await cancelAllPushRules(subscriptionJson);
            if (
                openGenerationRef.current === operationGeneration
                && samePushSubscription(currentSubscriptionRef.current, subscriptionJson)
            ) {
                removeMatchingStoredDefaults(rulesToCancel);
                setManagedRules([]);
                setManagementLoadStatus("success");
                setManagementErrorText(null);
                setSuccessMessage("All alerts cancelled.");
            }
        } catch {
            if (
                openGenerationRef.current === operationGeneration
                && samePushSubscription(currentSubscriptionRef.current, subscriptionJson)
            ) {
                setCancelAllErrorText(CANCEL_ALL_ERROR_TEXT);
            }
        } finally {
            cancelAllInFlightRef.current = false;
            setIsCancellingAll(false);
            requestDeferredManagementLoad();
        }
    };

    const creationErrorText = pushStatus === "unsupported"
        ? "Push notifications are not supported in this browser."
        : pushStatus === "blocked"
            ? "Notifications are blocked in browser settings."
            : pushStatus === "error"
                ? (pushErrorText ?? SUBSCRIBE_ERROR_TEXT)
                : pushErrorText;
    const subscribeDescriptionIds = [
        requireStandalonePwaForAlerts ? standaloneInfoId : null,
        alertsUnavailableText ? availabilityErrorId : null,
        creationErrorText ? subscribeErrorId : null,
    ].filter((value): value is string => Boolean(value)).join(" ") || undefined;
    const isCancellationSuccess = successMessage === "Alert cancelled."
        || successMessage === "All alerts cancelled.";

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
                        {section.summary.status === "partial"
                            ? ` (Coverage: ${Math.round(section.summary.coverage * 100)}%)`
                            : ""}
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
                            ? "Live occupancy unavailable for alert thresholds."
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
                    onClick={() => void handleSubscribe()}
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
                                const sectionLabel = resolveRuleSectionLabel(rule, facility, orderedSections);
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
                                                onClick={() => void handleCancelRule(rule)}
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
                                onClick={() => void handleCancelAll()}
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

            {successMessage && (
                <Typography
                    variant="caption"
                    color="success.main"
                    role={isCancellationSuccess ? "status" : undefined}
                    aria-live={isCancellationSuccess ? "polite" : undefined}
                >
                    {successMessage}
                </Typography>
            )}
            <LiveStatusAnnouncer status={liveStatus}/>
        </Stack>
    );
}
