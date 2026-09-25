import {useEffect, useRef, useState} from "react";
import {Stack, Typography} from "@mui/material";
import type {FacilityId} from "../../lib/types/facility";
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
} from "../../lib/api/pushNotifications";
import {LiveStatusAnnouncer, type LiveStatus} from "../../facilities/LiveStatusAnnouncer";
import type {AlertSectionOption} from "./alertTypes";
import {getThresholdUpperBound, hasUsableSummary, readStoredSubscriptions, writeStoredSubscriptions} from "./alertSubscriptionStorage";
import AlertRuleForm, {type AlertRuleSubmission, type SuccessfulAlertDraft} from "./AlertRuleForm";
import ActiveAlertRules, {type ManagementLoadStatus} from "./ActiveAlertRules";

interface Props {
    onClose: () => void;
    facility: FacilityId;
    sections: AlertSectionOption[];
    isOpen: boolean;
    requireStandalonePwaForAlerts?: boolean;
}

type PushStatus = "unsupported" | "idle" | "ready" | "blocked" | "error";

const LOOKUP_ERROR_TEXT = "Could not access this browser's alerts right now.";
const LIST_ERROR_TEXT = "Could not load alerts right now.";
const SUBSCRIBE_ERROR_TEXT = "Could not save this alert right now.";
const CANCEL_ERROR_TEXT = "Could not cancel this alert right now.";
const CANCEL_ALL_ERROR_TEXT = "Could not cancel alerts right now.";
const AVAILABILITY_ERROR_TEXT = "Alerts aren’t available right now. Try again.";

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

const samePushSubscription = (
    left: PushSubscriptionJSON | null,
    right: PushSubscriptionJSON
): boolean => {
    if (!left) return false;
    return left.endpoint === right.endpoint
        && left.keys?.p256dh === right.keys?.p256dh
        && left.keys?.auth === right.keys?.auth;
};

const upsertManagedRule = (rules: PushRule[], returnedRule: PushRule): PushRule[] => {
    const index = rules.findIndex((rule) => rule.id === returnedRule.id);
    if (index < 0) return [...rules, returnedRule];
    return rules.map((rule) => rule.id === returnedRule.id ? returnedRule : rule);
};

export default function AlertsDialog({
    facility,
    sections,
    isOpen,
    requireStandalonePwaForAlerts = false,
}: Props) {
    const [successfulDraft, setSuccessfulDraft] = useState<SuccessfulAlertDraft | null>(null);
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
                } else {
                    setAlertsUnavailableText(AVAILABILITY_ERROR_TEXT);
                }
            })
            .catch(() => {
                if (active) {
                    setAlertsUnavailableText(AVAILABILITY_ERROR_TEXT);
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

    const handleSubscribe = async (request: AlertRuleSubmission) => {
        const selectedSection = sections.find((section) => section.key === request.sectionKey && hasUsableSummary(section));
        const thresholdUpperBound = getThresholdUpperBound(selectedSection ?? null);
        const parsedThreshold = request.threshold;
        const canSubscribe = !creationBlocked
            && Number.isInteger(parsedThreshold)
            && parsedThreshold >= 1
            && parsedThreshold <= thresholdUpperBound;
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
                setSuccessfulDraft({facility, sectionKey: selectedSection.key, threshold: normalizedThreshold});
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
    const creationBlocked = isAvailabilityChecking
        || isCancellingAll
        || cancellingRuleIds.size > 0
        || managementLoadStatus === "loading"
        || requireStandalonePwaForAlerts
        || Boolean(alertsUnavailableText)
        || pushStatus === "unsupported"
        || pushStatus === "blocked";
    const isCancellationSuccess = successMessage === "Alert cancelled."
        || successMessage === "All alerts cancelled.";

    return (
        <Stack spacing={1.25}>
            <AlertRuleForm
                facility={facility}
                sections={sections}
                requireStandalonePwaForAlerts={requireStandalonePwaForAlerts}
                isAvailabilityChecking={isAvailabilityChecking}
                alertsUnavailableText={alertsUnavailableText}
                creationErrorText={creationErrorText}
                creationBlocked={creationBlocked}
                isSubmitting={isSubmitting}
                successfulDraft={successfulDraft}
                onSubmit={handleSubscribe}
                resetSubscribeError={resetSubscribeError}
            />
            <ActiveAlertRules
                facility={facility}
                sections={sections}
                managedRules={managedRules}
                managementLoadStatus={managementLoadStatus}
                managementErrorText={managementErrorText}
                cancellingRuleIds={cancellingRuleIds}
                cancelRuleErrors={cancelRuleErrors}
                isCancellingAll={isCancellingAll}
                cancelAllErrorText={cancelAllErrorText}
                isSubmitting={isSubmitting}
                onCancelRule={handleCancelRule}
                onCancelAll={handleCancelAll}
            />

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
