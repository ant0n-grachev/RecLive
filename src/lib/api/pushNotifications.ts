import type {ZodType} from "zod";
import {requestJson, type RequestOptions} from "./client";
import {
    pushAvailabilitySchema,
    pushCancelAllResponseSchema,
    pushCancelOneResponseSchema,
    pushPublicKeySchema,
    pushRuleListSchema,
    pushRuleResponseSchema,
    type PushAvailability,
    type PushRule as ValidatedPushRule,
} from "./schemas";

export type PushRule = ValidatedPushRule;

export interface SubscribePushRulePayload {
    subscription: PushSubscriptionJSON;
    facilityId: 1186 | 1656;
    sectionKey: string;
    threshold: number;
    ttlSeconds?: number;
}

export type PushAvailabilityPayload = PushAvailability;

const INVALID_SUBSCRIPTION_MESSAGE = "Push subscription unavailable.";
const PUBLIC_KEY_ERROR_MESSAGE = "Push public key unavailable";
const SUBSCRIBE_ERROR_MESSAGE = "Could not save this alert right now.";
const LIST_ERROR_MESSAGE = "Could not load alerts right now.";
const CANCEL_ERROR_MESSAGE = "Could not cancel this alert right now.";
const CANCEL_ALL_ERROR_MESSAGE = "Could not cancel alerts right now.";
const AVAILABILITY_ERROR_MESSAGE = "Push availability endpoint unavailable";
const MAX_SECTION_KEY_LENGTH = 80;
const MAX_RULE_TTL_SECONDS = 604_800;

interface CanonicalPushSubscription {
    endpoint: string;
    keys: {
        p256dh: string;
        auth: string;
    };
}

const isRecord = (value: unknown): value is Record<string, unknown> => (
    typeof value === "object" && value !== null && !Array.isArray(value)
);

const canonicalSectionKey = (value: string): string => (
    value.trim().toLowerCase().replace(/\s+/g, " ")
);

const isCanonicalSectionKey = (value: unknown): value is string => (
    typeof value === "string"
    && value.length >= 1
    && value.length <= MAX_SECTION_KEY_LENGTH
    && canonicalSectionKey(value) === value
);

const canonicalizeSubscription = (
    subscription: PushSubscriptionJSON
): CanonicalPushSubscription => {
    if (!isRecord(subscription) || !isRecord(subscription.keys)) {
        throw new Error(INVALID_SUBSCRIPTION_MESSAGE);
    }

    const endpoint = subscription.endpoint;
    const p256dh = subscription.keys.p256dh;
    const auth = subscription.keys.auth;
    const expirationTime = subscription.expirationTime;
    if (
        typeof endpoint !== "string"
        || endpoint.length === 0
        || endpoint.length > 2048
        || endpoint.trim() !== endpoint
        || typeof p256dh !== "string"
        || p256dh.length === 0
        || p256dh.length > 512
        || p256dh.trim() !== p256dh
        || typeof auth !== "string"
        || auth.length === 0
        || auth.length > 512
        || auth.trim() !== auth
        || (
            expirationTime !== undefined
            && expirationTime !== null
            && (
                typeof expirationTime !== "number"
                || !Number.isFinite(expirationTime)
                || expirationTime < 0
            )
        )
    ) {
        throw new Error(INVALID_SUBSCRIPTION_MESSAGE);
    }

    try {
        const parsedEndpoint = new URL(endpoint);
        if (
            parsedEndpoint.protocol !== "https:"
            || !parsedEndpoint.hostname
            || parsedEndpoint.username !== ""
            || parsedEndpoint.password !== ""
            || parsedEndpoint.hash !== ""
        ) {
            throw new Error(INVALID_SUBSCRIPTION_MESSAGE);
        }
    } catch {
        throw new Error(INVALID_SUBSCRIPTION_MESSAGE);
    }

    return {endpoint, keys: {p256dh, auth}};
};

const requestWithFixedError = async <Result>(
    path: string,
    schema: ZodType<Result>,
    options: RequestOptions,
    errorMessage: string,
): Promise<Result> => {
    try {
        return await requestJson(path, schema, options);
    } catch {
        throw new Error(errorMessage);
    }
};

const urlBase64ToUint8Array = (base64String: string): Uint8Array => {
    const padding = "=".repeat((4 - (base64String.length % 4)) % 4);
    const base64 = (base64String + padding).replace(/-/g, "+").replace(/_/g, "/");
    const rawData = window.atob(base64);
    const bytes = new Uint8Array(rawData.length);

    for (let index = 0; index < rawData.length; index += 1) {
        bytes[index] = rawData.charCodeAt(index);
    }

    return bytes;
};

export const isWebPushSupported = (): boolean => {
    if (typeof window === "undefined") return false;
    return "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;
};

const ensureActiveServiceWorker = async (): Promise<ServiceWorkerRegistration> => {
    const existing = await navigator.serviceWorker.getRegistration("/");
    const registration = existing ?? await navigator.serviceWorker.register("/sw.js");

    // Firefox can fail subscription when no active worker is ready yet.
    if (!registration.active) {
        await navigator.serviceWorker.ready;
    }

    return navigator.serviceWorker.ready;
};

const getPushPublicKey = async (): Promise<string> => {
    const payload = await requestWithFixedError(
        "/api/push/public-key",
        pushPublicKeySchema,
        {attempts: 3},
        PUBLIC_KEY_ERROR_MESSAGE
    );
    return payload.publicKey;
};

export const ensurePushSubscription = async (): Promise<PushSubscription> => {
    if (!isWebPushSupported()) {
        throw new Error("Push notifications are not supported in this browser.");
    }

    if (!window.isSecureContext) {
        throw new Error("Push requires HTTPS (or localhost) in this browser.");
    }

    const permission = await Notification.requestPermission();
    if (permission !== "granted") {
        throw new Error("Notifications permission was not granted.");
    }

    const registration = await ensureActiveServiceWorker();
    const existing = await registration.pushManager.getSubscription();
    if (existing) return existing;

    const publicKey = await getPushPublicKey();
    const applicationServerKey = urlBase64ToUint8Array(publicKey) as BufferSource;
    return registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey,
    });
};

export const getExistingPushSubscription = async (): Promise<PushSubscription | null> => {
    if (!isWebPushSupported()) return null;
    const registration = await ensureActiveServiceWorker();
    return registration.pushManager.getSubscription();
};

export const subscribePushRule = async (
    payload: SubscribePushRulePayload
): Promise<{created: boolean; rule: PushRule}> => {
    if (!isRecord(payload)) {
        throw new Error(SUBSCRIBE_ERROR_MESSAGE);
    }
    const canonicalSubscription = canonicalizeSubscription(payload.subscription);
    if (
        (payload.facilityId !== 1186 && payload.facilityId !== 1656)
        || !isCanonicalSectionKey(payload.sectionKey)
        || !Number.isInteger(payload.threshold)
        || payload.threshold < 1
        || payload.threshold > 100
        || (
            payload.ttlSeconds !== undefined
            && (
                !Number.isSafeInteger(payload.ttlSeconds)
                || payload.ttlSeconds < 1
                || payload.ttlSeconds > MAX_RULE_TTL_SECONDS
            )
        )
    ) {
        throw new Error(SUBSCRIBE_ERROR_MESSAGE);
    }

    const requestPayload: {
        subscription: CanonicalPushSubscription;
        facilityId: 1186 | 1656;
        sectionKey: string;
        threshold: number;
        ttlSeconds?: number;
    } = {
        subscription: canonicalSubscription,
        facilityId: payload.facilityId,
        sectionKey: payload.sectionKey,
        threshold: payload.threshold,
    };
    if (payload.ttlSeconds !== undefined) {
        requestPayload.ttlSeconds = payload.ttlSeconds;
    }

    const response = await requestWithFixedError(
        "/api/push/subscribe",
        pushRuleResponseSchema,
        {
            method: "POST",
            attempts: 1,
            body: requestPayload,
        },
        SUBSCRIBE_ERROR_MESSAGE
    );
    if (
        response.rule.facilityId !== requestPayload.facilityId
        || response.rule.sectionKey !== requestPayload.sectionKey
        || response.rule.threshold !== requestPayload.threshold
    ) {
        throw new Error(SUBSCRIBE_ERROR_MESSAGE);
    }
    return {created: response.created, rule: response.rule};
};

export const listPushRules = async (
    subscription: PushSubscriptionJSON
): Promise<PushRule[]> => {
    const canonicalSubscription = canonicalizeSubscription(subscription);
    const response = await requestWithFixedError(
        "/api/push/rules/list",
        pushRuleListSchema,
        {
            method: "POST",
            attempts: 3,
            body: {subscription: canonicalSubscription},
        },
        LIST_ERROR_MESSAGE
    );
    return response.rules;
};

export const cancelPushRule = async (
    id: number,
    subscription: PushSubscriptionJSON
): Promise<void> => {
    if (!Number.isSafeInteger(id) || id <= 0) {
        throw new Error(CANCEL_ERROR_MESSAGE);
    }
    const canonicalSubscription = canonicalizeSubscription(subscription);
    await requestWithFixedError(
        `/api/push/rules/${id}`,
        pushCancelOneResponseSchema,
        {
            method: "DELETE",
            attempts: 1,
            body: {subscription: canonicalSubscription},
        },
        CANCEL_ERROR_MESSAGE
    );
};

export const cancelAllPushRules = async (
    subscription: PushSubscriptionJSON
): Promise<number> => {
    const canonicalSubscription = canonicalizeSubscription(subscription);
    const response = await requestWithFixedError(
        "/api/push/rules/cancel-all",
        pushCancelAllResponseSchema,
        {
            method: "POST",
            attempts: 1,
            body: {subscription: canonicalSubscription},
        },
        CANCEL_ALL_ERROR_MESSAGE
    );
    return response.cancelled;
};

export const getPushAvailability = async (): Promise<PushAvailabilityPayload> => (
    requestWithFixedError(
        "/api/push/availability",
        pushAvailabilitySchema,
        {attempts: 3},
        AVAILABILITY_ERROR_MESSAGE
    )
);
