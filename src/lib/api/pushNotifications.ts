import {env} from "../config/env";

export interface PushRule {
    id: number;
    facilityId: 1186 | 1656;
    sectionKey: string;
    threshold: number;
    createdAt: string;
    expiresAt: string;
    status: "pending";
}

export interface SubscribePushRulePayload {
    subscription: PushSubscriptionJSON;
    facilityId: 1186 | 1656;
    sectionKey: string;
    threshold: number;
    ttlSeconds?: number;
}

export interface PushAvailabilityPayload {
    apiAvailable: boolean;
    dbAvailable: boolean;
    alertsAvailable: boolean;
    reason: string | null;
}

const PUSH_API_BASE_URL = env.pushApiBaseUrl;
const INVALID_SUBSCRIPTION_MESSAGE = "Push subscription unavailable.";
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

const resolveApiUrl = (path: string): string => {
    if (!PUSH_API_BASE_URL) return path;
    if (path.startsWith("/")) return `${PUSH_API_BASE_URL}${path}`;
    return `${PUSH_API_BASE_URL}/${path}`;
};

const isRecord = (value: unknown): value is Record<string, unknown> => (
    typeof value === "object" && value !== null && !Array.isArray(value)
);

const hasExactKeys = (value: unknown, expectedKeys: readonly string[]): value is Record<string, unknown> => {
    if (!isRecord(value)) return false;
    const keys = Object.keys(value);
    return keys.length === expectedKeys.length
        && expectedKeys.every((key) => Object.prototype.hasOwnProperty.call(value, key));
};

const canonicalSectionKey = (value: string): string => (
    value.trim().toLowerCase().replace(/\s+/g, " ")
);

const isCanonicalSectionKey = (value: unknown): value is string => (
    typeof value === "string"
    && value.length >= 1
    && value.length <= MAX_SECTION_KEY_LENGTH
    && canonicalSectionKey(value) === value
);

const daysInMonth = (year: number, month: number): number => {
    if (month === 2) {
        const leapYear = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
        return leapYear ? 29 : 28;
    }
    return [4, 6, 9, 11].includes(month) ? 30 : 31;
};

const parseExplicitZoneTimestamp = (value: unknown): number | null => {
    if (typeof value !== "string") return null;
    const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-](\d{2}):(\d{2}))$/.exec(value);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const hour = Number(match[4]);
    const minute = Number(match[5]);
    const second = Number(match[6]);
    const offsetHour = match[7] === undefined ? 0 : Number(match[7]);
    const offsetMinute = match[8] === undefined ? 0 : Number(match[8]);
    if (
        year < 1
        || month < 1
        || month > 12
        || day < 1
        || day > daysInMonth(year, month)
        || hour > 23
        || minute > 59
        || second > 59
        || offsetHour > 23
        || offsetMinute > 59
    ) {
        return null;
    }

    const timestamp = Date.parse(value);
    return Number.isFinite(timestamp) ? timestamp : null;
};

const parsePushRule = (value: unknown): PushRule => {
    if (!hasExactKeys(value, [
        "id",
        "facilityId",
        "sectionKey",
        "threshold",
        "createdAt",
        "expiresAt",
        "status",
    ])) {
        throw new Error("invalid push rule");
    }

    const id = value.id;
    const facilityId = value.facilityId;
    const threshold = value.threshold;
    const createdAt = value.createdAt;
    const expiresAt = value.expiresAt;
    const createdTimestamp = parseExplicitZoneTimestamp(createdAt);
    const expiresTimestamp = parseExplicitZoneTimestamp(expiresAt);
    if (
        !Number.isSafeInteger(id)
        || (id as number) <= 0
        || (facilityId !== 1186 && facilityId !== 1656)
        || !isCanonicalSectionKey(value.sectionKey)
        || !Number.isInteger(threshold)
        || (threshold as number) < 1
        || (threshold as number) > 100
        || createdTimestamp === null
        || expiresTimestamp === null
        || expiresTimestamp <= createdTimestamp
        || value.status !== "pending"
    ) {
        throw new Error("invalid push rule");
    }

    return {
        id: id as number,
        facilityId,
        sectionKey: value.sectionKey,
        threshold: threshold as number,
        createdAt: createdAt as string,
        expiresAt: expiresAt as string,
        status: "pending",
    };
};

const parseSubscribeResponse = (value: unknown): {created: boolean; rule: PushRule} => {
    if (
        !hasExactKeys(value, ["status", "created", "rule"])
        || value.status !== "ok"
        || typeof value.created !== "boolean"
    ) {
        throw new Error("invalid subscribe response");
    }
    return {created: value.created, rule: parsePushRule(value.rule)};
};

const parseListResponse = (value: unknown): PushRule[] => {
    if (
        !hasExactKeys(value, ["status", "rules"])
        || value.status !== "ok"
        || !Array.isArray(value.rules)
    ) {
        throw new Error("invalid list response");
    }

    const rules = value.rules.map((rule) => parsePushRule(rule));
    const ids = new Set<number>();
    for (const rule of rules) {
        if (ids.has(rule.id)) throw new Error("duplicate push rule");
        ids.add(rule.id);
    }
    return rules;
};

const parseCancelResponse = (value: unknown): void => {
    if (
        !hasExactKeys(value, ["status", "cancelled"])
        || value.status !== "ok"
        || value.cancelled !== 1
    ) {
        throw new Error("invalid cancel response");
    }
};

const parseCancelAllResponse = (value: unknown): number => {
    if (
        !hasExactKeys(value, ["status", "cancelled"])
        || value.status !== "ok"
        || !Number.isSafeInteger(value.cancelled)
        || (value.cancelled as number) < 0
    ) {
        throw new Error("invalid cancel-all response");
    }
    return value.cancelled as number;
};

const parseAvailabilityResponse = (value: unknown): PushAvailabilityPayload => {
    if (
        !hasExactKeys(value, [
            "apiAvailable",
            "dbAvailable",
            "alertsAvailable",
            "reason",
            "storeBackend",
        ])
        || value.apiAvailable !== true
        || typeof value.dbAvailable !== "boolean"
        || typeof value.alertsAvailable !== "boolean"
        || value.storeBackend !== "db"
    ) {
        throw new Error("invalid push availability");
    }

    const isDatabaseUnavailable = (
        value.dbAvailable === false
        && value.alertsAvailable === false
        && value.reason === "push_rules_db_unavailable"
    );
    const isConfigurationUnavailable = (
        value.dbAvailable === true
        && value.alertsAvailable === false
        && (
            value.reason === "push_vapid_unconfigured"
            || value.reason === "push_identity_unconfigured"
        )
    );
    const isAvailable = (
        value.dbAvailable === true
        && value.alertsAvailable === true
        && value.reason === null
    );
    if (!isDatabaseUnavailable && !isConfigurationUnavailable && !isAvailable) {
        throw new Error("invalid push availability");
    }

    return {
        apiAvailable: true,
        dbAvailable: value.dbAvailable,
        alertsAvailable: value.alertsAvailable,
        reason: value.reason as string | null,
    };
};

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

const fetchAndParse = async <Result>(
    path: string,
    init: RequestInit | undefined,
    errorMessage: string,
    parse: (value: unknown) => Result
): Promise<Result> => {
    try {
        const response = await fetch(resolveApiUrl(path), init);
        if (!response.ok) throw new Error(errorMessage);
        const value: unknown = await response.json();
        return parse(value);
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
    const response = await fetch(resolveApiUrl("/api/push/public-key"));
    if (!response.ok) {
        throw new Error("Push public key unavailable");
    }

    const payload = await response.json() as {publicKey?: string};
    if (!payload?.publicKey) {
        throw new Error("Push public key missing");
    }

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

    return fetchAndParse(
        "/api/push/subscribe",
        {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(requestPayload),
        },
        SUBSCRIBE_ERROR_MESSAGE,
        parseSubscribeResponse
    );
};

export const listPushRules = async (
    subscription: PushSubscriptionJSON
): Promise<PushRule[]> => {
    const canonicalSubscription = canonicalizeSubscription(subscription);
    return fetchAndParse(
        "/api/push/rules/list",
        {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({subscription: canonicalSubscription}),
        },
        LIST_ERROR_MESSAGE,
        parseListResponse
    );
};

export const cancelPushRule = async (
    id: number,
    subscription: PushSubscriptionJSON
): Promise<void> => {
    if (!Number.isSafeInteger(id) || id <= 0) {
        throw new Error(CANCEL_ERROR_MESSAGE);
    }
    const canonicalSubscription = canonicalizeSubscription(subscription);
    return fetchAndParse(
        `/api/push/rules/${id}`,
        {
            method: "DELETE",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({subscription: canonicalSubscription}),
        },
        CANCEL_ERROR_MESSAGE,
        parseCancelResponse
    );
};

export const cancelAllPushRules = async (
    subscription: PushSubscriptionJSON
): Promise<number> => {
    const canonicalSubscription = canonicalizeSubscription(subscription);
    return fetchAndParse(
        "/api/push/rules/cancel-all",
        {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({subscription: canonicalSubscription}),
        },
        CANCEL_ALL_ERROR_MESSAGE,
        parseCancelAllResponse
    );
};

export const getPushAvailability = async (): Promise<PushAvailabilityPayload> => (
    fetchAndParse(
        "/api/push/availability",
        undefined,
        AVAILABILITY_ERROR_MESSAGE,
        parseAvailabilityResponse
    )
);
