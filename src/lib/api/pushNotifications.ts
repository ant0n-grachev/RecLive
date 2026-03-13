import {env} from "../config/env";

export interface PushRulePayload {
    subscription: PushSubscriptionJSON;
    facilityId: number;
    sectionKey: string;
    threshold: number;
}

export interface PushRuleExistsPayload {
    subscription: PushSubscriptionJSON;
    facilityId: number;
    sectionKey: string;
    threshold: number;
}

export interface PushAvailabilityPayload {
    apiAvailable: boolean;
    dbAvailable: boolean;
    alertsAvailable: boolean;
    reason: string | null;
}

const PUSH_API_BASE_URL = env.pushApiBaseUrl;

const resolveApiUrl = (path: string): string => {
    if (!PUSH_API_BASE_URL) return path;
    if (path.startsWith("/")) return `${PUSH_API_BASE_URL}${path}`;
    return `${PUSH_API_BASE_URL}/${path}`;
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

export const upsertPushRule = async (payload: PushRulePayload): Promise<void> => {
    const response = await fetch(resolveApiUrl("/api/push/subscribe"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error("Failed to save push rule");
    }
};

export const hasMatchingPushRule = async (payload: PushRuleExistsPayload): Promise<boolean> => {
    const response = await fetch(resolveApiUrl("/api/push/rules/exists"), {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
    });

    if (!response.ok) {
        throw new Error("Failed to check existing push rule");
    }

    const result = await response.json() as {exists?: boolean};
    return Boolean(result?.exists);
};

export const getPushAvailability = async (): Promise<PushAvailabilityPayload> => {
    const response = await fetch(resolveApiUrl("/api/push/availability"));
    if (!response.ok) {
        throw new Error("Push availability endpoint unavailable");
    }

    const payload = await response.json() as Partial<PushAvailabilityPayload>;
    return {
        apiAvailable: payload.apiAvailable !== false,
        dbAvailable: payload.dbAvailable !== false,
        alertsAvailable: payload.alertsAvailable !== false,
        reason: payload.reason ?? null,
    };
};
