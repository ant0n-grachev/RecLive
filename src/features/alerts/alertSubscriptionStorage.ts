import type {FacilityId} from "../../lib/types/facility";
import type {AlertSectionOption} from "./alertTypes";

export interface StoredSubscription {
    sectionKey: string;
    threshold: number;
}

export type StoredSubscriptions = Partial<Record<FacilityId, StoredSubscription>>;

const STORAGE_KEY = "reclive:crowd-alert-subscriptions";

const isStoredSubscription = (value: unknown): value is StoredSubscription => {
    if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
    const candidate = value as Record<string, unknown>;
    return typeof candidate.sectionKey === "string"
        && candidate.sectionKey.length > 0
        && typeof candidate.threshold === "number"
        && Number.isFinite(candidate.threshold);
};

export const readStoredSubscriptions = (): StoredSubscriptions => {
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

const sanitizeStoredSubscriptions = (value: StoredSubscriptions): StoredSubscriptions => {
    const result: StoredSubscriptions = {};
    for (const facility of [1186, 1656] as const) {
        const candidate = value[facility];
        if (isStoredSubscription(candidate)) {
            result[facility] = {sectionKey: candidate.sectionKey, threshold: candidate.threshold};
        }
    }
    return result;
};

export const writeStoredSubscriptions = (value: StoredSubscriptions): void => {
    if (typeof window === "undefined") return;
    try {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sanitizeStoredSubscriptions(value)));
    } catch {
        // Storage is only a convenience default; server state remains authoritative.
    }
};

export const normalizePercentInt = (value: number): number => Math.max(0, Math.round(value));

export const hasUsableSummary = (section: AlertSectionOption): boolean => (
    (section.summary.status === "live" || section.summary.status === "partial")
    && section.summary.percent !== null
    && Number.isFinite(section.summary.percent)
);

export const getThresholdUpperBound = (section: AlertSectionOption | null): number => {
    if (!section || !hasUsableSummary(section) || section.summary.percent === null) return 0;
    return Math.min(100, Math.max(0, normalizePercentInt(section.summary.percent) - 1));
};

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

export const resolveDefaultThresholdInput = (
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
