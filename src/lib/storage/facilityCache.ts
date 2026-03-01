import type {FacilityId, FacilityPayload} from "../types/facility";

export const CACHE_KEY = "reclive:facilityCache";
export const CACHE_VERSION = 1;

type CacheEntry = {
    version: number;
    payload: FacilityPayload;
};

type CacheMap = Record<string, CacheEntry>;

const hasWindow = () => typeof window !== "undefined";

const readCache = (): CacheMap => {
    if (!hasWindow()) return {};

    try {
        const raw = window.localStorage.getItem(CACHE_KEY);
        if (!raw) return {};
        const parsed = JSON.parse(raw) as CacheMap;
        return parsed ?? {};
    } catch {
        return {};
    }
};

const writeCache = (map: CacheMap) => {
    if (!hasWindow()) return;
    try {
        window.localStorage.setItem(CACHE_KEY, JSON.stringify(map));
    } catch {
        // Ignore storage write failures (private mode/quota exceeded).
    }
};

export const getFacilityCache = (facilityId: FacilityId): CacheEntry | null => {
    const map = readCache();
    const entry = map[String(facilityId)];
    if (!entry || entry.version !== CACHE_VERSION) return null;
    return entry;
};

export const setFacilityCache = (
    facilityId: FacilityId,
    payload: FacilityPayload
): void => {
    if (!hasWindow()) return;
    const map = readCache();
    map[String(facilityId)] = {
        version: CACHE_VERSION,
        payload,
    };
    writeCache(map);
};
