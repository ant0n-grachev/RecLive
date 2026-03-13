import type {FacilityId, FacilityPayload} from "../types/facility";

export const CACHE_KEY = "reclive:facilityCache";
export const CACHE_VERSION = 2;
export const CACHE_MAX_AGE_MS = 24 * 60 * 60 * 1000;

export type CacheEntry = {
    version: number;
    cachedAt: number;
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

const isFreshEntry = (entry: CacheEntry | undefined, now = Date.now()): entry is CacheEntry => (
    Boolean(entry)
    && entry?.version === CACHE_VERSION
    && Number.isFinite(entry.cachedAt)
    && now - entry.cachedAt <= CACHE_MAX_AGE_MS
);

const pruneStaleEntries = (map: CacheMap, now = Date.now()): {map: CacheMap; changed: boolean} => {
    let changed = false;
    const freshMap: CacheMap = {};

    for (const [facilityId, entry] of Object.entries(map)) {
        if (!isFreshEntry(entry, now)) {
            changed = true;
            continue;
        }
        freshMap[facilityId] = entry;
    }

    return {map: freshMap, changed};
};

export const getFacilityCache = (facilityId: FacilityId): CacheEntry | null => {
    const {map, changed} = pruneStaleEntries(readCache());
    if (changed) {
        writeCache(map);
    }

    const entry = map[String(facilityId)];
    if (!isFreshEntry(entry)) return null;
    return entry;
};

export const setFacilityCache = (
    facilityId: FacilityId,
    payload: FacilityPayload
): void => {
    if (!hasWindow()) return;
    const map = readCache();
    const {map: prunedMap} = pruneStaleEntries(map);
    prunedMap[String(facilityId)] = {
        version: CACHE_VERSION,
        cachedAt: Date.now(),
        payload,
    };
    writeCache(prunedMap);
};
