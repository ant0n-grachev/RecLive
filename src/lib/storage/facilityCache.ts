import type {FacilityId, FacilityPayload} from "../types/facility";
import {facilityCacheSchema, type FacilityCache} from "../api/schemas";

export const CACHE_KEY = "reclive:facilityCache";
export const CACHE_VERSION = 3;
export const CACHE_MAX_AGE_MS = 24 * 60 * 60 * 1000;

export type CacheEntry = FacilityCache;

type CacheMap = Record<string, CacheEntry>;
type UntrustedCacheMap = Record<string, unknown>;

const hasWindow = () => typeof window !== "undefined";

const readCache = (): UntrustedCacheMap => {
    if (!hasWindow()) return {};

    try {
        const raw = window.localStorage.getItem(CACHE_KEY);
        if (!raw) return {};
        const parsed: unknown = JSON.parse(raw);
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
        return parsed as UntrustedCacheMap;
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

const hasUpstreamProvenance = (entry: CacheEntry): boolean => (
    entry.payload.liveDataSource === "facility_api"
    || entry.payload.liveDataSource === "fallback_api"
);

const validatedEntry = (
    key: string,
    value: unknown,
    now: number,
): CacheEntry | null => {
    const parsed = facilityCacheSchema.safeParse(value);
    if (!parsed.success) return null;

    const entry = parsed.data;
    if (
        key !== String(entry.payload.facilityId)
        || !hasUpstreamProvenance(entry)
        || entry.cachedAt > now
        || now - entry.cachedAt > CACHE_MAX_AGE_MS
    ) {
        return null;
    }
    return entry;
};

const pruneStaleEntries = (
    map: UntrustedCacheMap,
    now = Date.now(),
): {map: CacheMap; changed: boolean} => {
    let changed = false;
    const freshMap: CacheMap = {};

    for (const [facilityId, value] of Object.entries(map)) {
        const entry = validatedEntry(facilityId, value, now);
        if (!entry) {
            changed = true;
            continue;
        }
        freshMap[facilityId] = entry;
    }

    return {map: freshMap, changed};
};

export const getFacilityCache = (facilityId: FacilityId): CacheEntry | null => {
    const now = Date.now();
    const {map, changed} = pruneStaleEntries(readCache(), now);
    if (changed) {
        writeCache(map);
    }

    return map[String(facilityId)] ?? null;
};

export const setFacilityCache = (
    facilityId: FacilityId,
    payload: FacilityPayload
): void => {
    if (!hasWindow()) return;
    const now = Date.now();
    const key = String(facilityId);
    const candidate = validatedEntry(key, {
        version: CACHE_VERSION,
        cachedAt: now,
        payload,
    }, now);
    if (!candidate) return;

    const {map: prunedMap} = pruneStaleEntries(readCache(), now);
    prunedMap[key] = candidate;
    writeCache(prunedMap);
};
