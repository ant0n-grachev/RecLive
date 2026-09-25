export interface DebugOverrideInput {
    closureOverrideEnabled: boolean;
    debugNowValue: string | null;
}

export interface DebugOverrideSources {
    search: string;
    storage: Pick<Storage, "getItem">;
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1"]);
const CLOSURE_STORAGE_KEY = "reclive:closureOverride";
const DEBUG_NOW_STORAGE_KEY = "reclive:debugNow";

export function debugControlsEnabled(
    env: Pick<ImportMetaEnv, "DEV" | "MODE">,
    hostname: string,
): boolean {
    return env.DEV || (env.MODE === "local-debug" && LOCAL_HOSTS.has(hostname));
}

export function loadDebugOverrides(
    enabled: boolean,
    sources?: DebugOverrideSources,
): DebugOverrideInput {
    if (!enabled) return {closureOverrideEnabled: false, debugNowValue: null};

    try {
        const search = sources?.search ?? window.location.search;
        const storage = sources?.storage ?? window.localStorage;
        const params = new URLSearchParams(search);
        const closureQuery = params.get("overrideClosure") ?? params.get("debugClosure");
        const storedClosure = storage.getItem(CLOSURE_STORAGE_KEY);
        const debugNowQuery = params.get("debugNow");
        return {
            closureOverrideEnabled: closureQuery === null
                ? storedClosure === "true"
                : closureQuery !== "0" && closureQuery.toLowerCase() !== "false",
            debugNowValue: debugNowQuery ?? storage.getItem(DEBUG_NOW_STORAGE_KEY),
        };
    } catch {
        return {closureOverrideEnabled: false, debugNowValue: null};
    }
}
