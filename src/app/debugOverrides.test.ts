import {describe, expect, it, vi} from "vitest";
import {
    debugControlsEnabled,
    loadDebugOverrides,
} from "./debugOverrides";

describe("debug overrides", () => {
    it("does not read query or persisted overrides while controls are disabled", () => {
        const getItem = vi.fn(() => "true");
        expect(loadDebugOverrides(false, {
            search: "?debugNow=2026-08-31T09:00&overrideClosure=1",
            storage: {getItem},
        })).toEqual({closureOverrideEnabled: false, debugNowValue: null});
        expect(getItem).not.toHaveBeenCalled();
    });

    it("requires the explicit local-debug build mode outside development", () => {
        const production = {DEV: false, MODE: "production"};
        expect(debugControlsEnabled(production, "localhost")).toBe(false);
        expect(debugControlsEnabled(production, "127.0.0.1")).toBe(false);
        expect(debugControlsEnabled({DEV: false, MODE: "local-debug"}, "localhost")).toBe(true);
        expect(debugControlsEnabled({DEV: false, MODE: "local-debug"}, "recwell.wisc.edu")).toBe(false);
        expect(debugControlsEnabled({DEV: true, MODE: "development"}, "recwell.wisc.edu")).toBe(true);
    });

    it("reads only the current closure and debug-clock contracts when enabled", () => {
        const getItem = vi.fn((key: string) => (
            key === "reclive:closureOverride" ? "true" : "2026-08-31T08:00"
        ));
        expect(loadDebugOverrides(true, {
            search: "?debugNow=2026-08-31T09:00",
            storage: {getItem},
        })).toEqual({
            closureOverrideEnabled: true,
            debugNowValue: "2026-08-31T09:00",
        });
    });
});
