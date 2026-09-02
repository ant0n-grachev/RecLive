import {describe, expect, it} from "vitest";

import {parsePublicEnv} from "./env";

describe("parsePublicEnv", () => {
    it("normalizes the single API base and site URL", () => {
        expect(parsePublicEnv({
            VITE_API_BASE_URL: "https://api.example.test/",
            VITE_SITE_URL: "https://reclive.example.test/",
        }, true)).toEqual({
            apiBaseUrl: "https://api.example.test",
            siteUrl: "https://reclive.example.test",
            isDev: false,
        });
    });

    it("uses explicit empty development defaults", () => {
        expect(parsePublicEnv({}, false)).toEqual({
            apiBaseUrl: "",
            siteUrl: "",
            isDev: true,
        });
    });

    it.each(["VITE_API_BASE_URL", "VITE_SITE_URL"])(
        "requires %s in production",
        (name) => {
            const values = {
                VITE_API_BASE_URL: "https://api.example.test",
                VITE_SITE_URL: "https://reclive.example.test",
                [name]: "",
            };

            expect(() => parsePublicEnv(values, true)).toThrow(name);
        },
    );

    it.each([
        "change_me",
        "https://example.test/?AccountAPIKey=YOUR_ACCOUNT_API_KEY",
    ])("rejects production placeholder without disclosing it", (value) => {
        let error: unknown;
        try {
            parsePublicEnv({
                VITE_API_BASE_URL: value,
                VITE_SITE_URL: "https://site.test",
            }, true);
        } catch (cause) {
            error = cause;
        }

        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toContain("VITE_API_BASE_URL");
        expect((error as Error).message).not.toContain(value);
    });

    it.each([
        "javascript:alert(1)",
        "https://user:password@api.example.test",
        "https://api.example.test/?token=hidden",
        "https://api.example.test/#hidden",
    ])("rejects an unsafe public URL without disclosing it", (value) => {
        let error: unknown;
        try {
            parsePublicEnv({
                VITE_API_BASE_URL: value,
                VITE_SITE_URL: "https://site.test",
            }, true);
        } catch (cause) {
            error = cause;
        }

        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toContain("VITE_API_BASE_URL");
        expect((error as Error).message).not.toContain(value);
    });
});
