import {describe, expect, it} from "vitest";
import {safePushPayload} from "./pushPayload";

describe("safePushPayload", () => {
    const fallback = {
        title: "RecLive alert",
        body: "Your occupancy alert is ready.",
        url: "/",
    };

    it("uses bounded fallback content for missing or malformed data", () => {
        expect(safePushPayload(null)).toEqual(fallback);
        expect(safePushPayload({json: () => {
            throw new Error("bad json");
        }})).toEqual(fallback);
        expect(safePushPayload({json: () => "not an object"})).toEqual(fallback);
    });

    it("normalizes types, lengths, and an unsafe target", () => {
        const payload = safePushPayload({
            json: () => ({
                title: "x".repeat(81),
                body: "y".repeat(241),
                url: "https://other.example/a",
            }),
        });

        expect(payload).toEqual({
            title: "x".repeat(80),
            body: "y".repeat(240),
            url: "/",
        });
    });

    it("rejects protocol-relative and backslash targets", () => {
        expect(safePushPayload({json: () => ({url: "//other.example/a"})}).url).toBe("/");
        expect(safePushPayload({json: () => ({url: "/\\other.example/a"})}).url).toBe("/");
    });
});
