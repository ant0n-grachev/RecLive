import {describe, expect, it, vi} from "vitest";
import {openOrFocusSameOrigin} from "./notificationTarget";

describe("openOrFocusSameOrigin", () => {
    const origin = "https://dashboard.example";

    it("focuses an existing same-origin window and never opens a cross-origin target", async () => {
        const focus = vi.fn(async () => undefined);
        const navigate = vi.fn(async () => undefined);
        const openWindow = vi.fn(async () => undefined);

        const result = await openOrFocusSameOrigin("https://evil.example/x", {
            origin,
            matchAll: vi.fn(async () => [{url: `${origin}/old`, navigate, focus}]),
            openWindow,
        });

        expect(result).toBe("focused");
        expect(navigate).toHaveBeenCalledWith(`${origin}/`);
        expect(focus).toHaveBeenCalledOnce();
        expect(openWindow).not.toHaveBeenCalled();
    });

    it("opens the normalized path only when no same-origin client exists", async () => {
        const openWindow = vi.fn(async () => undefined);

        const result = await openOrFocusSameOrigin("/nick?floor=1", {
            origin,
            matchAll: vi.fn(async () => [{
                url: "https://other.example/",
                navigate: vi.fn(async () => undefined),
                focus: vi.fn(async () => undefined),
            }]),
            openWindow,
        });

        expect(result).toBe("opened");
        expect(openWindow).toHaveBeenCalledWith(`${origin}/nick?floor=1`);
    });

    it("falls back to the same-origin root for backslash targets", async () => {
        const openWindow = vi.fn(async () => undefined);

        await openOrFocusSameOrigin("/\\evil.example/x", {
            origin,
            matchAll: vi.fn(async () => []),
            openWindow,
        });

        expect(openWindow).toHaveBeenCalledWith(`${origin}/`);
    });

    it("still focuses an existing client when navigation is rejected", async () => {
        const focus = vi.fn(async () => undefined);
        const openWindow = vi.fn(async () => undefined);

        const result = await openOrFocusSameOrigin("/nick", {
            origin,
            matchAll: vi.fn(async () => [{
                url: `${origin}/old`,
                navigate: vi.fn(async () => Promise.reject(new Error("navigation denied"))),
                focus,
            }]),
            openWindow,
        });

        expect(result).toBe("focused");
        expect(focus).toHaveBeenCalledOnce();
        expect(openWindow).not.toHaveBeenCalled();
    });
});
