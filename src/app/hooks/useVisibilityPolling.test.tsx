import {act, renderHook} from "@testing-library/react";
import {afterEach, describe, expect, it, vi} from "vitest";
import {useVisibilityPolling} from "./useVisibilityPolling";

const setVisibility = (visibilityState: DocumentVisibilityState): void => {
    Object.defineProperty(document, "visibilityState", {
        configurable: true,
        value: visibilityState,
    });
};

describe("useVisibilityPolling", () => {
    afterEach(() => {
        vi.useRealTimers();
        setVisibility("visible");
    });

    it.each([
        ["live", 90_000],
        ["forecast", 15 * 60_000],
        ["schedule", 4 * 60 * 60_000],
    ])("refreshes the %s domain on its independent cadence", async (_domain, intervalMs) => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-09-01T12:00:00Z");
        setVisibility("visible");
        const onRefresh = vi.fn();

        renderHook(() => useVisibilityPolling({
            intervalMs,
            onRefresh,
            enabled: true,
        }));

        await act(async () => {
            await vi.advanceTimersByTimeAsync(intervalMs - 1);
        });
        expect(onRefresh).not.toHaveBeenCalled();

        await act(async () => {
            await vi.advanceTimersByTimeAsync(1);
        });
        expect(onRefresh).toHaveBeenCalledTimes(1);
    });

    it("pauses hidden probes without advancing the due clock", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-09-01T12:00:00Z");
        setVisibility("hidden");
        const onRefresh = vi.fn();

        renderHook(() => useVisibilityPolling({
            intervalMs: 90_000,
            onRefresh,
            enabled: true,
        }));

        await act(async () => {
            await vi.advanceTimersByTimeAsync(180_000);
        });
        expect(onRefresh).not.toHaveBeenCalled();

        act(() => {
            setVisibility("visible");
            document.dispatchEvent(new Event("visibilitychange"));
        });
        expect(onRefresh).not.toHaveBeenCalled();

        await act(async () => {
            await vi.advanceTimersByTimeAsync(29_999);
        });
        expect(onRefresh).not.toHaveBeenCalled();

        await act(async () => {
            await vi.advanceTimersByTimeAsync(1);
        });
        expect(onRefresh).toHaveBeenCalledTimes(1);
    });

    it("refreshes live data immediately on visibility restore and resets its clock", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-09-01T12:00:00Z");
        setVisibility("hidden");
        const onRefresh = vi.fn();

        renderHook(() => useVisibilityPolling({
            intervalMs: 90_000,
            onRefresh,
            enabled: true,
            refreshOnVisible: true,
        }));

        await act(async () => {
            await vi.advanceTimersByTimeAsync(180_000);
        });
        act(() => {
            setVisibility("visible");
            document.dispatchEvent(new Event("visibilitychange"));
        });
        expect(onRefresh).toHaveBeenCalledTimes(1);

        await act(async () => {
            await vi.advanceTimersByTimeAsync(89_999);
        });
        expect(onRefresh).toHaveBeenCalledTimes(1);

        await act(async () => {
            await vi.advanceTimersByTimeAsync(1);
        });
        expect(onRefresh).toHaveBeenCalledTimes(2);
    });

    it("does not poll while disabled and starts a fresh clock when enabled", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-09-01T12:00:00Z");
        setVisibility("visible");
        const onRefresh = vi.fn();

        const {rerender} = renderHook(
            ({enabled}) => useVisibilityPolling({
                intervalMs: 90_000,
                onRefresh,
                enabled,
            }),
            {initialProps: {enabled: false}},
        );

        await act(async () => {
            await vi.advanceTimersByTimeAsync(180_000);
        });
        expect(onRefresh).not.toHaveBeenCalled();

        rerender({enabled: true});
        await act(async () => {
            await vi.advanceTimersByTimeAsync(89_999);
        });
        expect(onRefresh).not.toHaveBeenCalled();

        await act(async () => {
            await vi.advanceTimersByTimeAsync(1);
        });
        expect(onRefresh).toHaveBeenCalledTimes(1);
    });

    it("uses the latest callback without resetting the active due clock", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-09-01T12:00:00Z");
        setVisibility("visible");
        const initialRefresh = vi.fn();
        const latestRefresh = vi.fn();

        const {rerender} = renderHook(
            ({onRefresh}) => useVisibilityPolling({
                intervalMs: 90_000,
                onRefresh,
                enabled: true,
            }),
            {initialProps: {onRefresh: initialRefresh}},
        );

        await act(async () => {
            await vi.advanceTimersByTimeAsync(60_000);
        });
        rerender({onRefresh: latestRefresh});

        await act(async () => {
            await vi.advanceTimersByTimeAsync(29_999);
        });
        expect(initialRefresh).not.toHaveBeenCalled();
        expect(latestRefresh).not.toHaveBeenCalled();

        await act(async () => {
            await vi.advanceTimersByTimeAsync(1);
        });
        expect(initialRefresh).not.toHaveBeenCalled();
        expect(latestRefresh).toHaveBeenCalledTimes(1);
    });
});
