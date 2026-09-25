import {useEffect, useEffectEvent} from "react";

interface UseVisibilityPollingArgs {
    intervalMs: number;
    onRefresh: () => void;
    enabled: boolean;
    refreshOnVisible?: boolean;
}

const MAX_PROBE_INTERVAL_MS = 30_000;

export function useVisibilityPolling({
    intervalMs,
    onRefresh,
    enabled,
    refreshOnVisible = false,
}: UseVisibilityPollingArgs): void {
    const refresh = useEffectEvent(onRefresh);

    useEffect(() => {
        if (!enabled) return;

        let lastRunAt = Date.now();
        const tick = () => {
            const now = Date.now();
            if (
                document.visibilityState !== "visible"
                || now - lastRunAt < intervalMs
            ) {
                return;
            }

            lastRunAt = now;
            refresh();
        };
        const timer = window.setInterval(
            tick,
            Math.min(intervalMs, MAX_PROBE_INTERVAL_MS),
        );
        const handleVisibilityChange = () => {
            if (document.visibilityState !== "visible" || !refreshOnVisible) {
                return;
            }

            lastRunAt = Date.now();
            refresh();
        };

        document.addEventListener("visibilitychange", handleVisibilityChange);
        return () => {
            window.clearInterval(timer);
            document.removeEventListener("visibilitychange", handleVisibilityChange);
        };
    }, [enabled, intervalMs, refreshOnVisible]);
}
