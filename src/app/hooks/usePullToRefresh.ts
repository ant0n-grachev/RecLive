import {type TouchEvent, useRef, useState} from "react";

const PULL_REFRESH_TRIGGER_PX = 72;
const PULL_REFRESH_MAX_PX = 120;
const PULL_REFRESH_RESISTANCE = 0.45;
const DISABLE_PULL_REFRESH_SELECTOR = "[data-disable-pull-refresh='true']";

interface UsePullToRefreshArgs {
    enabled: boolean;
    blocked: boolean;
    isLoading: boolean;
    hasData: boolean;
    onRefresh: () => void;
}

export interface PullToRefreshState {
    pullDistance: number;
    isPulling: boolean;
    isReadyToRefresh: boolean;
    showIndicator: boolean;
    resetPullGesture: () => void;
    handleTouchStart: (event: TouchEvent<HTMLDivElement>) => void;
    handleTouchMove: (event: TouchEvent<HTMLDivElement>) => void;
    handleTouchEnd: () => void;
}

export const usePullToRefresh = ({
    enabled,
    blocked,
    isLoading,
    hasData,
    onRefresh,
}: UsePullToRefreshArgs): PullToRefreshState => {
    const [pullDistance, setPullDistance] = useState(0);
    const pullStartYRef = useRef<number | null>(null);

    const startedInExcludedZone = (target: EventTarget | null): boolean => {
        if (typeof Element === "undefined") return false;
        if (!(target instanceof Element)) return false;
        return Boolean(target.closest(DISABLE_PULL_REFRESH_SELECTOR));
    };

    const resetPullGesture = () => {
        pullStartYRef.current = null;
        setPullDistance(0);
    };

    const handleTouchStart = (event: TouchEvent<HTMLDivElement>) => {
        if (!enabled || blocked) return;
        if (typeof window === "undefined") return;
        if (startedInExcludedZone(event.target)) {
            pullStartYRef.current = null;
            return;
        }
        if (isLoading || window.scrollY > 0 || event.touches.length !== 1) {
            pullStartYRef.current = null;
            return;
        }
        pullStartYRef.current = event.touches[0].clientY;
    };

    const handleTouchMove = (event: TouchEvent<HTMLDivElement>) => {
        if (!enabled || blocked) return;

        const startY = pullStartYRef.current;
        if (startY === null || event.touches.length !== 1) return;
        if (typeof window !== "undefined" && window.scrollY > 0) {
            resetPullGesture();
            return;
        }

        const delta = event.touches[0].clientY - startY;
        if (delta <= 0) {
            setPullDistance(0);
            return;
        }

        const nextDistance = Math.min(
            PULL_REFRESH_MAX_PX,
            Math.round(delta * PULL_REFRESH_RESISTANCE)
        );
        setPullDistance(nextDistance);

        if (nextDistance > 0) {
            event.preventDefault();
        }
    };

    const handleTouchEnd = () => {
        if (!enabled) return;
        if (blocked) {
            resetPullGesture();
            return;
        }
        if (pullDistance >= PULL_REFRESH_TRIGGER_PX) {
            onRefresh();
        }
        resetPullGesture();
    };

    return {
        pullDistance,
        isPulling: enabled && !blocked && pullDistance > 0,
        isReadyToRefresh: enabled && !blocked && pullDistance >= PULL_REFRESH_TRIGGER_PX,
        showIndicator: enabled && !blocked && isLoading && hasData,
        resetPullGesture,
        handleTouchStart,
        handleTouchMove,
        handleTouchEnd,
    };
};
