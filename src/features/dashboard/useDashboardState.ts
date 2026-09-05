import {useCallback, useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState} from "react";
import type {FacilityId} from "../../lib/types/facility";
import type {LiveStatus} from "../../facilities/LiveStatusAnnouncer";
import {getChicagoTimestampMs} from "../../shared/utils/chicagoTime";
import {useStandalonePwa} from "../../app/hooks/useStandalonePwa";
import {usePullToRefresh} from "../../app/hooks/usePullToRefresh";
import {useOnlineStatus} from "../../app/hooks/useOnlineStatus";
import {useLiveFacilityData} from "../../app/hooks/useLiveFacilityData";
import {useForecastData} from "../../app/hooks/useForecastData";
import {useFacilityHours} from "../../app/hooks/useFacilityHours";
import {useVisibilityPolling} from "../../app/hooks/useVisibilityPolling";
import {debugControlsEnabled, loadDebugOverrides} from "../../app/debugOverrides";
import {buildDashboardViewModel} from "./dashboardSelectors";
import type {DashboardState, ForecastDaySelection, UseDashboardStateArgs} from "./dashboardTypes";

const FACILITY_STORAGE_KEY = "reclive:selectedFacility";
const CLOSURE_OVERRIDE_STORAGE_KEY = "reclive:closureOverride";
const DEBUG_NOW_STORAGE_KEY = "reclive:debugNow";
const LIVE_REFRESH_INTERVAL_MS = 90 * 1000;
const FORECAST_REFRESH_INTERVAL_MS = 15 * 60 * 1000;
const SCHEDULE_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000;
const MANUAL_REFRESH_COOLDOWN_MS = 3000;
const CLOCK_TICK_MS = 30 * 1000;

declare global {
    interface Window {
        recliveShowPredictions?: () => string;
        recliveRestoreWarnings?: () => string;
        reclivePredictionOverrideStatus?: () => boolean;
        recliveOverrideClosure?: (enabled?: boolean) => string;
        recliveRestoreClosure?: () => string;
        recliveClosureOverrideStatus?: () => boolean;
        recliveSetDebugNow?: (value: string) => string;
        recliveClearDebugNow?: () => string;
        recliveDebugNowStatus?: () => string | null;
        recliveDebugDashboardState?: () => unknown;
    }
}

const getStoredFacility = (): FacilityId => {
    if (typeof window === "undefined") return 1186;
    const stored = Number(window.localStorage.getItem(FACILITY_STORAGE_KEY));
    return stored === 1656 ? 1656 : 1186;
};

const writeClosureOverrideStorage = (enabled: boolean, debugEnabled: boolean) => {
    if (!debugEnabled || typeof window === "undefined") return;
    try {
        if (enabled) {
            window.localStorage.setItem(CLOSURE_OVERRIDE_STORAGE_KEY, "true");
        } else {
            window.localStorage.removeItem(CLOSURE_OVERRIDE_STORAGE_KEY);
        }
    } catch {
        // Ignore storage write failures (private mode/quota exceeded).
    }
};

const parseDebugNowMs = (value: string | null | undefined): number | null => {
    const text = typeof value === "string" ? value.trim() : "";
    if (!text) return null;

    const parsed = getChicagoTimestampMs(text);
    return parsed === null ? null : parsed;
};

const writeDebugNowStorage = (value: string | null, debugEnabled: boolean) => {
    if (!debugEnabled || typeof window === "undefined") return;
    try {
        if (value) {
            window.localStorage.setItem(DEBUG_NOW_STORAGE_KEY, value);
        } else {
            window.localStorage.removeItem(DEBUG_NOW_STORAGE_KEY);
        }
    } catch {
        // Ignore storage write failures (private mode/quota exceeded).
    }
};

export function useDashboardState({
    initialFacility,
    onFacilityRouteChange,
    isPhoneViewport,
}: UseDashboardStateArgs): DashboardState {
    const {isStandalonePwa, isTouchCapable} = useStandalonePwa();
    const [facility, setFacility] = useState<FacilityId>(() => initialFacility ?? getStoredFacility());
    const [liveRefreshKey, bumpLiveRefresh] = useReducer((value: number) => value + 1, 0);
    const [forecastRefreshKey, bumpForecastRefresh] = useReducer((value: number) => value + 1, 0);
    const [scheduleRefreshKey, bumpScheduleRefresh] = useReducer((value: number) => value + 1, 0);
    const isOffline = useOnlineStatus();
    const {
        data,
        isLoading,
        error,
        liveDataSource,
        liveOutageState,
        hasPendingLiveRetry,
        cacheTimestampMs,
        prepareRefresh,
    } = useLiveFacilityData({facility, refreshKey: liveRefreshKey, isOffline});
    const {
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        isForecastLoading,
        hasPendingForecastRetry,
    } = useForecastData({facility, refreshKey: forecastRefreshKey});
    const {
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        hasPendingScheduleRetry,
    } = useFacilityHours({facility, refreshKey: scheduleRefreshKey});
    useVisibilityPolling({
        intervalMs: LIVE_REFRESH_INTERVAL_MS,
        onRefresh: bumpLiveRefresh,
        enabled: !isOffline,
        refreshOnVisible: true,
    });
    useVisibilityPolling({
        intervalMs: FORECAST_REFRESH_INTERVAL_MS,
        onRefresh: bumpForecastRefresh,
        enabled: !isOffline,
        refreshOnVisible: false,
    });
    useVisibilityPolling({
        intervalMs: SCHEDULE_REFRESH_INTERVAL_MS,
        onRefresh: bumpScheduleRefresh,
        enabled: !isOffline,
        refreshOnVisible: false,
    });
    const [lastManualRefresh, setLastManualRefresh] = useState(0);
    const [liveStatus, setLiveStatus] = useState<LiveStatus>("idle");
    const previousIsLoadingRef = useRef(isLoading);
    const manualRefreshFacilityRef = useRef<FacilityId | null>(null);
    const [forecastDaySelection, setForecastDaySelection] = useState<ForecastDaySelection>({
        key: null,
        offset: 0,
    });
    const [isCrowdAlertOpen, setIsCrowdAlertOpen] = useState(false);
    const [isInstallGuideOpen, setIsInstallGuideOpen] = useState(false);
    const debugEnabled = debugControlsEnabled(import.meta.env, window.location.hostname);
    const [initialDebugOverrides] = useState(() => loadDebugOverrides(debugEnabled));
    const [debugNowMs, setDebugNowMs] = useState<number | null>(() => (
        parseDebugNowMs(initialDebugOverrides.debugNowValue)
    ));
    const [clockTickTs, setClockTickTs] = useState(() => Date.now());
    const nowTs = debugNowMs ?? clockTickTs;
    const [predictionOverrideEnabled, setPredictionOverrideEnabled] = useState(false);
    const [closureOverrideEnabled, setClosureOverrideEnabled] = useState(
        initialDebugOverrides.closureOverrideEnabled
    );
    useEffect(() => {
        if (debugNowMs !== null) return;
        if (typeof window === "undefined") return;

        const timer = window.setInterval(() => {
            setClockTickTs(Date.now());
        }, CLOCK_TICK_MS);

        return () => {
            window.clearInterval(timer);
        };
    }, [debugNowMs]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        try {
            window.localStorage.setItem(FACILITY_STORAGE_KEY, String(facility));
        } catch {
            // Ignore storage write failures (private mode/quota exceeded).
        }
    }, [facility]);

    const handleFacilitySelect = (next: FacilityId) => {
        if (next === facility) return;
        onFacilityRouteChange?.(next);
        manualRefreshFacilityRef.current = null;
        setLiveStatus("idle");
        setLastManualRefresh(0);
        setForecastDaySelection({key: null, offset: 0});
        prepareRefresh();
        setFacility(next);
    };

    useEffect(() => {
        const wasLoading = previousIsLoadingRef.current;
        previousIsLoadingRef.current = isLoading;
        if (!wasLoading || isLoading) return;

        if (manualRefreshFacilityRef.current !== facility) return;
        manualRefreshFacilityRef.current = null;
        const refreshFailed = Boolean(error)
            || hasPendingLiveRetry
            || liveOutageState !== "none";
        const nextStatus: LiveStatus = refreshFailed ? "refresh-error" : "updated";
        let isCancelled = false;
        void Promise.resolve().then(() => {
            if (!isCancelled) setLiveStatus(nextStatus);
        });
        return () => {
            isCancelled = true;
        };
    }, [error, facility, hasPendingLiveRetry, isLoading, liveOutageState]);

    const activeData = data?.facilityId === facility ? data : null;
    useLayoutEffect(() => {
        if (debugNowMs !== null || !activeData) return;

        let isCancelled = false;
        // Sample the clock when a new live snapshot is accepted without making render impure.
        void Promise.resolve().then(() => {
            if (!isCancelled) {
                setClockTickTs(Date.now());
            }
        });

        return () => {
            isCancelled = true;
        };
    }, [activeData, debugNowMs]);
    const view = useMemo(() => buildDashboardViewModel({
        facility,
        nowTs,
        data,
        isLoading,
        error,
        liveDataSource,
        liveOutageState,
        hasPendingLiveRetry,
        cacheTimestampMs,
        isOffline,
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        isForecastLoading,
        hasPendingForecastRetry,
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        hasPendingScheduleRetry,
        forecastDaySelection,
        predictionOverrideEnabled,
        closureOverrideEnabled,
    }), [
        facility,
        nowTs,
        data,
        isLoading,
        error,
        liveDataSource,
        liveOutageState,
        hasPendingLiveRetry,
        cacheTimestampMs,
        isOffline,
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        isForecastLoading,
        hasPendingForecastRetry,
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        hasPendingScheduleRetry,
        forecastDaySelection,
        predictionOverrideEnabled,
        closureOverrideEnabled,
    ]);
    const {
        todayDateKey,
        tomorrowDateKey,
        nextOpenDateKey,
        scheduleStatus,
        showClosedFacilityMode,
        isExpectedOpenTomorrow,
        visibleForecastDays,
        canShowClosedTomorrowForecast,
        canShowActiveDailyForecast,
        canShowDailyForecastCard,
        warning,
        facilitySummary,
    } = view;

    const setPredictionOverride = useCallback((enabled: boolean) => {
        setPredictionOverrideEnabled(enabled);
    }, []);
    const setClosureOverride = useCallback((enabled: boolean) => {
        writeClosureOverrideStorage(enabled, debugEnabled);
        setClosureOverrideEnabled(enabled);
    }, [debugEnabled]);
    const setDebugNowOverride = useCallback((value: string | null) => {
        const parsed = parseDebugNowMs(value);
        writeDebugNowStorage(value && parsed !== null ? value : null, debugEnabled);
        setDebugNowMs(parsed);
        if (parsed === null) {
            setClockTickTs(Date.now());
        }
        return parsed;
    }, [debugEnabled]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        if (!debugEnabled) {
            delete window.recliveShowPredictions;
            delete window.recliveRestoreWarnings;
            delete window.reclivePredictionOverrideStatus;
            delete window.recliveOverrideClosure;
            delete window.recliveRestoreClosure;
            delete window.recliveClosureOverrideStatus;
            delete window.recliveSetDebugNow;
            delete window.recliveClearDebugNow;
            delete window.recliveDebugNowStatus;
            delete window.recliveDebugDashboardState;
            return;
        }

        window.recliveShowPredictions = () => {
            setPredictionOverride(true);
            return "Prediction override enabled. Warnings hidden; predictions forced visible.";
        };
        window.recliveRestoreWarnings = () => {
            setPredictionOverride(false);
            return "Prediction override disabled. Normal warning and prediction rules restored.";
        };
        window.reclivePredictionOverrideStatus = () => predictionOverrideEnabled;
        window.recliveOverrideClosure = (enabled = true) => {
            setClosureOverride(Boolean(enabled));
            return `Closure override ${enabled ? "enabled" : "disabled"}. Closed schedule mode ${enabled ? "bypassed" : "restored"}.`;
        };
        window.recliveRestoreClosure = () => {
            setClosureOverride(false);
            return "Closure override disabled. Normal schedule closure behavior restored.";
        };
        window.recliveClosureOverrideStatus = () => closureOverrideEnabled;
        window.recliveSetDebugNow = (value: string) => {
            const parsed = setDebugNowOverride(value);
            if (parsed === null) {
                return "Debug clock not changed. Use a Chicago-local timestamp like 2026-05-18T04:00:00.";
            }
            return `Debug clock set to ${value}.`;
        };
        window.recliveClearDebugNow = () => {
            setDebugNowOverride(null);
            return "Debug clock cleared. Live clock restored.";
        };
        window.recliveDebugNowStatus = () => {
            try {
                return window.localStorage.getItem(DEBUG_NOW_STORAGE_KEY);
            } catch {
                return null;
            }
        };

        return () => {
            delete window.recliveShowPredictions;
            delete window.recliveRestoreWarnings;
            delete window.reclivePredictionOverrideStatus;
            delete window.recliveOverrideClosure;
            delete window.recliveRestoreClosure;
            delete window.recliveClosureOverrideStatus;
            delete window.recliveSetDebugNow;
            delete window.recliveClearDebugNow;
            delete window.recliveDebugNowStatus;
            delete window.recliveDebugDashboardState;
        };
    }, [closureOverrideEnabled, debugEnabled, predictionOverrideEnabled, setClosureOverride, setDebugNowOverride, setPredictionOverride]);

    const manualRefresh = () => {
        if (isLoading) return;
        const now = Date.now();
        if (now - lastManualRefresh < MANUAL_REFRESH_COOLDOWN_MS) return;
        setLastManualRefresh(now);
        manualRefreshFacilityRef.current = facility;
        setLiveStatus("refreshing");
        prepareRefresh();
        bumpLiveRefresh();
    };

    const enablePullToRefresh = isPhoneViewport && isTouchCapable && isStandalonePwa;
    const {
        pullDistance,
        isPulling,
        isReadyToRefresh,
        showIndicator: showPullIndicator,
        resetPullGesture,
        handleTouchStart,
        handleTouchMove,
        handleTouchEnd,
    } = usePullToRefresh({
        enabled: enablePullToRefresh,
        blocked: isCrowdAlertOpen,
        isLoading,
        hasData: Boolean(data),
        onRefresh: manualRefresh,
    });

    useEffect(() => {
        if (!enablePullToRefresh || isCrowdAlertOpen) {
            resetPullGesture();
        }
    }, [enablePullToRefresh, isCrowdAlertOpen, resetPullGesture]);

    useEffect(() => {
        if (typeof window === "undefined") return;
        if (!debugEnabled) {
            delete window.recliveDebugDashboardState;
            return;
        }

        window.recliveDebugDashboardState = () => ({
            nowTs,
            todayDateKey,
            tomorrowDateKey,
            nextOpenDateKey,
            scheduleState: scheduleStatus.state,
            showClosedFacilityMode,
            isExpectedOpenTomorrow,
            forecastError,
            forecastDays: forecastDays.map((day) => ({date: day.date, dayName: day.dayName})),
            visibleForecastDays: visibleForecastDays.map((day) => ({date: day.date, dayName: day.dayName})),
            canShowClosedTomorrowForecast,
            canShowActiveDailyForecast,
            canShowDailyForecastCard,
            warningKind: warning.kind,
            occupancyStatus: facilitySummary.status,
        });

        return () => {
            delete window.recliveDebugDashboardState;
        };
    }, [
        canShowActiveDailyForecast,
        canShowClosedTomorrowForecast,
        canShowDailyForecastCard,
        debugEnabled,
        forecastDays,
        forecastError,
        facilitySummary.status,
        isExpectedOpenTomorrow,
        nextOpenDateKey,
        nowTs,
        scheduleStatus.state,
        showClosedFacilityMode,
        todayDateKey,
        tomorrowDateKey,
        visibleForecastDays,
        warning.kind,
    ]);

    return {
        facility,
        nowTs,
        data,
        isLoading,
        error,
        liveDataSource,
        liveOutageState,
        hasPendingLiveRetry,
        cacheTimestampMs,
        isOffline,
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        isForecastLoading,
        hasPendingForecastRetry,
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        hasPendingScheduleRetry,
        forecastDaySelection,
        predictionOverrideEnabled,
        closureOverrideEnabled,
        view,
        liveRefreshKey,
        forecastRefreshKey,
        scheduleRefreshKey,
        liveStatus,
        lastManualRefresh,
        debugEnabled,
        debugNowMs,
        isCrowdAlertOpen,
        isInstallGuideOpen,
        isStandalonePwa,
        isTouchCapable,
        enablePullToRefresh,
        pullDistance,
        isPulling,
        isReadyToRefresh,
        showPullIndicator,
        handleFacilitySelect,
        manualRefresh,
        setForecastDaySelection,
        setIsCrowdAlertOpen,
        setIsInstallGuideOpen,
        setPredictionOverride,
        setClosureOverride,
        setDebugNowOverride,
        resetPullGesture,
        handleTouchStart,
        handleTouchMove,
        handleTouchEnd,
    };
}
