import {
    Suspense,
    lazy,
    useCallback,
    useEffect,
    useMemo,
    useState,
} from "react";
import {
    Alert,
    Box,
    CircularProgress,
    Container,
    Stack,
    Typography,
    useMediaQuery,
} from "@mui/material";
import {AnimatePresence, motion, useReducedMotion} from "framer-motion";
import {useTheme, type PaletteMode} from "@mui/material/styles";
import FacilitySelector from "../facilities/FacilitySelector";
import type {AlertSectionOption} from "../facilities/CrowdAlertSubscriptionCard";
import OccupancyHero from "../facilities/OccupancyHero";
import SectionCommandCenter from "../facilities/SectionCommandCenter";
import SectionSummaryOther from "../facilities/SectionSummaryOther";
import ForecastWindowsCard from "../facilities/ForecastWindowsCard";
import ScheduleStatusCard from "../facilities/ScheduleStatusCard";
import FacilityHoursBlock from "../facilities/FacilityHoursBlock";
import {
    FACILITY_DASHBOARD_CONFIG,
    FACILITY_KNOWN_IDS,
    isSectionRow,
} from "../facilities/constants";
import type {FacilityId, FacilityPayload} from "../lib/types/facility";
import type {ForecastDay, ForecastHour} from "../lib/types/forecast";
import {
    getChicagoDayAge,
    getChicagoTimestampMs,
    getChicagoHour,
    isWithinChicagoHours,
} from "../shared/utils/chicagoTime";
import {
    clampPercent,
    combineOccupancyThresholds,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {useFacilitySeo} from "./seo";
import AlertsPanel from "./components/AlertsPanel";
import InstallGuideDialog from "./components/InstallGuideDialog";
import AppFooter from "./components/AppFooter";
import ThemeModeToggle from "../shared/components/ThemeModeToggle";
import {useStandalonePwa} from "./hooks/useStandalonePwa";
import {usePullToRefresh} from "./hooks/usePullToRefresh";
import {useOnlineStatus} from "./hooks/useOnlineStatus";
import {useLiveFacilityData} from "./hooks/useLiveFacilityData";
import {useForecastData, type ForecastHourBounds} from "./hooks/useForecastData";
import {useFacilityHours} from "./hooks/useFacilityHours";
import {resolveDashboardWarning} from "./warningStatus";
import {
    getFacilityNextOpenTimestamp,
    getFacilityOpenStatus,
    getFacilityOpenWindowsForDate,
    type FacilityOpenWindow,
} from "../shared/utils/facilityScheduleStatus";

const FloorHeatMapCard = lazy(() => import("../facilities/FloorHeatMapCard"));

const FACILITY_STORAGE_KEY = "reclive:selectedFacility";
const CLOSURE_OVERRIDE_STORAGE_KEY = "reclive:closureOverride";
const DEBUG_NOW_STORAGE_KEY = "reclive:debugNow";
const AUTO_REFRESH_INTERVAL_MS = 15 * 60 * 1000;
const AUTO_REFRESH_RETRY_INTERVAL_MS = 2 * 60 * 1000;
const AUTO_REFRESH_CHECK_INTERVAL_MS = 60 * 1000;
const MANUAL_REFRESH_COOLDOWN_MS = 3000;
const STALE_HIDE_DAILY_FORECAST_AFTER_DAYS = 2;
const FORECAST_VISIBLE_SECTIONS = new Set(["fitness floors", "basketball courts"]);
const CLOCK_TICK_MS = 30 * 1000;
const CONTENT_EASE = [0.22, 1, 0.36, 1] as const;
const CONTENT_EXIT_EASE = [0.4, 0, 1, 1] as const;
const MS_PER_HOUR = 60 * 60 * 1000;
const MS_PER_DAY = 24 * MS_PER_HOUR;

interface ChicagoDateParts {
    year: number;
    month: number;
    day: number;
    hour: number;
    minute: number;
}

interface AppProps {
    initialFacility?: FacilityId;
    onFacilityRouteChange?: (facility: FacilityId) => void;
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}

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

const getInitialClosureOverride = (): boolean => {
    if (typeof window === "undefined") return false;

    const params = new URLSearchParams(window.location.search);
    const queryValue = params.get("overrideClosure") ?? params.get("debugClosure");
    if (queryValue !== null) {
        return queryValue !== "0" && queryValue.toLowerCase() !== "false";
    }

    try {
        return window.localStorage.getItem(CLOSURE_OVERRIDE_STORAGE_KEY) === "true";
    } catch {
        return false;
    }
};

const writeClosureOverrideStorage = (enabled: boolean) => {
    if (typeof window === "undefined") return;
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

const getInitialDebugNowMs = (): number | null => {
    if (typeof window === "undefined") return null;

    const params = new URLSearchParams(window.location.search);
    const queryValue = params.get("debugNow");
    if (queryValue !== null) {
        return parseDebugNowMs(queryValue);
    }

    try {
        return parseDebugNowMs(window.localStorage.getItem(DEBUG_NOW_STORAGE_KEY));
    } catch {
        return null;
    }
};

const writeDebugNowStorage = (value: string | null) => {
    if (typeof window === "undefined") return;
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

const shiftDateKeyByDays = (dateKey: string | null, dayShift: number): string | null => {
    if (!dateKey) return null;
    const match = dateKey.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
        return null;
    }

    const shifted = new Date(Date.UTC(year, month - 1, day + dayShift));
    if (Number.isNaN(shifted.getTime())) return null;
    return `${shifted.getUTCFullYear()}-${pad2(shifted.getUTCMonth() + 1)}-${pad2(shifted.getUTCDate())}`;
};

const getLatestTimestamp = (payload: FacilityPayload | null | undefined): string | null => {
    if (!payload) return null;

    let latest: string | null = null;
    let latestTs: number | null = null;
    for (const location of payload.locations) {
        const value = location.lastUpdated;
        if (!value) continue;

        const timestampMs = getChicagoTimestampMs(value);
        if (timestampMs !== null) {
            if (latestTs === null || timestampMs > latestTs) {
                latestTs = timestampMs;
                latest = value;
            }
            continue;
        }

        if (!latest) {
            latest = value;
        }
    }

    return latest;
};

const normalizeSectionTitle = (title: string): string =>
    title.replace(/^[^a-zA-Z0-9]+/, "").replace(/[_\s]+/g, " ").trim().toLowerCase();

const chicagoPartsFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
});
const chicagoHourMinuteFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
});

const pad2 = (value: number): string => String(value).padStart(2, "0");

const getChicagoDateParts = (value: Date): ChicagoDateParts | null => {
    const parts = chicagoPartsFormatter.formatToParts(value);
    const year = Number(parts.find((part) => part.type === "year")?.value);
    const month = Number(parts.find((part) => part.type === "month")?.value);
    const day = Number(parts.find((part) => part.type === "day")?.value);
    const hour = Number(parts.find((part) => part.type === "hour")?.value);
    const minute = Number(parts.find((part) => part.type === "minute")?.value);

    if (
        !Number.isInteger(year)
        || !Number.isInteger(month)
        || !Number.isInteger(day)
        || !Number.isInteger(hour)
        || !Number.isInteger(minute)
    ) {
        return null;
    }

    return {year, month, day, hour, minute};
};

const chicagoDayKey = (value: ChicagoDateParts): string => `${value.year}-${pad2(value.month)}-${pad2(value.day)}`;

const formatNextOpenRelative = (
    nextOpenTimestamp: string | null | undefined,
    nowTimestampMs: number
): string | null => {
    if (!nextOpenTimestamp) return null;

    const nextOpenMs = getChicagoTimestampMs(nextOpenTimestamp);
    if (nextOpenMs === null) return null;

    const diffMs = Math.max(0, nextOpenMs - nowTimestampMs);
    if (diffMs < MS_PER_DAY) {
        const hours = Math.max(1, Math.ceil(diffMs / MS_PER_HOUR));
        return `in ${hours} hour${hours === 1 ? "" : "s"}`;
    }

    const days = Math.max(1, Math.ceil(diffMs / MS_PER_DAY));
    return `in ${days} day${days === 1 ? "" : "s"}`;
};

const formatCacheAge = (cacheTimestampMs: number | null, nowTimestampMs: number): string | null => {
    if (cacheTimestampMs === null || !Number.isFinite(cacheTimestampMs)) return null;

    const ageMs = Math.max(0, nowTimestampMs - cacheTimestampMs);
    const minutes = Math.floor(ageMs / (60 * 1000));
    if (minutes < 1) return "under a minute old";
    if (minutes < 60) {
        return `${minutes} minute${minutes === 1 ? "" : "s"} old`;
    }

    const hours = Math.floor(minutes / 60);
    if (hours < 24) {
        return `${hours} hour${hours === 1 ? "" : "s"} old`;
    }

    const days = Math.floor(hours / 24);
    return `${days} day${days === 1 ? "" : "s"} old`;
};

const getChicagoMinuteOfDay = (value: string | null | undefined): number | null => {
    const timestampMs = getChicagoTimestampMs(value);
    if (timestampMs === null) return null;

    const parts = chicagoHourMinuteFormatter.formatToParts(new Date(timestampMs));
    const hour = Number(parts.find((part) => part.type === "hour")?.value);
    const minute = Number(parts.find((part) => part.type === "minute")?.value);
    if (!Number.isInteger(hour) || !Number.isInteger(minute)) return null;

    return Math.max(0, Math.min((24 * 60) - 1, (hour * 60) + minute));
};

const isMinuteWithinOpenWindows = (minute: number, windows: FacilityOpenWindow[]): boolean =>
    windows.some((window) => minute >= window.startMinutes && minute < window.endMinutes);

const normalizeHoursText = (value: string): string =>
    value
        .toLowerCase()
        .replace(/[–—]/g, "-")
        .replace(/\s+/g, " ")
        .trim();

const parseClockMinutes = (token: string): number | null => {
    const normalized = normalizeHoursText(token).replace(/\./g, "");
    if (!normalized) return null;
    if (normalized === "midnight") return 0;
    if (normalized === "noon") return 12 * 60;

    const match = normalized.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/);
    if (!match) return null;

    let hour = Number(match[1]);
    const minute = Number(match[2] ?? "0");
    const suffix = match[3] ?? "";
    if (!Number.isInteger(hour) || !Number.isInteger(minute) || minute < 0 || minute > 59) {
        return null;
    }

    if (suffix === "am") {
        if (hour === 12) hour = 0;
    } else if (suffix === "pm") {
        if (hour < 12) hour += 12;
    }

    if (hour < 0 || hour > 23) return null;
    return hour * 60 + minute;
};

const parseScheduleStartMinutes = (hoursText: string): number | null => {
    const normalized = normalizeHoursText(hoursText);
    if (!normalized) return null;
    if (normalized.includes("closed")) return null;
    if (normalized.includes("24 hours")) return 0;

    const parts = normalized.split(/\s*-\s*/).filter(Boolean);
    if (parts.length !== 2) return null;
    return parseClockMinutes(parts[0]);
};

const normalizeHour = (value: number | null | undefined): number | null => {
    if (typeof value !== "number" || !Number.isInteger(value)) return null;
    return Math.max(0, Math.min(23, value));
};

const normalizeForecastBounds = (bounds: ForecastHourBounds): ForecastHourBounds => {
    const startHour = normalizeHour(bounds.startHour);
    const endHour = normalizeHour(bounds.endHour);
    if (startHour === null || endHour === null) {
        return {startHour, endHour};
    }
    if (endHour < startHour) {
        return {startHour: endHour, endHour: startHour};
    }
    return {startHour, endHour};
};

const deriveForecastBounds = (days: ForecastDay[]): ForecastHourBounds => {
    let startHour: number | null = null;
    let endHour: number | null = null;

    for (const day of days) {
        for (const category of day.categories ?? []) {
            for (const hour of category.hours ?? []) {
                const timestampMs = getChicagoTimestampMs(hour.hourStart);
                if (timestampMs === null) continue;
                const chicagoHour = getChicagoHour(new Date(timestampMs));
                if (chicagoHour === null) continue;
                if (startHour === null || chicagoHour < startHour) {
                    startHour = chicagoHour;
                }
                if (endHour === null || chicagoHour > endHour) {
                    endHour = chicagoHour;
                }
            }
        }
    }

    return normalizeForecastBounds({startHour, endHour});
};

const buildSectionForecastMap = (
    day: ForecastDay | null,
    nowTs: number,
    openWindows: FacilityOpenWindow[],
    enforceWorkingHours: boolean
): Record<string, ForecastHour[]> => {
    if (!day?.categories) return {};
    if (enforceWorkingHours && openWindows.length === 0) return {};

    const sectionMap: Record<string, ForecastHour[]> = {};

    for (const category of day.categories) {
        const key = normalizeSectionTitle(category.title);
        if (!FORECAST_VISIBLE_SECTIONS.has(key)) {
            continue;
        }

        const sorted = [...(category.hours ?? [])].sort(
            (a, b) => {
                const leftTs = getChicagoTimestampMs(a.hourStart);
                const rightTs = getChicagoTimestampMs(b.hourStart);
                if (leftTs === null && rightTs === null) return 0;
                if (leftTs === null) return 1;
                if (rightTs === null) return -1;
                return leftTs - rightTs;
            }
        );
        const currentChicagoHour = getChicagoHour(new Date(nowTs));
        const futureHourlyBuckets = new Map<number, {
            expectedSum: number;
            sampleCount: number;
            hourStart: string;
        }>();

        for (const hour of sorted
            .filter((hour) => {
                const hourTs = getChicagoTimestampMs(hour.hourStart);
                if (hourTs === null || hourTs <= nowTs) return false;
                // Keep "+1/+2/+3h" chips in the same Chicago day, never after midnight.
                if (getChicagoDayAge(hour.hourStart, new Date(nowTs)) !== 0) {
                    return false;
                }
                if (!enforceWorkingHours) {
                    return true;
                }

                const minuteOfDay = getChicagoMinuteOfDay(hour.hourStart);
                if (minuteOfDay === null) return false;
                return isMinuteWithinOpenWindows(minuteOfDay, openWindows);
            })) {
            const minuteOfDay = getChicagoMinuteOfDay(hour.hourStart);
            if (minuteOfDay === null) continue;

            const chicagoHour = Math.floor(minuteOfDay / 60);
            if (currentChicagoHour !== null && chicagoHour <= currentChicagoHour) {
                continue
            }

            const current = futureHourlyBuckets.get(chicagoHour);
            futureHourlyBuckets.set(chicagoHour, {
                expectedSum: (current?.expectedSum ?? 0) + Math.max(0, hour.expectedCount),
                sampleCount: (current?.sampleCount ?? 0) + 1,
                hourStart: current?.hourStart ?? hour.hourStart,
            });
        }

        sectionMap[key] = [...futureHourlyBuckets.entries()]
            .sort((a, b) => a[0] - b[0])
            .slice(0, 3)
            .map(([, bucket]) => ({
                hourStart: bucket.hourStart,
                expectedCount: bucket.expectedSum / Math.max(1, bucket.sampleCount),
            }));
    }

    return sectionMap;
};

export default function App({
    initialFacility,
    onFacilityRouteChange,
    themeMode,
    onThemeModeChange,
}: AppProps) {
    const reduceMotion = useReducedMotion();
    const theme = useTheme();
    const useDesktopAlertsModal = useMediaQuery(theme.breakpoints.up("md"));
    const isPhoneViewport = useMediaQuery(theme.breakpoints.down("sm"));
    const {isStandalonePwa, isTouchCapable} = useStandalonePwa();
    const [facility, setFacility] = useState<FacilityId>(() => initialFacility ?? getStoredFacility());
    const [refreshKey, setRefreshKey] = useState(0);
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
    } = useLiveFacilityData({facility, refreshKey, isOffline});
    const {
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        isForecastLoading,
        hasPendingForecastRetry,
    } = useForecastData({facility, refreshKey});
    const {
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        hasPendingScheduleRetry,
    } = useFacilityHours({facility, refreshKey});
    const [lastAutoRefresh, setLastAutoRefresh] = useState(() => Date.now());
    const [lastManualRefresh, setLastManualRefresh] = useState(0);
    const [forecastDaySelection, setForecastDaySelection] = useState<{key: string | null; offset: number}>({
        key: null,
        offset: 0,
    });
    const [isCrowdAlertOpen, setIsCrowdAlertOpen] = useState(false);
    const [isInstallGuideOpen, setIsInstallGuideOpen] = useState(false);
    const [debugNowMs, setDebugNowMs] = useState<number | null>(() => getInitialDebugNowMs());
    const [nowTs, setNowTs] = useState(() => getInitialDebugNowMs() ?? Date.now());
    const [predictionOverrideEnabled, setPredictionOverrideEnabled] = useState(false);
    const [closureOverrideEnabled, setClosureOverrideEnabled] = useState(() => getInitialClosureOverride());
    useFacilitySeo(facility);
    const facilityContentVariants = useMemo(
        () => (reduceMotion
            ? {
                hidden: {opacity: 1, y: 0},
                show: {opacity: 1, y: 0},
                exit: {opacity: 1, y: 0},
            }
            : {
                hidden: {opacity: 0, y: 14},
                show: {
                    opacity: 1,
                    y: 0,
                    transition: {
                        duration: 0.32,
                        ease: CONTENT_EASE,
                        staggerChildren: 0.075,
                        delayChildren: 0.03,
                    },
                },
                exit: {
                    opacity: 0,
                    y: -8,
                    transition: {duration: 0.18, ease: CONTENT_EXIT_EASE},
                },
            }),
        [reduceMotion]
    );
    const facilityItemVariants = useMemo(
        () => (reduceMotion
            ? {
                hidden: {opacity: 1, y: 0},
                show: {opacity: 1, y: 0},
            }
            : {
                hidden: {opacity: 0, y: 10},
                show: {
                    opacity: 1,
                    y: 0,
                    transition: {duration: 0.28, ease: CONTENT_EASE},
                },
            }),
        [reduceMotion]
    );

    useEffect(() => {
        if (debugNowMs !== null) return;
        if (typeof window === "undefined") return;

        const timer = window.setInterval(() => {
            setNowTs(Date.now());
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

    const triggerRefresh = useCallback((action: () => void) => {
        prepareRefresh();
        setLastAutoRefresh(Date.now());
        action();
    }, [prepareRefresh]);

    const autoRefreshIntervalMs = (
        hasPendingLiveRetry
        || hasPendingForecastRetry
        || hasPendingScheduleRetry
        || liveOutageState !== "none"
        || liveDataSource === "fallback_api"
        || Boolean(forecastError)
        || Boolean(facilityHoursError)
    )
        ? AUTO_REFRESH_RETRY_INTERVAL_MS
        : AUTO_REFRESH_INTERVAL_MS;

    useEffect(() => {
        if (typeof window === "undefined") return;

        const interval = window.setInterval(() => {
            if (!isLoading && Date.now() - lastAutoRefresh >= autoRefreshIntervalMs) {
                triggerRefresh(() => setRefreshKey((key) => key + 1));
            }
        }, AUTO_REFRESH_CHECK_INTERVAL_MS);

        return () => {
            window.clearInterval(interval);
        };
    }, [autoRefreshIntervalMs, isLoading, lastAutoRefresh, triggerRefresh]);

    const handleFacilitySelect = (next: FacilityId) => {
        if (next === facility) return;
        onFacilityRouteChange?.(next);
        setLastManualRefresh(0);
        setForecastDaySelection({key: null, offset: 0});
        triggerRefresh(() => setFacility(next));
    };

    const activeData = data?.facilityId === facility ? data : null;

    const total = activeData
        ? activeData.locations.reduce((sum, l) => sum + (l.currentCapacity ?? 0), 0)
        : 0;

    const max = activeData
        ? activeData.locations.reduce((sum, l) => sum + (l.maxCapacity ?? 0), 0)
        : 0;
    const lastUpdated = getLatestTimestamp(activeData);
    const inferredHourBounds = useMemo(
        () => deriveForecastBounds(forecastDays),
        [forecastDays]
    );
    const resolvedHourBounds = normalizeForecastBounds({
        startHour: forecastHourBounds.startHour ?? inferredHourBounds.startHour,
        endHour: forecastHourBounds.endHour ?? inferredHourBounds.endHour,
    });
    const predictionStartHour = resolvedHourBounds.startHour ?? 0;
    const predictionEndHour = resolvedHourBounds.endHour ?? 23;
    const predictionEndHourExclusive = Math.min(24, predictionEndHour + 1);
    const lastUpdatedDayAge = getChicagoDayAge(lastUpdated, new Date(nowTs));
    const todayDateKey = useMemo(() => {
        const nowParts = getChicagoDateParts(new Date(nowTs));
        return nowParts ? chicagoDayKey(nowParts) : null;
    }, [nowTs]);
    const tomorrowDateKey = useMemo(
        () => shiftDateKeyByDays(todayDateKey, 1),
        [todayDateKey]
    );
    const scheduleStatus = useMemo(
        () => getFacilityOpenStatus(activeSchedule, new Date(nowTs)),
        [activeSchedule, nowTs]
    );
    const isScheduledClosedNow = scheduleStatus.state === "closed";
    const isScheduledOpenButDataNotLive = (() => {
        if (scheduleStatus.state !== "open") return false;

        const rule = scheduleStatus.matchedRule;
        const startMinutes = rule ? parseScheduleStartMinutes(rule.hours) : null;
        if (startMinutes === null) return false;

        if (!lastUpdated) return true;

        const lastUpdatedMs = getChicagoTimestampMs(lastUpdated);
        if (lastUpdatedMs === null) return true;

        const nowParts = getChicagoDateParts(new Date(nowTs));
        const updatedParts = getChicagoDateParts(new Date(lastUpdatedMs));
        if (!nowParts || !updatedParts) return false;

        const nowDay = chicagoDayKey(nowParts);
        const updatedDay = chicagoDayKey(updatedParts);
        if (updatedDay < nowDay) return true;
        if (updatedDay > nowDay) return false;

        const updatedMinutes = updatedParts.hour * 60 + updatedParts.minute;
        return updatedMinutes < startMinutes;
    })();
    const hasAnyError = Boolean(error) || Boolean(forecastError);
    const isDataLikelyStale = typeof lastUpdatedDayAge === "number" && lastUpdatedDayAge >= 1;
    const isWithinPredictionHours = isWithinChicagoHours(
        predictionStartHour,
        predictionEndHourExclusive,
        new Date(nowTs)
    );
    const warning = resolveDashboardWarning({
        hasAnyError,
        isOffline,
        liveOutageState,
        liveDataSource,
        forecastError,
        isScheduledClosedNow: isScheduledClosedNow && !closureOverrideEnabled,
        isScheduledOpenButDataNotLive,
        isDataLikelyStale,
    });
    const isDataVeryStale = typeof lastUpdatedDayAge === "number" && lastUpdatedDayAge >= STALE_HIDE_DAILY_FORECAST_AFTER_DAYS;
    const cacheAgeText = liveDataSource === "cache" ? formatCacheAge(cacheTimestampMs, nowTs) : null;
    const baseWarningText = predictionOverrideEnabled || closureOverrideEnabled
        ? null
        : (
            warning.kind === "stale" && isDataVeryStale
                ? "The gym may be closed right now. Occupancy info is not live, and forecasts are hidden."
                : warning.text
        );
    const warningText = (
        baseWarningText
        && cacheAgeText
        && (warning.kind === "offline_cache" || warning.kind === "total_outage_cache")
    )
        ? `${baseWarningText} Saved snapshot is ${cacheAgeText}.`
        : baseWarningText;
    const canShowHourlyRoomForecasts = (
        predictionOverrideEnabled
        || closureOverrideEnabled
        || (!warning.hidePredictions && isWithinPredictionHours)
    );
    const hasResolvedSchedule = Boolean(activeSchedule && Array.isArray(activeSchedule.sections) && activeSchedule.sections.length > 0);
    const nextOpenTimestamp = useMemo(
        () => getFacilityNextOpenTimestamp(activeSchedule, new Date(nowTs)),
        [activeSchedule, nowTs]
    );
    const nextOpenLabel = useMemo(
        () => formatNextOpenRelative(nextOpenTimestamp, nowTs),
        [nextOpenTimestamp, nowTs]
    );
    const nextOpenDateKey = useMemo(() => {
        const nextOpenMs = getChicagoTimestampMs(nextOpenTimestamp);
        if (nextOpenMs === null) return null;

        const nextOpenParts = getChicagoDateParts(new Date(nextOpenMs));
        return nextOpenParts ? chicagoDayKey(nextOpenParts) : null;
    }, [nextOpenTimestamp]);
    const showClosedFacilityMode = hasResolvedSchedule && scheduleStatus.state === "closed" && !closureOverrideEnabled;
    const isExpectedOpenTomorrow = Boolean(tomorrowDateKey && nextOpenDateKey === tomorrowDateKey);

    const setPredictionOverride = useCallback((enabled: boolean) => {
        setPredictionOverrideEnabled(enabled);
    }, []);
    const setClosureOverride = useCallback((enabled: boolean) => {
        writeClosureOverrideStorage(enabled);
        setClosureOverrideEnabled(enabled);
    }, []);
    const setDebugNowOverride = useCallback((value: string | null) => {
        const parsed = parseDebugNowMs(value);
        writeDebugNowStorage(value && parsed !== null ? value : null);
        setDebugNowMs(parsed);
        setNowTs(parsed ?? Date.now());
        return parsed;
    }, []);

    useEffect(() => {
        if (typeof window === "undefined") return;
        const isLocalDebugHost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
        if (!import.meta.env.DEV && !isLocalDebugHost) {
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
    }, [closureOverrideEnabled, predictionOverrideEnabled, setClosureOverride, setDebugNowOverride, setPredictionOverride]);

    const manualRefresh = () => {
        if (isLoading) return;
        const now = Date.now();
        if (now - lastManualRefresh < MANUAL_REFRESH_COOLDOWN_MS) return;
        setLastManualRefresh(now);
        triggerRefresh(() => setRefreshKey((key) => key + 1));
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

    const dashboardConfig = FACILITY_DASHBOARD_CONFIG[facility];
    const knownIds = FACILITY_KNOWN_IDS[facility];
    const hasOtherSectionLocations = useMemo(
        () => {
            if (!activeData) return false;
            const knownIdSet = new Set(knownIds);
            return activeData.locations.some((location) => !knownIdSet.has(location.locationId));
        },
        [activeData, knownIds]
    );
    const sectionConfigs = useMemo(
        () => dashboardConfig.sections.flatMap((layout) => (isSectionRow(layout) ? [...layout] : [layout])),
        [dashboardConfig]
    );
    const alertSections: AlertSectionOption[] = useMemo(
        () => {
            const overall: AlertSectionOption = {
                key: "overall",
                label: "Entire Facility",
                total,
                max,
                percent: clampPercent(max ? (total / max) * 100 : 0),
            };

            const bySection = sectionConfigs.map((section) => {
                const idSet = new Set<number>(section.ids);
                const sectionLocations = activeData
                    ? activeData.locations.filter((location) => idSet.has(location.locationId))
                    : [];
                const sectionTotal = sectionLocations.reduce((sum, location) => sum + (location.currentCapacity ?? 0), 0);
                const sectionMax = sectionLocations.reduce((sum, location) => sum + (location.maxCapacity ?? 0), 0);
                const sectionPercent = clampPercent(sectionMax ? (sectionTotal / sectionMax) * 100 : 0);

                return {
                    key: normalizeSectionTitle(section.title),
                    label: section.title,
                    total: sectionTotal,
                    max: sectionMax,
                    percent: sectionPercent,
                };
            });

            return [overall, ...bySection];
        },
        [sectionConfigs, activeData, total, max]
    );
    const visibleForecastDays = useMemo(
        () => {
            const isOpenForecastDay = (forecastDay: ForecastDay) => (
                !hasResolvedSchedule
                || getFacilityOpenWindowsForDate(activeSchedule, forecastDay.date).length > 0
            );
            const openForecastDays = forecastDays.filter(isOpenForecastDay);

            if (showClosedFacilityMode) {
                if (!tomorrowDateKey || !isExpectedOpenTomorrow) {
                    return [];
                }

                const tomorrowIndex = forecastDays.findIndex((forecastDay) => forecastDay.date === tomorrowDateKey);
                const daysFromTomorrow = tomorrowIndex >= 0
                    ? forecastDays.slice(tomorrowIndex)
                    : forecastDays.slice(1);
                const openDaysFromTomorrow = daysFromTomorrow.filter(isOpenForecastDay);
                if (openDaysFromTomorrow.length > 0) {
                    return openDaysFromTomorrow.slice(0, 4);
                }

                return daysFromTomorrow.slice(0, 4);
            }

            return openForecastDays.slice(0, 4);
        },
        [
            activeSchedule,
            forecastDays,
            hasResolvedSchedule,
            isExpectedOpenTomorrow,
            showClosedFacilityMode,
            tomorrowDateKey,
        ]
    );
    const forecastDisplayKey = `${showClosedFacilityMode ? "closed" : "active"}:${visibleForecastDays.map((day) => day.date).join("|")}`;
    const todayForecastDay = visibleForecastDays[0] ?? null;
    const activeForecastDayOffset = forecastDaySelection.key === forecastDisplayKey
        ? forecastDaySelection.offset
        : 0;
    const resolvedForecastDayOffset = Math.min(
        activeForecastDayOffset,
        Math.max(0, visibleForecastDays.length - 1)
    );
    const selectedForecastDay =
        visibleForecastDays[resolvedForecastDayOffset] ?? todayForecastDay;
    const canShowClosedTomorrowForecast = showClosedFacilityMode
        && isExpectedOpenTomorrow
        && !forecastError
        && visibleForecastDays.length > 0;
    const canShowActiveDailyForecast = !showClosedFacilityMode && (predictionOverrideEnabled || closureOverrideEnabled || (
        !forecastError
        && !isDataVeryStale
        && warning.kind !== "offline_cache"
        && warning.kind !== "total_outage_cache"
        && warning.kind !== "prediction_unavailable"
        && warning.kind !== "facility_fallback"
    ));
    const canShowDailyForecastCard = canShowClosedTomorrowForecast || canShowActiveDailyForecast;
    useEffect(() => {
        if (typeof window === "undefined") return;
        const isLocalDebugHost = ["localhost", "127.0.0.1", "::1"].includes(window.location.hostname);
        if (!import.meta.env.DEV && !isLocalDebugHost) return;

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
            isDataVeryStale,
        });

        return () => {
            delete window.recliveDebugDashboardState;
        };
    }, [
        canShowActiveDailyForecast,
        canShowClosedTomorrowForecast,
        canShowDailyForecastCard,
        forecastDays,
        forecastError,
        isDataVeryStale,
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
    const occupancyThresholds = useMemo(
        () => forecastOccupancyThresholds ?? combineOccupancyThresholds(
            (activeData?.locations ?? []).map((location) => ({
                thresholds: forecastLocationOccupancyThresholds[location.locationId],
                weight: location.maxCapacity ?? 0,
            }))
        ),
        [activeData, forecastLocationOccupancyThresholds, forecastOccupancyThresholds]
    );
    const sectionOccupancyThresholds = useMemo(
        () => {
            const thresholdsBySection: Partial<Record<string, OccupancyThresholds>> = {
                ...forecastSectionOccupancyThresholds,
            };
            const activeLocations = activeData?.locations ?? [];

            for (const section of sectionConfigs) {
                const sectionKey = normalizeSectionTitle(section.title);
                if (thresholdsBySection[sectionKey]) {
                    continue;
                }
                const idSet = new Set<number>(section.ids);
                const sectionThresholds = combineOccupancyThresholds(
                    activeLocations
                        .filter((location) => idSet.has(location.locationId))
                        .map((location) => ({
                            thresholds: forecastLocationOccupancyThresholds[location.locationId],
                            weight: location.maxCapacity ?? 0,
                        }))
                ) ?? occupancyThresholds;

                if (sectionThresholds) {
                    thresholdsBySection[sectionKey] = sectionThresholds;
                }
            }

            return thresholdsBySection;
        },
        [
            activeData,
            forecastLocationOccupancyThresholds,
            forecastSectionOccupancyThresholds,
            occupancyThresholds,
            sectionConfigs,
        ]
    );
    const enforceSectionForecastWorkingHours = Boolean(activeSchedule && todayForecastDay?.date);
    const sectionForecastOpenWindows = useMemo(
        () => getFacilityOpenWindowsForDate(activeSchedule, todayForecastDay?.date ?? null),
        [activeSchedule, todayForecastDay?.date]
    );

    const sectionForecastMap = useMemo(
        () => (
            canShowHourlyRoomForecasts
                ? buildSectionForecastMap(
                    todayForecastDay,
                    nowTs,
                    sectionForecastOpenWindows,
                    enforceSectionForecastWorkingHours
                )
                : {}
        ),
        [
            canShowHourlyRoomForecasts,
            nowTs,
            todayForecastDay,
            sectionForecastOpenWindows,
            enforceSectionForecastWorkingHours,
        ]
    );
    const pullIndicatorHeight = showPullIndicator
        ? 28
        : (isPulling ? Math.min(28, Math.max(12, Math.round(pullDistance * 0.35))) : 0);
    const pullIndicatorText = showPullIndicator
        ? "Refreshing..."
        : (isReadyToRefresh ? "Release to refresh" : "Pull down to refresh");
    const sectionBlockGap = {xs: 2, sm: 2.5} as const;
    return (
        <Box
            onTouchStart={enablePullToRefresh ? handleTouchStart : undefined}
            onTouchMove={enablePullToRefresh ? handleTouchMove : undefined}
            onTouchEnd={enablePullToRefresh ? handleTouchEnd : undefined}
            onTouchCancel={enablePullToRefresh ? resetPullGesture : undefined}
            sx={{
                pt: {xs: 2, sm: 3},
                pb: {xs: 2.5, sm: 3.5},
                bgcolor: "background.default",
                minHeight: "100vh",
                overscrollBehaviorY: enablePullToRefresh ? "contain" : undefined,
            }}
        >
            <Container
                maxWidth="sm"
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: {xs: 2, sm: 3},
                    pt: {xs: 1.5, sm: 2},
                    px: {xs: 2, sm: 0},
                }}
            >
                <Box
                    sx={{
                        height: pullIndicatorHeight,
                        opacity: pullIndicatorHeight > 0 ? 1 : 0,
                        overflow: "hidden",
                        transition: "height 180ms ease, opacity 180ms ease",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >
                    <Stack direction="row" spacing={0.8} alignItems="center">
                        {showPullIndicator && (
                            <CircularProgress size={14} thickness={6}/>
                        )}
                        <Typography variant="caption" color="text.secondary" sx={{fontWeight: 700}}>
                            {pullIndicatorText}
                        </Typography>
                    </Stack>
                </Box>

                <FacilitySelector facility={facility} onSelect={handleFacilitySelect}/>

                {isLoading && !data && (
                    <Box
                        sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: 2,
                            py: 5,
                        }}
                    >
                        <CircularProgress size={28} thickness={5}/>
                        <Typography color="text.secondary" fontWeight={600}>
                            Loading...
                        </Typography>
                    </Box>
                )}

                {error && !activeData && (
                    <Alert severity="warning" sx={{borderRadius: 2}}>
                        {error}
                    </Alert>
                )}

                <AnimatePresence mode="wait" initial={false}>
                    {activeData && (
                        <Box
                            key={facility}
                            component={motion.div}
                            variants={facilityContentVariants}
                            initial="hidden"
                            animate="show"
                            exit="exit"
                            sx={{display: "flex", flexDirection: "column", gap: sectionBlockGap}}
                        >
                            {!showClosedFacilityMode && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <OccupancyHero
                                        total={total}
                                        max={max}
                                        lastUpdated={lastUpdated}
                                        facilityId={facility}
                                        headerAction={
                                            !showClosedFacilityMode
                                                ? (
                                                    <ThemeModeToggle
                                                        themeMode={themeMode}
                                                        onThemeModeChange={onThemeModeChange}
                                                    />
                                                )
                                                : null
                                        }
                                        occupancyThresholds={occupancyThresholds}
                                        onOpenAlerts={() => setIsCrowdAlertOpen(true)}
                                    />
                                </Box>
                            )}

                            <Box component={motion.div} variants={facilityItemVariants}>
                                <ScheduleStatusCard
                                    status={scheduleStatus}
                                    nextOpenLabel={nextOpenLabel}
                                    footerAction={
                                        showClosedFacilityMode
                                            ? (
                                                <ThemeModeToggle
                                                    themeMode={themeMode}
                                                    onThemeModeChange={onThemeModeChange}
                                                />
                                            )
                                            : null
                                    }
                                />
                            </Box>

                            {!showClosedFacilityMode && warningText && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <Alert severity="warning" variant="outlined" sx={{borderRadius: 2}}>
                                        {warningText}
                                    </Alert>
                                </Box>
                            )}

                            {canShowDailyForecastCard && visibleForecastDays.length > 0 && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <ForecastWindowsCard
                                        day={selectedForecastDay}
                                        comparisonDays={visibleForecastDays}
                                        schedule={activeSchedule}
                                        occupancyThresholds={occupancyThresholds}
                                        dayOffset={resolvedForecastDayOffset}
                                        titleDayOffset={
                                            showClosedFacilityMode
                                                ? resolvedForecastDayOffset + 1
                                                : resolvedForecastDayOffset
                                        }
                                        totalDays={visibleForecastDays.length}
                                        canPrev={resolvedForecastDayOffset > 0}
                                        canNext={resolvedForecastDayOffset < Math.min(3, visibleForecastDays.length - 1)}
                                        onPrev={() => setForecastDaySelection({
                                            key: forecastDisplayKey,
                                            offset: Math.max(0, resolvedForecastDayOffset - 1),
                                        })}
                                        onNext={() =>
                                            setForecastDaySelection({
                                                key: forecastDisplayKey,
                                                offset: Math.min(
                                                    Math.min(3, visibleForecastDays.length - 1),
                                                    resolvedForecastDayOffset + 1
                                                ),
                                            })
                                        }
                                        isLoading={isForecastLoading}
                                        error={forecastError}
                                    />
                                </Box>
                            )}

                            {!showClosedFacilityMode && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <Suspense
                                        fallback={(
                                            <Box sx={{display: "grid", placeItems: "center", py: 4}}>
                                                <CircularProgress size={24} thickness={5}/>
                                            </Box>
                                        )}
                                    >
                                        <FloorHeatMapCard
                                            facilityId={facility}
                                            locations={activeData.locations}
                                            occupancyThresholds={occupancyThresholds}
                                            locationOccupancyThresholds={forecastLocationOccupancyThresholds}
                                        />
                                    </Suspense>
                                </Box>
                            )}

                            {!showClosedFacilityMode && (
                                <Stack
                                    component={motion.div}
                                    variants={facilityItemVariants}
                                    spacing={sectionBlockGap}
                                    sx={{mt: sectionBlockGap, mb: sectionBlockGap}}
                                >
                                    {sectionConfigs.map((section) => (
                                        <Box key={section.title} component={motion.div} variants={facilityItemVariants}>
                                            <SectionCommandCenter
                                                title={section.title}
                                                ids={[...section.ids]}
                                                locations={activeData.locations}
                                                forecast={sectionForecastMap[normalizeSectionTitle(section.title)]}
                                                occupancyThresholds={
                                                    sectionOccupancyThresholds[normalizeSectionTitle(section.title)]
                                                    ?? occupancyThresholds
                                                }
                                                locationOccupancyThresholds={forecastLocationOccupancyThresholds}
                                            />
                                        </Box>
                                    ))}
                                </Stack>
                            )}

                            {!showClosedFacilityMode && hasOtherSectionLocations && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <SectionSummaryOther
                                        title={dashboardConfig.otherTitle}
                                        exclude={knownIds}
                                        locations={activeData.locations}
                                        occupancyThresholds={occupancyThresholds}
                                        locationOccupancyThresholds={forecastLocationOccupancyThresholds}
                                    />
                                </Box>
                            )}

                            <Box component={motion.div} variants={facilityItemVariants}>
                                <FacilityHoursBlock
                                    facilityName={activeData.facilityName}
                                    schedule={activeSchedule}
                                    isLoading={isFacilityHoursLoading}
                                    error={facilityHoursError}
                                />
                            </Box>
                        </Box>
                    )}
                </AnimatePresence>

                <AlertsPanel
                    open={isCrowdAlertOpen}
                    onClose={() => setIsCrowdAlertOpen(false)}
                    facility={facility}
                    sections={alertSections}
                    useDesktopModal={useDesktopAlertsModal}
                    isStandalonePwa={isStandalonePwa}
                    isTouchCapable={isTouchCapable}
                />

                <InstallGuideDialog
                    open={!isStandalonePwa && isInstallGuideOpen}
                    onClose={() => setIsInstallGuideOpen(false)}
                />

                <AppFooter
                    isStandalonePwa={isStandalonePwa}
                    onOpenInstallGuide={() => setIsInstallGuideOpen(true)}
                />
            </Container>
        </Box>
    );
}
