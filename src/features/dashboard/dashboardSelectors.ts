import type {ForecastDay, ForecastHour} from "../../lib/types/forecast";
import type {AlertSectionOption} from "../alerts/alertTypes";
import type {ForecastHourBounds} from "../../app/hooks/useForecastData";
import type {DashboardSelectorInput, DashboardViewModel} from "./dashboardTypes";
import {FACILITY_DASHBOARD_CONFIG, FACILITY_KNOWN_IDS, isSectionRow} from "../../facilities/constants";
import {getChicagoTimestampMs, getChicagoHour, isWithinChicagoHours} from "../../shared/utils/chicagoTime";
import {combineOccupancyThresholds, type OccupancyThresholds} from "../../shared/utils/styles";
import {computeOccupancySummary} from "../../shared/occupancy/computeOccupancySummary";
import {resolveDashboardWarning} from "../../app/warningStatus";
import {getFacilityNextOpenTimestamp, getFacilityOpenStatus, getFacilityOpenWindowsForDate, type FacilityOpenWindow} from "../../shared/utils/facilityScheduleStatus";
import {buildForecastDisplaySlots, clipBandsToWorkingHours, filterWindowsToWorkingHours} from "../forecast/forecastTime";

const FORECAST_VISIBLE_SECTIONS = new Set(["fitness floors", "basketball courts"]);
const MS_PER_HOUR = 60 * 60 * 1000;
const MS_PER_DAY = 24 * MS_PER_HOUR;
interface ChicagoDateParts {
    year: number;
    month: number;
    day: number;
    hour: number;
    minute: number;
}

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

export const normalizeSectionTitle = (title: string): string =>
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

export const deriveForecastBounds = (days: readonly ForecastDay[]): ForecastHourBounds => {
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

export const buildSectionForecastMap = (
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
                const hourDate = getChicagoDateParts(new Date(hourTs));
                const today = getChicagoDateParts(new Date(nowTs));
                if (!hourDate || !today || chicagoDayKey(hourDate) !== chicagoDayKey(today)) {
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

export function buildDashboardViewModel(input: DashboardSelectorInput): DashboardViewModel {
    const {
        facility,
        nowTs,
        data,
        error,
        liveDataSource,
        liveOutageState,
        cacheTimestampMs,
        isOffline,
        forecastDays,
        forecastOccupancyThresholds,
        forecastSectionOccupancyThresholds,
        forecastLocationOccupancyThresholds,
        forecastHourBounds,
        forecastError,
        activeSchedule,
        forecastDaySelection,
        predictionOverrideEnabled,
        closureOverrideEnabled,
    } = input;
    const activeData = data?.facilityId === facility ? data : null;
    const facilitySummary = computeOccupancySummary(activeData?.locations ?? [], {nowMs: nowTs});
    const inferredHourBounds = deriveForecastBounds(forecastDays);
    const resolvedHourBounds = normalizeForecastBounds({
        startHour: forecastHourBounds.startHour ?? inferredHourBounds.startHour,
        endHour: forecastHourBounds.endHour ?? inferredHourBounds.endHour,
    });
    const predictionStartHour = resolvedHourBounds.startHour ?? 0;
    const predictionEndHour = resolvedHourBounds.endHour ?? 23;
    const predictionEndHourExclusive = Math.min(24, predictionEndHour + 1);
    const todayDateKey = (() => {
        const nowParts = getChicagoDateParts(new Date(nowTs));
        return nowParts ? chicagoDayKey(nowParts) : null;
    })();
    const tomorrowDateKey = shiftDateKeyByDays(todayDateKey, 1);
    const scheduleStatus = getFacilityOpenStatus(activeSchedule, new Date(nowTs));
    const isScheduledClosedNow = scheduleStatus.state === "closed";
    const isScheduledOpenButDataNotLive = (() => {
        if (scheduleStatus.state !== "open") return false;

        const rule = scheduleStatus.matchedRule;
        const startMinutes = rule ? parseScheduleStartMinutes(rule.hours) : null;
        if (startMinutes === null) return false;

        const latestFetchedAt = facilitySummary.latestFetchedAt;
        if (!latestFetchedAt) return true;

        const latestFetchedMs = getChicagoTimestampMs(latestFetchedAt);
        if (latestFetchedMs === null) return true;

        const nowParts = getChicagoDateParts(new Date(nowTs));
        const updatedParts = getChicagoDateParts(new Date(latestFetchedMs));
        if (!nowParts || !updatedParts) return false;

        const nowDay = chicagoDayKey(nowParts);
        const updatedDay = chicagoDayKey(updatedParts);
        if (updatedDay < nowDay) return true;
        if (updatedDay > nowDay) return false;

        const updatedMinutes = updatedParts.hour * 60 + updatedParts.minute;
        return updatedMinutes < startMinutes;
    })();
    const hasAnyError = Boolean(error) || Boolean(forecastError);
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
        occupancyStatus: facilitySummary.status,
    });
    const cacheAgeText = liveDataSource === "cache" ? formatCacheAge(cacheTimestampMs, nowTs) : null;
    const baseWarningText = predictionOverrideEnabled || closureOverrideEnabled
        ? null
        : warning.text;
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
    const nextOpenTimestamp = getFacilityNextOpenTimestamp(activeSchedule, new Date(nowTs));
    const nextOpenLabel = formatNextOpenRelative(nextOpenTimestamp, nowTs);
    const nextOpenDateKey = (() => {
        const nextOpenMs = getChicagoTimestampMs(nextOpenTimestamp);
        if (nextOpenMs === null) return null;

        const nextOpenParts = getChicagoDateParts(new Date(nextOpenMs));
        return nextOpenParts ? chicagoDayKey(nextOpenParts) : null;
    })();
    const showClosedFacilityMode = hasResolvedSchedule && scheduleStatus.state === "closed" && !closureOverrideEnabled;
    const isExpectedOpenTomorrow = Boolean(tomorrowDateKey && nextOpenDateKey === tomorrowDateKey);

    const dashboardConfig = FACILITY_DASHBOARD_CONFIG[facility];
    const knownIds = FACILITY_KNOWN_IDS[facility];
    const hasOtherSectionLocations = (() => {
        if (!activeData) return false;
        const knownIdSet = new Set(knownIds);
        return activeData.locations.some((location) => !knownIdSet.has(location.locationId));
    })();
    const sectionConfigs = dashboardConfig.sections.flatMap((layout) => (isSectionRow(layout) ? [...layout] : [layout]));
    const alertSections: AlertSectionOption[] = (() => {
        const overall: AlertSectionOption = {
            key: "overall",
            label: "Entire Facility",
            summary: facilitySummary,
        };

        const bySection = sectionConfigs.map((section) => {
            const idSet = new Set<number>(section.ids);
            const sectionLocations = activeData
                ? activeData.locations.filter((location) => idSet.has(location.locationId))
                : [];

            return {
                key: normalizeSectionTitle(section.title),
                label: section.title,
                summary: computeOccupancySummary(sectionLocations, {nowMs: nowTs}),
            };
        });

        return [overall, ...bySection];
    })();
    const visibleForecastDays = (() => {
        const isDisplayableForecastDay = (forecastDay: ForecastDay) => {
            const windows = getFacilityOpenWindowsForDate(activeSchedule, forecastDay.date);
            const enforceHours = Boolean(activeSchedule && forecastDay.date);
            const bands = forecastDay.crowdBands ?? [];
            return buildForecastDisplaySlots(forecastDay, windows, enforceHours, forecastOccupancyThresholds, bands, nowTs).length > 0
                || clipBandsToWorkingHours(bands, windows, enforceHours, forecastDay.date).length > 0
                || filterWindowsToWorkingHours(forecastDay.bestWindows ?? [], windows, enforceHours, forecastDay.date).length > 0
                || filterWindowsToWorkingHours(forecastDay.avoidWindows ?? [], windows, enforceHours, forecastDay.date).length > 0;
        };
        const openForecastDays = forecastDays.filter(isDisplayableForecastDay);

        if (showClosedFacilityMode) {
            if (!tomorrowDateKey || !isExpectedOpenTomorrow) {
                return [];
            }

            const tomorrowIndex = forecastDays.findIndex((forecastDay) => forecastDay.date === tomorrowDateKey);
            const daysFromTomorrow = tomorrowIndex >= 0
                ? forecastDays.slice(tomorrowIndex)
                : forecastDays.slice(1);
            return daysFromTomorrow.filter(isDisplayableForecastDay).slice(0, 4);
        }

        return openForecastDays.slice(0, 4);
    })();
    const forecastDisplayKey = `${showClosedFacilityMode ? "closed" : "active"}:${visibleForecastDays.map((day) => day.date).join("|")}`;
    const todayForecastDay = forecastDays.find((day) => day.date === todayDateKey) ?? null;
    const activeForecastDayOffset = forecastDaySelection.key === forecastDisplayKey
        ? forecastDaySelection.offset
        : 0;
    const resolvedForecastDayOffset = Math.min(
        activeForecastDayOffset,
        Math.max(0, visibleForecastDays.length - 1)
    );
    const selectedForecastDay =
        visibleForecastDays[resolvedForecastDayOffset] ?? null;
    const canShowClosedTomorrowForecast = showClosedFacilityMode
        && isExpectedOpenTomorrow
        && !forecastError
        && visibleForecastDays.length > 0;
    const canShowActiveDailyForecast = !showClosedFacilityMode && (predictionOverrideEnabled || closureOverrideEnabled || (
        !forecastError
        && warning.kind !== "offline_cache"
        && warning.kind !== "total_outage_cache"
        && warning.kind !== "prediction_unavailable"
        && warning.kind !== "facility_fallback"
    ));
    const canShowDailyForecastCard = canShowClosedTomorrowForecast || canShowActiveDailyForecast;
    const occupancyThresholds = forecastOccupancyThresholds ?? combineOccupancyThresholds(
        (activeData?.locations ?? []).map((location) => ({
            thresholds: forecastLocationOccupancyThresholds[location.locationId],
            weight: typeof location.maxCapacity === "number" && Number.isFinite(location.maxCapacity)
                ? Math.max(0, location.maxCapacity)
                : 0,
        }))
    );
    const sectionOccupancyThresholds = (() => {
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
                        weight: typeof location.maxCapacity === "number" && Number.isFinite(location.maxCapacity)
                            ? Math.max(0, location.maxCapacity)
                            : 0,
                    }))
            ) ?? occupancyThresholds;

            if (sectionThresholds) {
                thresholdsBySection[sectionKey] = sectionThresholds;
            }
        }

        return thresholdsBySection;
    })();
    const enforceSectionForecastWorkingHours = Boolean(activeSchedule && todayForecastDay?.date);
    const sectionForecastOpenWindows = getFacilityOpenWindowsForDate(activeSchedule, todayForecastDay?.date ?? null);

    const sectionForecastMap = (
        canShowHourlyRoomForecasts
            ? buildSectionForecastMap(
                todayForecastDay,
                nowTs,
                sectionForecastOpenWindows,
                enforceSectionForecastWorkingHours
            )
            : {}
    );

    const sectionSummaries = new Map(alertSections.slice(1).map(({key, summary}) => [key, summary]));
    const otherLocations = (activeData?.locations ?? []).filter((location) => !knownIds.includes(location.locationId));
    const otherSummary = otherLocations.length > 0 ? computeOccupancySummary(otherLocations, {nowMs: nowTs}) : null;
    return {
        activeData,
        facilitySummary,
        sectionSummaries,
        otherSummary,
        alertSections,
        dashboardConfig,
        knownIds,
        sectionConfigs,
        hasOtherSectionLocations,
        visibleForecastDays,
        todayForecastDay,
        selectedForecastDay,
        forecastDisplayKey,
        resolvedForecastDayOffset,
        todayDateKey,
        tomorrowDateKey,
        nextOpenDateKey,
        nextOpenLabel,
        scheduleStatus,
        showClosedFacilityMode,
        isExpectedOpenTomorrow,
        canShowClosedTomorrowForecast,
        canShowActiveDailyForecast,
        canShowDailyForecastCard,
        canShowHourlyRoomForecasts,
        warning,
        warningText,
        occupancyThresholds,
        sectionOccupancyThresholds,
        sectionForecastMap,
        sectionForecastOpenWindows,
        enforceSectionForecastWorkingHours,
        forecastHourBounds: resolvedHourBounds,
    };
}
