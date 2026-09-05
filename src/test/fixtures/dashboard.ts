import type {FacilityId, FacilityPayload, Location} from "../../lib/types/facility";
import type {ForecastDay, ForecastHour} from "../../lib/types/forecast";
import type {FacilityScheduleResponse} from "../../lib/api/schemas";
import type {OccupancyThresholds} from "../../shared/utils/styles";
import {computeOccupancySummary, type OccupancySummary} from "../../shared/occupancy/computeOccupancySummary";
import {FACILITY_DISPLAY_NAMES} from "../../lib/config/facilitySections";
import type {DashboardSelectorInput, DashboardState} from "../../features/dashboard/dashboardTypes";
import {buildDashboardViewModel} from "../../features/dashboard/dashboardSelectors";

export const fixtureNowTs = Date.parse("2026-08-31T12:00:00Z");
const fetchedAt = "2026-08-31T11:59:00Z";
export const fixtureThresholds: OccupancyThresholds = {lowMax: 35, peakMin: 70};

const locationRows: Record<FacilityId, readonly [number, string, number][]> = {
    1186: [
        [5761, "Power House", 0], [5764, "Pool", 0],
        [5760, "Level 1 Fitness", 1], [7089, "Courts 1 & 2", 1],
        [5762, "Level 2 Fitness", 2], [5758, "Level 3 Fitness", 3],
        [7090, "Courts 3-6", 3], [5766, "Courts 7 & 8", 3],
        [5763, "Track", 4], [5753, "Racquetball Court 1", 4], [5754, "Racquetball Court 2", 4],
    ],
    1656: [
        [8717, "Level 1 Fitness", 1], [8720, "Courts 1 & 2", 1],
        [8698, "Courts 5-8", 1], [8716, "Cove Pool", 1], [10550, "Ice Center", 1],
        [8705, "Level 2 Fitness", 2], [8700, "Level 3 Fitness", 3], [8714, "Courts 3 & 4", 3],
        [8694, "Track", 4], [8699, "Level 4 Fitness", 4], [8696, "Orbit", 4], [8695, "Skybox", 4],
    ],
};

const makeLocations = (facilityId: FacilityId): Location[] => locationRows[facilityId].map(
    ([locationId, locationName, floor]) => ({
        facilityId, locationId, locationName, floor,
        isClosed: false, currentCapacity: 20, maxCapacity: 100,
        lastUpdated: fetchedAt, fetchedAt,
    })
);

export const fixtureLocationsByFacility: Record<FacilityId, Location[]> = {
    1186: makeLocations(1186),
    1656: makeLocations(1656),
};
export const fixtureLocations: Location[] = fixtureLocationsByFacility[1186];

const makeLive = (facilityId: FacilityId): FacilityPayload => {
    const locations = fixtureLocationsByFacility[facilityId];
    const floors: Record<number, Location[]> = {};
    for (const location of locations) (floors[location.floor] ??= []).push(location);
    return {facilityId, facilityName: FACILITY_DISPLAY_NAMES[facilityId], floors, locations, liveDataSource: "facility_api"};
};
export const fixtureLiveByFacility: Record<FacilityId, FacilityPayload> = {1186: makeLive(1186), 1656: makeLive(1656)};
export const fixtureLive: FacilityPayload = fixtureLiveByFacility[1186];

const makeSchedule = (facilityId: FacilityId): FacilityScheduleResponse => ({
    generatedAt: fetchedAt,
    sourceSite: "https://recwell.wisc.edu/",
    facilityId,
    facilityName: FACILITY_DISPLAY_NAMES[facilityId],
    slug: facilityId === 1186 ? "nick" : "bakke",
    url: facilityId === 1186 ? "https://recwell.wisc.edu/nick/" : "https://recwell.wisc.edu/bakke/",
    resolvedUrl: facilityId === 1186 ? "https://recwell.wisc.edu/nick/" : "https://recwell.wisc.edu/bakke/",
    status: "ok", source: "direct_html", sourceModifiedGmt: null,
    sections: [{title: "Building Hours", rows: [{label: "Monday - Sunday", hours: "6am - 11pm"}], note: null}],
    sourceFetchedAt: fetchedAt, lastSuccessfulAt: fetchedAt,
    stale: false, error: null, errorCategory: null, updatedAt: fetchedAt,
});
export const fixtureScheduleByFacility: Record<FacilityId, FacilityScheduleResponse> = {
    1186: makeSchedule(1186), 1656: makeSchedule(1656),
};
export const fixtureSchedule: FacilityScheduleResponse = fixtureScheduleByFacility[1186];

const hoursForDate = (date: string): ForecastHour[] => [
    {hourStart: `${date}T07:30:00-05:00`, expectedCount: 10, expectedPct: 0.1},
    {hourStart: `${date}T08:00:00-05:00`, expectedCount: 20, expectedPct: 0.2},
    {hourStart: `${date}T08:30:00-05:00`, expectedCount: 40, expectedPct: 0.4},
    {hourStart: `${date}T09:00:00-05:00`, expectedCount: 50, expectedPct: 0.5},
    {hourStart: `${date}T10:00:00-05:00`, expectedCount: 60, expectedPct: 0.6},
];
export const fixtureForecastDays: ForecastDay[] = [
    ["2026-08-31", "Monday"], ["2026-09-01", "Tuesday"], ["2026-09-02", "Wednesday"],
].map(([date, dayName]) => ({
    date, dayName,
    totalHours: hoursForDate(date).map((hour) => ({...hour, expectedCount: hour.expectedCount * 2})),
    categories: [
        {key: "fitness floors", title: "🏋️ Fitness_Floors", maxCapacity: 100, hours: hoursForDate(date)},
        {key: "basketball courts", title: "Basketball Courts", maxCapacity: 100, hours: hoursForDate(date)},
        {key: "swimming pool", title: "Swimming Pool", maxCapacity: 100, hours: hoursForDate(date)},
    ],
    bestWindows: [{start: `${date}T08:00:00-05:00`, end: `${date}T09:00:00-05:00`, expectedAvg: 60}],
    avoidWindows: [{start: `${date}T10:00:00-05:00`, end: `${date}T11:00:00-05:00`, expectedAvg: 120}],
    crowdBands: [{start: `${date}T08:00:00-05:00`, end: `${date}T09:00:00-05:00`, level: "low"}],
}));
export const fixtureCategoryForecastDays: ForecastDay[] = fixtureForecastDays.map((day) => ({...day, totalHours: []}));

export const liveSummary: OccupancySummary = computeOccupancySummary(fixtureLocations, {nowMs: fixtureNowTs});
export const closedSummary: OccupancySummary = computeOccupancySummary(
    fixtureLocations.map((location) => ({...location, isClosed: true})), {nowMs: fixtureNowTs}
);
export const partialSummary: OccupancySummary = computeOccupancySummary([
    fixtureLocations[9], {...fixtureLocations[10], fetchedAt: "2026-08-31T11:00:00Z"},
], {nowMs: fixtureNowTs});

export const fixtureDashboardInput: DashboardSelectorInput = {
    facility: 1186, nowTs: fixtureNowTs, data: fixtureLive,
    isLoading: false, error: null, liveDataSource: "facility_api", liveOutageState: "none",
    hasPendingLiveRetry: false, cacheTimestampMs: null, isOffline: false,
    forecastDays: fixtureForecastDays, forecastOccupancyThresholds: fixtureThresholds,
    forecastSectionOccupancyThresholds: {}, forecastLocationOccupancyThresholds: {},
    forecastHourBounds: {startHour: null, endHour: null}, forecastError: null,
    isForecastLoading: false, hasPendingForecastRetry: false,
    activeSchedule: fixtureSchedule, isFacilityHoursLoading: false, facilityHoursError: null,
    hasPendingScheduleRetry: false, forecastDaySelection: {key: null, offset: 0},
    predictionOverrideEnabled: false, closureOverrideEnabled: false,
};

export function createDashboardStateForTest(overrides: Partial<DashboardState> = {}): DashboardState {
    const facility = overrides.facility ?? 1186;
    const input: DashboardSelectorInput = {
        ...fixtureDashboardInput, facility,
        data: fixtureLiveByFacility[facility], activeSchedule: fixtureScheduleByFacility[facility],
        ...overrides,
    };
    const noop = () => {};
    return {
        ...input, view: buildDashboardViewModel(input),
        liveRefreshKey: 0, forecastRefreshKey: 0, scheduleRefreshKey: 0,
        liveStatus: "idle", lastManualRefresh: 0, debugEnabled: false, debugNowMs: null,
        isCrowdAlertOpen: false, isInstallGuideOpen: false,
        isStandalonePwa: false, isTouchCapable: false, enablePullToRefresh: false,
        pullDistance: 0, isPulling: false, isReadyToRefresh: false, showPullIndicator: false,
        handleFacilitySelect: noop, manualRefresh: noop, setForecastDaySelection: noop,
        setIsCrowdAlertOpen: noop, setIsInstallGuideOpen: noop,
        setPredictionOverride: noop, setClosureOverride: noop, setDebugNowOverride: () => null,
        resetPullGesture: noop, handleTouchStart: noop, handleTouchMove: noop, handleTouchEnd: noop,
        ...overrides,
    };
}
