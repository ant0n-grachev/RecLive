import {buildDashboardViewModel, buildSectionForecastMap, deriveForecastBounds} from "./dashboardSelectors";
import {
    fixtureDashboardInput, fixtureForecastDays, fixtureLiveByFacility,
    fixtureSchedule, fixtureScheduleByFacility, fixtureThresholds,
    fixtureLocationsByFacility, liveSummary, closedSummary, partialSummary,
    createDashboardStateForTest,
} from "../../test/fixtures/dashboard";
import {facilityScheduleSchema} from "../../lib/api/schemas";

describe("dashboard selection", () => {
    it("skips empty forecast days without hiding populated later days or using them for today's chips", () => {
        const view = buildDashboardViewModel({...fixtureDashboardInput, forecastDays: [
            {...fixtureForecastDays[0], totalHours: [], categories: [], crowdBands: [], bestWindows: [], avoidWindows: []},
            fixtureForecastDays[1], fixtureForecastDays[2],
        ]});
        expect(view.visibleForecastDays.map((day) => day.date)).toEqual(["2026-09-01", "2026-09-02"]);
        expect(view.selectedForecastDay?.date).toBe("2026-09-01");
        expect(view.sectionForecastMap).toEqual({});
    });

    it("skips days whose forecast information falls entirely outside opening hours", () => {
        const view = buildDashboardViewModel({...fixtureDashboardInput, forecastDays: [
            {...fixtureForecastDays[0], totalHours: [], categories: [], crowdBands: [], avoidWindows: [],
                bestWindows: [{start: "2026-08-31T01:00:00-05:00", end: "2026-08-31T02:00:00-05:00", expectedAvg: 10}]},
            fixtureForecastDays[1],
        ]});
        expect(view.visibleForecastDays.map((day) => day.date)).toEqual(["2026-09-01"]);
    });

    it("never averages tomorrow's same hour into today's section chips", () => {
        const day = {date: "2026-08-31", dayName: "Monday", categories: [{key: "fitness floors", title: "Fitness Floors", hours: [
            {hourStart: "2026-08-31T09:00:00-05:00", expectedCount: 20},
            {hourStart: "2026-09-01T09:00:00-05:00", expectedCount: 900},
        ]}]};
        const result = buildSectionForecastMap(day, Date.parse("2026-08-31T08:00:00-05:00"), [], false);
        expect(result["fitness floors"].map(({expectedCount}) => expectedCount)).toEqual([20]);
    });
    it("keeps active facility summaries and ordered full alert sections", () => {
        const view = buildDashboardViewModel(fixtureDashboardInput);
        expect(view.facilitySummary).toMatchObject({status: "live", count: 220, coverage: 1, percent: 20});
        expect(view.alertSections.map(({key}) => key)).toEqual([
            "overall", "fitness floors", "basketball courts", "racquetball courts", "running track", "swimming pool",
        ]);
        expect(view.alertSections[0]).toEqual({key: "overall", label: "Entire Facility", summary: view.facilitySummary});
        expect(view.sectionSummaries.get("fitness floors")).toMatchObject({count: 80, observedCapacity: 400, status: "live"});
        expect(view.warning.kind).toBe("none");
        expect(view.visibleForecastDays.map((day) => day.date)).toEqual(["2026-08-31", "2026-09-01", "2026-09-02"]);
        expect(view.selectedForecastDay?.date).toBe("2026-08-31");
        expect(view.otherSummary).toBeNull();
    });

    it("does not use a retained snapshot belonging to the previous facility", () => {
        const view = buildDashboardViewModel({...fixtureDashboardInput, facility: 1656});
        expect(view.activeData).toBeNull();
        expect(view.facilitySummary).toMatchObject({status: "unknown", count: null});
        expect(view.alertSections.every(({summary}) => summary.count === null)).toBe(true);
        expect(view.alertSections.map(({key}) => key)).toEqual([
            "overall", "fitness floors", "basketball courts", "running track", "swimming pool",
            "rock climbing", "ice skating", "esports room", "sports simulators",
        ]);
    });

    it("aggregates Other only from unconfigured locations", () => {
        const other = {...fixtureDashboardInput.data!.locations[0], locationId: 99999, currentCapacity: 7};
        const view = buildDashboardViewModel({...fixtureDashboardInput, data: {
            ...fixtureLiveByFacility[1186], locations: [...fixtureLiveByFacility[1186].locations, other],
        }});
        expect(view.hasOtherSectionLocations).toBe(true);
        expect(view.otherSummary).toMatchObject({count: 7, observedCapacity: 100, status: "live"});
        expect(view.sectionSummaries.get("fitness floors")?.count).toBe(80);
    });

    it("resolves selected-day keys, clamps offsets, and resets when the displayed dates change", () => {
        const key = "active:2026-08-31|2026-09-01|2026-09-02";
        const select = (key: string, offset: number) => buildDashboardViewModel({
            ...fixtureDashboardInput, forecastDaySelection: {key, offset},
        });
        expect(select(key, 1).selectedForecastDay?.date).toBe("2026-09-01");
        expect(select(key, 9).resolvedForecastDayOffset).toBe(2);
        expect(select("old dates", 1).selectedForecastDay?.date).toBe("2026-08-31");
        expect(buildDashboardViewModel({...fixtureDashboardInput, forecastDays: []}).selectedForecastDay).toBeNull();
    });

    it("filters closed forecast days and shows tomorrow only when the next opening is tomorrow", () => {
        const schedule = {...fixtureSchedule, sections: [{title: "Building Hours", rows: [
            {label: "Monday", hours: "Closed"}, {label: "Tuesday", hours: "6am - 11pm"},
            {label: "Wednesday", hours: "Closed"},
        ]}]};
        const view = buildDashboardViewModel({...fixtureDashboardInput, activeSchedule: schedule});
        expect(view.showClosedFacilityMode).toBe(true);
        expect(view.isExpectedOpenTomorrow).toBe(true);
        expect(view.visibleForecastDays.map((day) => day.date)).toEqual(["2026-09-01"]);
        expect(view.forecastDisplayKey).toBe("closed:2026-09-01");
        expect(view.canShowClosedTomorrowForecast).toBe(true);
        expect(view.nextOpenLabel).toBe("in 23 hours");
        const laterSchedule = {...schedule, sections: [{title: "Building Hours", rows: [
            {label: "Monday - Tuesday", hours: "Closed"}, {label: "Wednesday", hours: "6am - 11pm"},
        ]}]};
        const later = buildDashboardViewModel({...fixtureDashboardInput, activeSchedule: laterSchedule});
        expect(later.visibleForecastDays).toEqual([]);
        expect(later.canShowDailyForecastCard).toBe(false);
    });

    it("keeps outage warning priority, cache age, and independent daily/hourly forecast gates", () => {
        const view = buildDashboardViewModel({...fixtureDashboardInput,
            isOffline: true, liveOutageState: "cache", liveDataSource: "cache",
            forecastError: "Unavailable", cacheTimestampMs: fixtureDashboardInput.nowTs - 120000,
        });
        expect(view.warning.kind).toBe("offline_cache");
        expect(view.warningText).toContain("Saved snapshot is 2 minutes old.");
        expect(view.sectionForecastMap).toEqual({});
        expect(view.canShowDailyForecastCard).toBe(false);
        const early = buildDashboardViewModel({...fixtureDashboardInput,
            forecastHourBounds: {startHour: 9, endHour: 21},
        });
        expect(early.canShowHourlyRoomForecasts).toBe(false);
        expect(early.canShowDailyForecastCard).toBe(true);
        const override = buildDashboardViewModel({...fixtureDashboardInput,
            predictionOverrideEnabled: true, forecastError: "Unavailable",
        });
        expect(override.warningText).toBeNull();
        expect(override.canShowDailyForecastCard).toBe(true);
        expect(override.canShowHourlyRoomForecasts).toBe(true);
    });

    it("keeps opening-time freshness warnings ahead of forecast errors", () => {
        const view = buildDashboardViewModel({...fixtureDashboardInput, forecastError: "Unavailable",
            activeSchedule: {...fixtureSchedule, sections: [{title: "Building Hours", rows: [
                {label: "Monday - Sunday", hours: "7am - 11pm"},
            ]}]},
        });
        expect(view.warning.kind).toBe("scheduled_open_not_live");
        expect(view.canShowHourlyRoomForecasts).toBe(true);
        expect(view.canShowDailyForecastCard).toBe(false);
    });

    it("retains facility, section and weighted location threshold precedence", () => {
        const view = buildDashboardViewModel({...fixtureDashboardInput,
            forecastOccupancyThresholds: null,
            forecastLocationOccupancyThresholds: {5761: {lowMax: 20, peakMin: 60}, 5760: {lowMax: 40, peakMin: 80}},
            forecastSectionOccupancyThresholds: {"basketball courts": {lowMax: 10, peakMin: 90}},
        });
        expect(view.occupancyThresholds).toEqual({lowMax: 30, peakMin: 70});
        expect(view.sectionOccupancyThresholds["fitness floors"]).toEqual({lowMax: 30, peakMin: 70});
        expect(view.sectionOccupancyThresholds["basketball courts"]).toEqual({lowMax: 10, peakMin: 90});
        expect(view.sectionOccupancyThresholds["running track"]).toEqual({lowMax: 30, peakMin: 70});
        expect(buildDashboardViewModel(fixtureDashboardInput).occupancyThresholds).toEqual(fixtureThresholds);
    });
});

describe("forecast helpers", () => {
    it("derives bounds from valid category timestamps, preserving category-only inference", () => {
        expect(deriveForecastBounds(fixtureForecastDays)).toEqual({startHour: 7, endHour: 10});
        expect(deriveForecastBounds([{date: "2026-08-31", dayName: "Monday", totalHours: fixtureForecastDays[0].totalHours}]))
            .toEqual({startHour: null, endHour: null});
        expect(deriveForecastBounds([{date: "2026-08-31", dayName: "Monday", categories: [{
            key: "fitness floors", title: "Fitness Floors", hours: [{hourStart: "bad", expectedCount: 4}],
        }]}])).toEqual({startHour: null, endHour: null});
        expect(buildDashboardViewModel({...fixtureDashboardInput, forecastHourBounds: {startHour: 21, endHour: 6}}).forecastHourBounds)
            .toEqual({startHour: 6, endHour: 21});
    });

    it("averages future half-hours, omits current-hour points, and clips to open windows", () => {
        const map = buildSectionForecastMap(fixtureForecastDays[0], fixtureDashboardInput.nowTs,
            [{startMinutes: 8 * 60, endMinutes: 10 * 60}], true);
        expect(map["fitness floors"]).toEqual([
            {hourStart: "2026-08-31T08:00:00-05:00", expectedCount: 30},
            {hourStart: "2026-08-31T09:00:00-05:00", expectedCount: 50},
        ]);
        expect(map["swimming pool"]).toBeUndefined();
        expect(buildSectionForecastMap(fixtureForecastDays[0], fixtureDashboardInput.nowTs, [], true)).toEqual({});
        expect(buildSectionForecastMap(null, fixtureDashboardInput.nowTs, [], false)).toEqual({});
        expect(buildSectionForecastMap(fixtureForecastDays[0], fixtureDashboardInput.nowTs, [], false)["fitness floors"])
            .toHaveLength(3);
    });

    it("rejects malformed and past points and clamps negative samples before averaging", () => {
        const day = {date: "2026-08-31", dayName: "Monday", categories: [{
            key: "fitness floors", title: "Fitness Floors", hours: [
                {hourStart: "bad", expectedCount: 900},
                {hourStart: "2026-08-31T06:00:00-05:00", expectedCount: 900},
                {hourStart: "2026-08-31T07:30:00-05:00", expectedCount: 900},
                {hourStart: "2026-08-31T08:30:00-05:00", expectedCount: -10},
                {hourStart: "2026-08-31T08:00:00-05:00", expectedCount: 30},
            ],
        }]};
        expect(buildSectionForecastMap(day, fixtureDashboardInput.nowTs, [], false)).toEqual({
            "fitness floors": [{hourStart: "2026-08-31T08:00:00-05:00", expectedCount: 15}],
        });
    });
});

describe("shared typed dashboard fixtures", () => {
    it.each([1186, 1656] as const)("uses coherent per-facility live rows and strict fresh schedule for %s", (facility) => {
        const payload = fixtureLiveByFacility[facility];
        expect(payload.locations.every((location) => location.facilityId === facility)).toBe(true);
        expect(Object.values(payload.floors).flat()).toEqual(payload.locations);
        expect(payload.locations).toEqual(fixtureLocationsByFacility[facility]);
        expect(facilityScheduleSchema.safeParse(fixtureScheduleByFacility[facility]).success).toBe(true);
        const state = createDashboardStateForTest({facility});
        expect(state.view.activeData?.facilityId).toBe(facility);
        expect(state.liveStatus).toBe("idle");
    });

    it("uses real occupancy contracts for live, closed and partial summaries", () => {
        expect(liveSummary).toMatchObject({status: "live", count: 220, coverage: 1});
        expect(closedSummary).toMatchObject({status: "closed", count: null, percent: null, expectedOpenCapacity: 0});
        expect(partialSummary).toMatchObject({status: "partial", coverage: 0.5, count: 20, percent: 20});
    });
});
