import type {ForecastDay} from "../../lib/types/forecast";
import {
    fixtureCategoryForecastDays,
    fixtureForecastDays,
    fixtureSchedule,
    fixtureThresholds,
} from "../../test/fixtures/dashboard";
import {buildCrowdBandsFromDisplaySlots} from "./forecastBands";
import {buildForecastDisplaySlots} from "./forecastTime";
import {buildHistogramModel, type ForecastDisplaySlot} from "./forecastHistogram";
import {getFacilityOpenWindowsForDate} from "../../shared/utils/facilityScheduleStatus";

const fixtureSlots: ForecastDisplaySlot[] = [
    {startMinute: 360, endMinute: 390, startTs: 0, endTs: 1, count: 40, percent: 0.20, source: "actual", level: "low"},
    {startMinute: 390, endMinute: 420, startTs: 1, endTs: 2, count: 44, percent: 0.22, source: "actual", level: "low"},
    {startMinute: 420, endMinute: 450, startTs: 2, endTs: 3, count: 100, percent: 0.50, source: "predicted", level: "medium"},
    {startMinute: 450, endMinute: 480, startTs: 3, endTs: 4, count: 104, percent: 0.52, source: "predicted", level: "medium"},
    {startMinute: 480, endMinute: 510, startTs: 4, endTs: 5, count: 120, percent: 0.60, source: "actual", level: "medium"},
    {startMinute: 510, endMinute: 540, startTs: 5, endTs: 6, count: 124, percent: 0.62, source: "predicted", level: "medium"},
    {startMinute: 540, endMinute: 570, startTs: 6, endTs: 7, count: 0, percent: null, source: "predicted", level: "unknown"},
    {startMinute: 570, endMinute: 600, startTs: 7, endTs: 8, count: 0, percent: null, source: "predicted", level: "unknown"},
];

it.each(["total", "category"])("retains date identity with explicitly supplied extended helper windows for %s", (path) => {
    const hours = [
        {hourStart: "2026-08-31T23:00:00-05:00", expectedCount: 20},
        {hourStart: "2026-09-01T23:00:00-05:00", expectedCount: 900},
        {hourStart: "2026-09-01T01:00:00-05:00", expectedCount: 30},
        {hourStart: "2026-08-31T01:00:00-05:00", expectedCount: 800},
        {hourStart: "2026-09-02T01:00:00-05:00", expectedCount: 700},
    ];
    const day: ForecastDay = {date: "2026-08-31", dayName: "Monday",
        ...(path === "total" ? {totalHours: hours} : {categories: [{key: "fitness floors", title: "Fitness Floors", hours}]}),
    };
    const slots = buildForecastDisplaySlots(day, [{startMinutes: 22 * 60, endMinutes: 26 * 60}], true,
        fixtureThresholds, [], Date.parse("2026-08-31T12:00:00Z"));
    expect(slots.map(({count, startMinute, startTs}) => ({count, startMinute, startTs}))).toEqual([
        {count: 20, startMinute: 1380, startTs: Date.parse("2026-08-31T23:00:00-05:00")},
        {count: 30, startMinute: 1500, startTs: Date.parse("2026-09-01T01:00:00-05:00")},
    ]);
    const unrestricted = buildForecastDisplaySlots(day, [], false, fixtureThresholds, [], Date.parse("2026-08-31T12:00:00Z"));
    expect(unrestricted.map(({count}) => count)).toEqual([800, 20]);
});

it.each(["total", "category"])("keeps real schedule post-midnight spillover on its own date for %s", (path) => {
    const schedule = {...fixtureSchedule, sections: [{title: "Building Hours", note: null,
        rows: [{label: "Monday - Sunday", hours: "10pm - 2am"}],
    }]};
    const hours = [
        {hourStart: "2026-08-31T01:00:00-05:00", expectedCount: 20},
        {hourStart: "2026-09-01T01:00:00-05:00", expectedCount: 900},
        {hourStart: "2026-08-31T23:00:00-05:00", expectedCount: 30},
    ];
    const day: ForecastDay = {date: "2026-08-31", dayName: "Monday",
        ...(path === "total" ? {totalHours: hours} : {categories: [{key: "fitness floors", title: "Fitness Floors", hours}]}),
    };
    const windows = getFacilityOpenWindowsForDate(schedule, day.date);
    expect(windows).toEqual([{startMinutes: 0, endMinutes: 120}, {startMinutes: 1320, endMinutes: 1440}]);
    const slots = buildForecastDisplaySlots(day, windows, true, fixtureThresholds, [], Date.parse("2026-08-31T12:00:00Z"));
    expect(slots.map(({count, startTs}) => ({count, startTs}))).toEqual([
        {count: 20, startTs: Date.parse("2026-08-31T01:00:00-05:00")},
        {count: 30, startTs: Date.parse("2026-08-31T23:00:00-05:00")},
    ]);
});

it("preserves actual, predicted, mixed, and unknown bar counts", () => {
    const model = buildHistogramModel(fixtureSlots, 240, fixtureThresholds);

    expect(model).toMatchObject({
        actualBarCount: 1,
        predictedBarCount: 2,
        mixedBarCount: 1,
        unknownBarCount: 1,
        yMax: 250,
    });
});

it("returns a nullable empty model and duplicates a lone half-hour within its hourly bar", () => {
    expect(buildHistogramModel([], null, fixtureThresholds)).toBeNull();

    const model = buildHistogramModel([fixtureSlots[0]], null, fixtureThresholds);
    expect(model?.bars[0]).toMatchObject({
        count: 40,
        segmentCounts: [40, 40],
        segmentLevels: ["low", "low"],
        source: "actual",
    });
});

it("rounds hourly averages, smooths isolated spikes, and honors the cross-day scale override", () => {
    const slots: ForecastDisplaySlot[] = [
        {...fixtureSlots[0], startMinute: 360, endMinute: 390, count: 40},
        {...fixtureSlots[1], startMinute: 390, endMinute: 420, count: 41},
        {...fixtureSlots[2], startMinute: 420, endMinute: 450, count: 200},
        {...fixtureSlots[3], startMinute: 450, endMinute: 480, count: 200},
        {...fixtureSlots[4], startMinute: 480, endMinute: 510, count: 42},
        {...fixtureSlots[5], startMinute: 510, endMinute: 540, count: 43},
    ];

    const model = buildHistogramModel(slots, 240, fixtureThresholds);
    expect(model?.bars.map(({rawCount, count, wasSmoothed}) => ({rawCount, count, wasSmoothed}))).toEqual([
        {rawCount: 41, count: 41, wasSmoothed: false},
        {rawCount: 200, count: 42, wasSmoothed: true},
        {rawCount: 43, count: 43, wasSmoothed: false},
    ]);
    expect(model?.yMax).toBe(250);
});

it("keeps totalHours precedence, Chicago cutoff, ratio values, and schedule clipping", () => {
    const day: ForecastDay = {
        ...fixtureForecastDays[0],
        totalHours: [
            {hourStart: "2026-08-31T07:30:00-05:00", expectedCount: 10, expectedPct: 0.10, actualCount: 70, actualPct: 0.70},
            {hourStart: "2026-08-31T08:00:00-05:00", expectedCount: 20, expectedPct: 0.20, actualCount: 80, actualPct: 0.80},
            {hourStart: "2026-08-31T08:30:00-05:00", expectedCount: 30, expectedPct: 0.30, actualCount: 90, actualPct: 0.90},
            {hourStart: "2026-08-31T09:00:00-05:00", expectedCount: 40, expectedPct: 0.40, actualCount: 100, actualPct: 1},
        ],
    };
    const slots = buildForecastDisplaySlots(
        day,
        [{startMinutes: 480, endMinutes: 570}],
        true,
        fixtureThresholds,
        day.crowdBands ?? [],
        Date.parse("2026-08-31T14:15:00Z")
    );

    expect(slots.map(({startMinute, count, percent, source, level}) => (
        {startMinute, count, percent, source, level}
    ))).toEqual([
        {startMinute: 480, count: 80, percent: 0.80, source: "actual", level: "peak"},
        {startMinute: 510, count: 90, percent: 0.90, source: "actual", level: "peak"},
        {startMinute: 540, count: 40, percent: 0.40, source: "predicted", level: "medium"},
    ]);
});

it("uses category aggregation only when totalHours is empty", () => {
    const totalSlots = buildForecastDisplaySlots(
        fixtureForecastDays[0], [], false, fixtureThresholds,
        fixtureForecastDays[0].crowdBands ?? [], Date.parse("2026-08-31T12:00:00Z")
    );
    const categorySlots = buildForecastDisplaySlots(
        fixtureCategoryForecastDays[0], [], false, fixtureThresholds,
        fixtureCategoryForecastDays[0].crowdBands ?? [], Date.parse("2026-08-31T12:00:00Z")
    );

    expect(totalSlots[0]).toMatchObject({startMinute: 450, count: 20, percent: 0.1});
    expect(categorySlots[0]).toMatchObject({startMinute: 450, count: 30, percent: null});
});

it("merges and bridges slot-derived crowd bands without changing source values", () => {
    const slots: ForecastDisplaySlot[] = [
        {...fixtureSlots[0], startTs: 0, endTs: 1_800_000, level: "low"},
        {...fixtureSlots[1], startTs: 1_800_000, endTs: 3_600_000, level: "medium"},
        {...fixtureSlots[2], startTs: 3_600_000, endTs: 5_400_000, level: "low"},
    ];

    expect(buildCrowdBandsFromDisplaySlots(slots)).toEqual([{
        start: "1970-01-01T00:00:00.000Z",
        end: "1970-01-01T01:30:00.000Z",
        level: "low",
    }]);
    expect(slots.map((slot) => slot.source)).toEqual(["actual", "actual", "predicted"]);
});
