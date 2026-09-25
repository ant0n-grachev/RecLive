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
    {startMinute: 360, endMinute: 420, startTs: 0, endTs: 3_600_000, count: 40, percent: 0.20, source: "actual", level: "low"},
    {startMinute: 420, endMinute: 480, startTs: 3_600_000, endTs: 7_200_000, count: 44, percent: 0.22, source: "actual", level: "low"},
    {startMinute: 480, endMinute: 540, startTs: 7_200_000, endTs: 10_800_000, count: 100, percent: 0.50, source: "predicted", level: "medium"},
    {startMinute: 540, endMinute: 600, startTs: 10_800_000, endTs: 14_400_000, count: 104, percent: 0.52, source: "predicted", level: "medium"},
    {startMinute: 600, endMinute: 660, startTs: 14_400_000, endTs: 18_000_000, count: 120, percent: 0.60, source: "actual", level: "medium"},
    {startMinute: 660, endMinute: 720, startTs: 18_000_000, endTs: 21_600_000, count: 124, percent: 0.62, source: "mixed", level: "medium"},
    {startMinute: 720, endMinute: 780, startTs: 21_600_000, endTs: 25_200_000, count: 0, percent: null, source: "predicted", level: "unknown"},
    {startMinute: 780, endMinute: 840, startTs: 25_200_000, endTs: 28_800_000, count: 0, percent: null, source: "predicted", level: "unknown"},
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
    const model = buildHistogramModel(fixtureSlots, 240);

    expect(model).toMatchObject({
        actualBarCount: 3,
        predictedBarCount: 4,
        mixedBarCount: 1,
        unknownBarCount: 2,
        yMax: 250,
    });
});

it("shows the available hourly interval without adding neighboring hours", () => {
    expect(buildHistogramModel([])).toBeNull();

    const model = buildHistogramModel([fixtureSlots[0]]);
    expect(model?.bars[0]).toMatchObject({
        count: 40,
        startMinute: 360,
        endMinute: 420,
        rangeLabel: "6:00 AM – 7:00 AM",
        level: "low",
        source: "actual",
    });
});

it("preserves an actual hourly spike on the shared scale", () => {
    const model = buildHistogramModel([
        {...fixtureSlots[0], count: 40},
        {...fixtureSlots[1], count: 200, level: "peak"},
        {...fixtureSlots[2], count: 42, source: "actual", level: "low"},
    ], 240);

    expect(model?.bars.map((bar) => bar.count)).toEqual([40, 200, 42]);
    expect(model?.maxCount).toBe(200);
    expect(model?.bars[1].height).toBeCloseTo(116.8);
    expect(model?.yMax).toBe(250);
});

it("uses the same hourly aggregate for bar height, count, color, and crowd ranges", () => {
    const day: ForecastDay = {date: "2026-09-24", dayName: "Thursday", totalHours: [
        {hourStart: "2026-09-24T06:00:00-05:00", expectedCount: 40, expectedPct: 0.2},
        {hourStart: "2026-09-24T06:15:00-05:00", expectedCount: 40, expectedPct: 0.2},
        {hourStart: "2026-09-24T06:30:00-05:00", expectedCount: 160, expectedPct: 0.8},
        {hourStart: "2026-09-24T06:45:00-05:00", expectedCount: 160, expectedPct: 0.8},
        {hourStart: "2026-09-24T07:00:00-05:00", expectedCount: 160, expectedPct: 0.8},
        {hourStart: "2026-09-24T07:30:00-05:00", expectedCount: 160, expectedPct: 0.8},
    ]};
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, [], Date.parse("2026-09-24T06:00:00-05:00"));
    const model = buildHistogramModel(slots, 200);

    expect(model?.bars.map(({startMinute, endMinute, count, level}) => (
        {startMinute, endMinute, count, level}
    ))).toEqual([
        {startMinute: 360, endMinute: 420, count: 100, level: "medium"},
        {startMinute: 420, endMinute: 480, count: 160, level: "peak"},
    ]);
    expect(model?.bars[0].height).toBeCloseTo(73);
    expect(model?.bars[1].height).toBeCloseTo(116.8);
    expect(buildCrowdBandsFromDisplaySlots(slots)).toEqual([
        {start: "2026-09-24T11:00:00.000Z", end: "2026-09-24T12:00:00.000Z", level: "medium"},
        {start: "2026-09-24T12:00:00.000Z", end: "2026-09-24T13:00:00.000Z", level: "peak"},
    ]);
});

it("does not classify an hourly count using a percentage from only part of the hour", () => {
    const day: ForecastDay = {date: "2026-09-24", dayName: "Thursday", totalHours: [
        {hourStart: "2026-09-24T06:00:00-05:00", expectedCount: 40, expectedPct: 0.2},
        {hourStart: "2026-09-24T06:30:00-05:00", expectedCount: 160},
    ]};
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, [], Date.parse("2026-09-24T06:00:00-05:00"));

    expect(slots).toHaveLength(1);
    expect(slots[0]).toMatchObject({count: 100, percent: null, level: "unknown"});
});

it("does not use one half-hour fallback band to color the whole category hour", () => {
    const day: ForecastDay = {date: "2026-09-24", dayName: "Thursday", categories: [{
        key: "fitness floors", title: "Fitness Floors", hours: [
            {hourStart: "2026-09-24T06:00:00-05:00", expectedCount: 40},
            {hourStart: "2026-09-24T06:30:00-05:00", expectedCount: 160},
        ],
    }]};
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, [
        {start: "2026-09-24T06:00:00-05:00", end: "2026-09-24T06:30:00-05:00", level: "low"},
        {start: "2026-09-24T06:30:00-05:00", end: "2026-09-24T07:00:00-05:00", level: "peak"},
    ], Date.parse("2026-09-24T06:00:00-05:00"));

    expect(slots).toHaveLength(1);
    expect(slots[0]).toMatchObject({count: 100, percent: null, level: "unknown"});
    expect(buildCrowdBandsFromDisplaySlots(slots)).toEqual([]);
});

it("keeps totalHours precedence, Chicago cutoff, ratio values, and schedule clipping", () => {
    const day: ForecastDay = {
        ...fixtureForecastDays[0],
        totalHours: [
            {hourStart: "2026-08-31T07:00:00-05:00", expectedCount: 999, expectedPct: 1, actualCount: 999, actualPct: 1},
            {hourStart: "2026-08-31T07:30:00-05:00", expectedCount: 10, expectedPct: 0.10, actualCount: 70, actualPct: 0.70},
            {hourStart: "2026-08-31T08:00:00-05:00", expectedCount: 20, expectedPct: 0.20, actualCount: 80, actualPct: 0.80},
            {hourStart: "2026-08-31T08:30:00-05:00", expectedCount: 30, expectedPct: 0.30, actualCount: 90, actualPct: 0.90},
            {hourStart: "2026-08-31T09:00:00-05:00", expectedCount: 40, expectedPct: 0.40, actualCount: 100, actualPct: 1},
        ],
    };
    const slots = buildForecastDisplaySlots(
        day,
        [{startMinutes: 450, endMinutes: 570}],
        true,
        fixtureThresholds,
        day.crowdBands ?? [],
        Date.parse("2026-08-31T14:15:00Z")
    );

    expect(slots.map(({startMinute, endMinute, count, percent, source, level}) => (
        {startMinute, endMinute, count, percent, source, level}
    ))).toEqual([
        {startMinute: 450, endMinute: 480, count: 70, percent: 0.7, source: "actual", level: "peak"},
        {startMinute: 480, endMinute: 540, count: 85, percent: expect.closeTo(0.85), source: "actual", level: "peak"},
        {startMinute: 540, endMinute: 570, count: 40, percent: 0.40, source: "predicted", level: "medium"},
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

    expect(totalSlots[0]).toMatchObject({startMinute: 420, count: 20, percent: 0.1});
    expect(categorySlots[0]).toMatchObject({startMinute: 420, count: 30, percent: null});
});

it.each(["total", "category"])("keeps repeated clock hours separate and uses actuals only after each hour completes in %s data", (path) => {
    const hours = ["-05:00", "-06:00"].flatMap((offset) => ["00", "15", "30", "45"].map((minute) => ({
        hourStart: `2026-11-01T01:${minute}:00${offset}`, expectedCount: 90, expectedPct: 0.45,
        actualCount: offset === "-05:00" ? 30 : 40, actualPct: offset === "-05:00" ? 0.15 : 0.2,
    })));
    const day: ForecastDay = {date: "2026-11-01", dayName: "Sunday",
        ...(path === "total" ? {totalHours: hours} : {categories: [{key: "fitness floors", title: "Fitness Floors", hours}]}),
    };
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, [], Date.parse("2026-11-01T01:15:00-06:00"));

    expect(slots.map(({startTs, count, source}) => ({startTs, count, source}))).toEqual([
        {startTs: Date.parse("2026-11-01T06:00:00Z"), count: 30, source: "actual"},
        {startTs: Date.parse("2026-11-01T07:00:00Z"), count: 90, source: "predicted"},
    ]);
});

it("derives a missing actual percentage from its reported capacity", () => {
    const day: ForecastDay = {date: "2026-09-24", dayName: "Thursday", totalHours: [{
        hourStart: "2026-09-24T08:00:00-05:00", expectedCount: 180, expectedPct: 0.9,
        actualCount: 40, expectedCapacity: 200,
    }]};
    const bands = [{start: "2026-09-24T08:00:00-05:00", end: "2026-09-24T09:00:00-05:00", level: "peak" as const}];
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, bands, Date.parse("2026-09-24T10:00:00-05:00"));

    expect(slots[0]).toMatchObject({count: 40, percent: 0.2, source: "actual", level: "low"});
});

it.each(["total", "category"])("does not borrow a forecast color for an unclassified actual count in %s data", (path) => {
    const hours = [{hourStart: "2026-09-24T08:00:00-05:00", expectedCount: 180, expectedPct: 0.9, actualCount: 40}];
    const day: ForecastDay = {date: "2026-09-24", dayName: "Thursday",
        ...(path === "total" ? {totalHours: hours} : {categories: [{key: "fitness floors", title: "Fitness Floors", hours}]}),
    };
    const bands = [{start: "2026-09-24T08:00:00-05:00", end: "2026-09-24T09:00:00-05:00", level: "peak" as const}];
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, bands, Date.parse("2026-09-24T10:00:00-05:00"));

    expect(slots[0]).toMatchObject({count: 40, percent: null, source: "actual", level: "unknown"});
    expect(buildCrowdBandsFromDisplaySlots(slots)).toEqual([]);
});

it("keeps a partially observed category slot labeled as actual plus forecast", () => {
    const day: ForecastDay = {date: "2026-09-24", dayName: "Thursday", categories: [{
        key: "fitness floors", title: "Fitness Floors", hours: [
            {hourStart: "2026-09-24T08:00:00-05:00", expectedCount: 180, actualCount: 40},
            {hourStart: "2026-09-24T08:15:00-05:00", expectedCount: 180},
        ],
    }]};
    const slots = buildForecastDisplaySlots(day, [], false, fixtureThresholds, [], Date.parse("2026-09-24T10:00:00-05:00"));

    expect(slots[0]).toMatchObject({count: 110, source: "mixed", level: "unknown"});
});

it("keeps a short different crowd level in the range summary", () => {
    const slots: ForecastDisplaySlot[] = [
        {...fixtureSlots[0], startTs: 0, endTs: 1_800_000, level: "low"},
        {...fixtureSlots[1], startTs: 1_800_000, endTs: 3_600_000, level: "medium"},
        {...fixtureSlots[2], startTs: 3_600_000, endTs: 5_400_000, level: "low"},
    ];

    expect(buildCrowdBandsFromDisplaySlots(slots)).toEqual([
        {start: "1970-01-01T00:00:00.000Z", end: "1970-01-01T00:30:00.000Z", level: "low"},
        {start: "1970-01-01T00:30:00.000Z", end: "1970-01-01T01:00:00.000Z", level: "medium"},
        {start: "1970-01-01T01:00:00.000Z", end: "1970-01-01T01:30:00.000Z", level: "low"},
    ]);
    expect(slots.map((slot) => slot.source)).toEqual(["actual", "actual", "predicted"]);
});
