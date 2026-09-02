import {http, HttpResponse} from "msw";
import {env} from "../config/env";
import {server} from "../../test/msw/server";
import * as forecastParser from "./forecastParser";
import type {ForecastDay} from "../types/forecast";

const fallbackForecastDays = (): ForecastDay[] => [
    {
        dayName: "Sunday",
        date: "2026-11-01",
        totalHours: [
            {hourStart: "2026-11-01T01:00:00-05:00", expectedCount: 50},
            {hourStart: "2026-11-01T01:00:00-06:00", expectedCount: 60},
        ],
        categories: [
            {
                key: "fitness_floors",
                title: "Fitness Floors",
                hours: [
                    {hourStart: "2026-11-01T01:00:00-05:00", expectedCount: 30},
                    {hourStart: "2026-11-01T01:00:00-06:00", expectedCount: 40},
                ],
            },
        ],
    },
];

const actualHour = (overrides: Record<string, unknown> = {}) => ({
    hourStart: "2026-11-01T01:00:00-06:00",
    observedCount: 40,
    observedCapacity: 200,
    expectedCapacity: 200,
    actualCoverage: 1,
    temporalCoverage: 1,
    coverageThreshold: 0.75,
    actualCount: 40,
    actualPct: 0.2,
    ...overrides,
});

const actualPayload = (overrides: Record<string, unknown> = {}) => ({
    facilityId: 1186,
    date: "2026-11-01",
    categories: [],
    totalHours: [actualHour()],
    ...overrides,
});

const futureForecastResponse = () => ({
    facilityId: 1186,
    facilityName: "Nick",
    forecastDayStartHour: 6,
    forecastDayEndHour: 23,
    occupancyThresholds: {lowMax: 34, peakMin: 70},
    sectionOccupancyThresholds: {},
    locationOccupancyThresholds: {},
    weeklyForecast: [
        {
            dayName: "Thursday",
            date: "2999-01-01",
            totalHours: [{
                hour: 9,
                hourStart: "2999-01-01T09:00:00-06:00",
                expectedCount: 50,
                spikeAdjusted: true,
            }],
        },
    ],
});

const futureActualResponse = () => ({
    facilityId: 1186,
    date: "2999-01-01",
    categories: [],
    totalHours: [],
});

const chicagoDateToday = (): string => {
    const parts = new Intl.DateTimeFormat("en-US", {
        timeZone: "America/Chicago",
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    }).formatToParts(new Date());
    const year = parts.find((part) => part.type === "year")?.value;
    const month = parts.find((part) => part.type === "month")?.value;
    const day = parts.find((part) => part.type === "day")?.value;
    if (!year || !month || !day) throw new Error("Chicago date unavailable in test runtime");
    return `${year}-${month}-${day}`;
};

const validActualHourAt = (hourStart: string) => ({
    hourStart,
    observedCount: 40,
    observedCapacity: 200,
    expectedCapacity: 200,
    actualCoverage: 1,
    temporalCoverage: 1,
    coverageThreshold: 0.75,
    actualCount: 40,
    actualPct: 0.2,
});

const apiUrl = (path: string): string => env.apiBaseUrl
    ? `${env.apiBaseUrl}${path}`
    : `${window.location.origin}${path}`;

const FORECAST_PATH = "/api/forecast/facilities/1186";
const FORECAST_URL = apiUrl(FORECAST_PATH);
const ACTUAL_URL = apiUrl(`${FORECAST_PATH}/actual-hours`);

const deferred = <T,>() => {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
    });
    return {promise, resolve, reject};
};

describe("mergeActualHoursIntoDays", () => {
    it.each([
        ["low coverage with a non-null actual", {
            hourStart: "2026-11-01T01:00:00-06:00",
            observedCount: 40,
            observedCapacity: 100,
            expectedCapacity: 200,
            actualCoverage: 0.5,
            temporalCoverage: 1,
            coverageThreshold: 0.75,
            actualCount: 40,
            actualPct: 0.2,
        }],
        ["negative counts", {
            hourStart: "2026-11-01T01:00:00-06:00",
            observedCount: -1,
            observedCapacity: 200,
            expectedCapacity: 200,
            actualCoverage: 1,
            temporalCoverage: 1,
            coverageThreshold: 0.75,
            actualCount: -1,
            actualPct: 0,
        }],
        ["coverage inconsistent with capacity", {
            hourStart: "2026-11-01T01:00:00-06:00",
            observedCount: 40,
            observedCapacity: 100,
            expectedCapacity: 200,
            actualCoverage: 1,
            temporalCoverage: 1,
            coverageThreshold: 0.75,
            actualCount: 40,
            actualPct: 0.2,
        }],
    ])("fails closed for %s", (_label, invalidHour) => {
        const days = fallbackForecastDays();

        expect(forecastParser.mergeActualHoursIntoDays(
            days,
            actualPayload({totalHours: [invalidHour]}),
        )).toBe(days);
    });

    it.each([
        ["a missing observation", {
            observedCount: null,
            observedCapacity: 0,
            actualCoverage: 0,
            actualCount: null,
            actualPct: null,
        }],
        ["low location coverage", {
            observedCapacity: 100,
            actualCoverage: 0.5,
            actualCount: null,
            actualPct: null,
        }],
        ["low temporal coverage", {
            temporalCoverage: 0.5,
            actualCount: null,
            actualPct: null,
        }],
    ])("retains explicit null actual occupancy for %s", (_label, overrides) => {
        const merged = forecastParser.mergeActualHoursIntoDays(
            fallbackForecastDays(),
            actualPayload({totalHours: [actualHour(overrides)]}),
        );

        expect(merged[0].totalHours?.[1]).toMatchObject({
            hourStart: "2026-11-01T01:00:00-06:00",
            expectedCount: 60,
            actualCount: null,
        });
    });

    it("matches exact finite epochs while keeping both fall-back hours distinct", () => {
        const merged = forecastParser.mergeActualHoursIntoDays(
            fallbackForecastDays(),
            actualPayload({
                totalHours: [actualHour({hourStart: "2026-11-01T07:00:00Z"})],
            }),
        );

        expect(merged[0].totalHours?.[0]?.actualCount).toBeUndefined();
        expect(merged[0].totalHours?.[1]?.actualCount).toBe(40);
    });

    it.each([
        ["fall-back hour", "2026-11-01", "2026-11-01T01:00:00"],
        ["spring-gap hour", "2026-03-08", "2026-03-08T02:00:00"],
    ])("rejects a naive %s even when the host normalizes it to the forecast instant", (
        _label,
        date,
        naiveHourStart,
    ) => {
        const normalizedEpoch = Date.parse(naiveHourStart);
        expect(Number.isFinite(normalizedEpoch)).toBe(true);
        const explicitHourStart = new Date(normalizedEpoch).toISOString();
        const days: ForecastDay[] = [{
            dayName: "Sunday",
            date,
            totalHours: [{hourStart: explicitHourStart, expectedCount: 70}],
        }];

        const merged = forecastParser.mergeActualHoursIntoDays(
            days,
            actualPayload({
                date,
                totalHours: [actualHour({hourStart: naiveHourStart})],
            }),
        );

        expect(merged[0].totalHours?.[0]).toEqual({
            hourStart: explicitHourStart,
            expectedCount: 70,
        });
    });

    it("fails closed for duplicate total-hour epochs while preserving a unique sibling", () => {
        const days = fallbackForecastDays();
        days[0].totalHours = [
            ...(days[0].totalHours ?? []),
            {hourStart: "2026-11-01T02:00:00-06:00", expectedCount: 70},
        ];

        const merged = forecastParser.mergeActualHoursIntoDays(
            days,
            actualPayload({
                totalHours: [
                    actualHour({hourStart: "2026-11-01T01:00:00-06:00", observedCount: 41, actualCount: 41}),
                    actualHour({hourStart: "2026-11-01T07:00:00Z", observedCount: 42, actualCount: 42}),
                    actualHour({hourStart: "2026-11-01T08:00:00Z", observedCount: 43, actualCount: 43}),
                ],
            }),
        );

        expect(merged[0].totalHours?.[1]).toEqual({
            hourStart: "2026-11-01T01:00:00-06:00",
            expectedCount: 60,
        });
        expect(merged[0].totalHours?.[2]?.actualCount).toBe(43);
    });

    it("fails closed for duplicate qualified epochs within one category", () => {
        const merged = forecastParser.mergeActualHoursIntoDays(
            fallbackForecastDays(),
            actualPayload({
                totalHours: [],
                categories: [
                    {
                        key: "fitness_floors",
                        title: "Fitness Floors",
                        hours: [
                            actualHour({hourStart: "2026-11-01T06:00:00Z", observedCount: 31, actualCount: 31}),
                            actualHour({hourStart: "2026-11-01T01:00:00-06:00", observedCount: 41, actualCount: 41}),
                            actualHour({hourStart: "2026-11-01T07:00:00Z", observedCount: 42, actualCount: 42}),
                        ],
                    },
                ],
            }),
        );

        expect(merged[0].categories?.[0]?.hours[0]?.actualCount).toBe(31);
        expect(merged[0].categories?.[0]?.hours[1]).toEqual({
            hourStart: "2026-11-01T01:00:00-06:00",
            expectedCount: 40,
        });
    });

    it("fails closed when a total-hour duplicate mixes qualified and low-coverage rows", () => {
        const days = fallbackForecastDays();
        days[0].totalHours = [
            ...(days[0].totalHours ?? []),
            {hourStart: "2026-11-01T02:00:00-06:00", expectedCount: 70},
        ];

        const merged = forecastParser.mergeActualHoursIntoDays(
            days,
            actualPayload({
                totalHours: [
                    actualHour({hourStart: "2026-11-01T01:00:00-06:00", observedCount: 41, actualCount: 41}),
                    actualHour({
                        hourStart: "2026-11-01T07:00:00Z",
                        observedCapacity: 100,
                        actualCoverage: 0.5,
                        actualCount: null,
                        actualPct: null,
                    }),
                    actualHour({hourStart: "2026-11-01T08:00:00Z", observedCount: 43, actualCount: 43}),
                ],
            }),
        );

        expect(merged[0].totalHours?.[1]).toEqual({
            hourStart: "2026-11-01T01:00:00-06:00",
            expectedCount: 60,
        });
        expect(merged[0].totalHours?.[2]?.actualCount).toBe(43);
    });

    it("fails closed when a category-hour duplicate mixes qualified and null rows", () => {
        const merged = forecastParser.mergeActualHoursIntoDays(
            fallbackForecastDays(),
            actualPayload({
                totalHours: [],
                categories: [
                    {
                        key: "fitness_floors",
                        title: "Fitness Floors",
                        hours: [
                            actualHour({hourStart: "2026-11-01T06:00:00Z", observedCount: 31, actualCount: 31}),
                            actualHour({hourStart: "2026-11-01T01:00:00-06:00", observedCount: 41, actualCount: 41}),
                            actualHour({
                                hourStart: "2026-11-01T07:00:00Z",
                                observedCapacity: 100,
                                actualCoverage: 0.5,
                                actualCount: null,
                                actualPct: null,
                            }),
                        ],
                    },
                ],
            }),
        );

        expect(merged[0].categories?.[0]?.hours[0]?.actualCount).toBe(31);
        expect(merged[0].categories?.[0]?.hours[1]).toEqual({
            hourStart: "2026-11-01T01:00:00-06:00",
            expectedCount: 40,
        });
    });

    it("fails closed for duplicate normalized categories while preserving a unique category", () => {
        const days = fallbackForecastDays();
        days[0].categories = [
            ...(days[0].categories ?? []),
            {
                key: "pool",
                title: "Pool",
                hours: [{hourStart: "2026-11-01T01:00:00-06:00", expectedCount: 20}],
            },
        ];

        const merged = forecastParser.mergeActualHoursIntoDays(
            days,
            actualPayload({
                totalHours: [],
                categories: [
                    {
                        key: "Fitness Floors",
                        title: "Fitness Floors",
                        hours: [actualHour({hourStart: "2026-11-01T06:00:00Z", observedCount: 31, actualCount: 31})],
                    },
                    {
                        key: "fitness_floors",
                        title: "Fitness Floors",
                        hours: [actualHour({hourStart: "2026-11-01T07:00:00Z", observedCount: 41, actualCount: 41})],
                    },
                    {
                        key: "pool",
                        title: "Pool",
                        hours: [actualHour({hourStart: "2026-11-01T07:00:00Z", observedCount: 18, actualCount: 18})],
                    },
                ],
            }),
        );

        expect(merged[0].categories?.[0]?.hours).toEqual([
            {hourStart: "2026-11-01T01:00:00-05:00", expectedCount: 30},
            {hourStart: "2026-11-01T01:00:00-06:00", expectedCount: 40},
        ]);
        expect(merged[0].categories?.[1]?.hours[0]?.actualCount).toBe(18);
    });

    it("merges qualified total and category observations with their coverage evidence", () => {
        const categoryHour = actualHour({
            observedCount: 32,
            observedCapacity: 80,
            expectedCapacity: 100,
            actualCoverage: 0.8,
            actualCount: 32,
            actualPct: 0.32,
        });
        const merged = forecastParser.mergeActualHoursIntoDays(
            fallbackForecastDays(),
            actualPayload({
                categories: [
                    {
                        key: "Fitness Floors",
                        title: "Fitness Floors",
                        hours: [categoryHour],
                    },
                ],
            }),
        );

        expect(merged[0].totalHours?.[1]).toMatchObject({
            expectedCount: 60,
            actualCount: 40,
            actualPct: 0.2,
            observedCount: 40,
            observedCapacity: 200,
            expectedCapacity: 200,
            actualCoverage: 1,
            temporalCoverage: 1,
            coverageThreshold: 0.75,
        });
        expect(merged[0].categories?.[0]?.hours[1]).toMatchObject({
            expectedCount: 40,
            actualCount: 32,
            observedCount: 32,
            observedCapacity: 80,
            expectedCapacity: 100,
        });
    });

    it.each([
        ["non-date hourStart", {hourStart: "not-a-date"}],
        ["nonfinite actualCount", {actualCount: Number.POSITIVE_INFINITY}],
        ["nonfinite observedCount", {observedCount: Number.NaN}],
        ["negative observed capacity", {observedCapacity: -1}],
        ["string expected capacity", {expectedCapacity: "200"}],
        ["nonfinite actual coverage", {actualCoverage: Number.NaN}],
        ["nonfinite temporal coverage", {temporalCoverage: Number.POSITIVE_INFINITY}],
        ["nonfinite coverage threshold", {coverageThreshold: Number.NaN}],
        ["nonfinite actual percentage", {actualPct: Number.NEGATIVE_INFINITY}],
    ])("ignores a malformed actual row with %s", (_label, overrides) => {
        const days = fallbackForecastDays();

        expect(forecastParser.mergeActualHoursIntoDays(
            days,
            actualPayload({totalHours: [actualHour(overrides)]}),
        )).toEqual(days);
    });

    it("ignores malformed containers instead of disturbing forecast data", () => {
        const days = fallbackForecastDays();

        expect(forecastParser.mergeActualHoursIntoDays(days, null)).toBe(days);
        expect(forecastParser.mergeActualHoursIntoDays(days, {date: 20261101})).toBe(days);
        expect(forecastParser.mergeActualHoursIntoDays(days, {
            facilityId: 1186,
            date: "2026-11-01",
            categories: [{key: {}, hours: "not-hours"}],
            totalHours: "not-hours",
        })).toBe(days);
    });
});

describe("fetchForecastDays", () => {
    it("starts forecast and actual-hour requests concurrently", async () => {
        const releaseForecast = deferred<void>();
        let forecastStarted = false;
        let actualStarted = false;
        server.use(
            http.get(FORECAST_URL, async ({request}) => {
                forecastStarted = true;
                expect(new URL(request.url).searchParams.get("compact")).toBe("1");
                await releaseForecast.promise;
                return HttpResponse.json(futureForecastResponse());
            }),
            http.get(ACTUAL_URL, ({request}) => {
                actualStarted = true;
                expect(new URL(request.url).searchParams.get("date")).toMatch(/^\d{4}-\d{2}-\d{2}$/u);
                return HttpResponse.json(futureActualResponse());
            }),
        );
        const request = forecastParser.fetchForecastDays(1186);

        try {
            await vi.waitFor(() => {
                expect(forecastStarted).toBe(true);
                expect(actualStarted).toBe(true);
            });
        } finally {
            releaseForecast.resolve();
            await request;
        }
    });

    it("normalizes a pre-aborted request through the shared client", async () => {
        const controller = new AbortController();
        controller.abort();

        await expect(forecastParser.fetchForecastDays(1186, controller.signal)).rejects.toMatchObject({
            kind: "aborted",
            message: "Request aborted",
        });
    });

    it("retains compact hour and spike-adjustment fields", async () => {
        server.use(
            http.get(FORECAST_URL, () => HttpResponse.json(futureForecastResponse())),
            http.get(ACTUAL_URL, () => HttpResponse.json(futureActualResponse())),
        );

        const payload = await forecastParser.fetchForecastDays(1186);
        const hour = payload.days[0]?.totalHours?.[0];

        expect(hour?.hour).toBe(9);
        expect(hour?.spikeAdjusted).toBe(true);
    });

    it("rejects a forecast envelope for a different requested facility", async () => {
        server.use(
            http.get(FORECAST_URL, () => HttpResponse.json({
                ...futureForecastResponse(),
                facilityId: 1656,
            })),
            http.get(ACTUAL_URL, () => HttpResponse.json(futureActualResponse())),
        );

        await expect(forecastParser.fetchForecastDays(1186)).rejects.toMatchObject({
            kind: "schema",
            message: "API response did not match its contract",
        });
    });

    it("ignores optional actual hours for a different facility", async () => {
        const today = chicagoDateToday();
        const hourStart = `${today}T15:00:00Z`;
        const forecast = {
            ...futureForecastResponse(),
            weeklyForecast: [{
                dayName: "Tuesday",
                date: today,
                totalHours: [{hourStart, expectedCount: 50}],
            }],
        };
        server.use(
            http.get(FORECAST_URL, () => HttpResponse.json(forecast)),
            http.get(ACTUAL_URL, () => HttpResponse.json({
                facilityId: 1656,
                date: today,
                categories: [],
                totalHours: [validActualHourAt(hourStart)],
            })),
        );

        const payload = await forecastParser.fetchForecastDays(1186);

        expect(payload.days[0]?.totalHours?.[0]).not.toHaveProperty("actualCount");
    });

    it("ignores optional actual hours for a date other than the requested date", async () => {
        const hourStart = "2999-01-01T09:00:00-06:00";
        server.use(
            http.get(FORECAST_URL, () => HttpResponse.json(futureForecastResponse())),
            http.get(ACTUAL_URL, () => HttpResponse.json({
                facilityId: 1186,
                date: "2999-01-01",
                categories: [],
                totalHours: [validActualHourAt(hourStart)],
            })),
        );

        const payload = await forecastParser.fetchForecastDays(1186);

        expect(payload.days[0]?.totalHours?.[0]).not.toHaveProperty("actualCount");
    });

    it("retains a valid forecast when actual-hours schema validation fails", async () => {
        server.use(
            http.get(FORECAST_URL, () => HttpResponse.json(futureForecastResponse())),
            http.get(ACTUAL_URL, () => HttpResponse.json({
                facilityId: 1186,
                date: "2999-01-01",
                categories: [],
                totalHours: [{
                    hourStart: "2999-01-01T09:00:00-06:00",
                    observedCount: 40,
                    observedCapacity: 100,
                    expectedCapacity: 200,
                    actualCoverage: 1,
                    temporalCoverage: 1,
                    coverageThreshold: 0.75,
                    actualCount: 40,
                    actualPct: 0.2,
                }],
            })),
        );

        const payload = await forecastParser.fetchForecastDays(1186);

        expect(payload.days[0]).toMatchObject({
            date: "2999-01-01",
            totalHours: [{expectedCount: 50}],
        });
        expect(payload.days[0]?.totalHours?.[0]).not.toHaveProperty("actualCount");
    });

    it("rejects a schema-invalid forecast even when actual hours fulfill", async () => {
        server.use(
            http.get(FORECAST_URL, () => HttpResponse.json({
                ...futureForecastResponse(),
                facilityId: "1186",
            })),
            http.get(ACTUAL_URL, () => HttpResponse.json(futureActualResponse())),
        );

        await expect(forecastParser.fetchForecastDays(1186)).rejects.toMatchObject({kind: "schema"});
    });

    it("uses the shared client's bounded three-attempt GET policy", async () => {
        let forecastRequests = 0;
        server.use(
            http.get(FORECAST_URL, () => {
                forecastRequests += 1;
                if (forecastRequests < 3) {
                    return new HttpResponse(null, {
                        status: 503,
                        headers: {"Retry-After": "0"},
                    });
                }
                return HttpResponse.json(futureForecastResponse());
            }),
            http.get(ACTUAL_URL, () => HttpResponse.json(futureActualResponse())),
        );

        await expect(forecastParser.fetchForecastDays(1186)).resolves.toMatchObject({
            days: [{date: "2999-01-01"}],
        });
        expect(forecastRequests).toBe(3);
    });
});
