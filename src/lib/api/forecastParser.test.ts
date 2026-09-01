import axios, {AxiosHeaders, type AxiosResponse} from "axios";
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
    observedCapacity: 100,
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

const axiosResponse = <T,>(data: T): AxiosResponse<T> => ({
    data,
    status: 200,
    statusText: "OK",
    headers: {},
    config: {headers: new AxiosHeaders()},
});

const futureForecastResponse = () => axiosResponse({
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
            totalHours: [{hourStart: "2999-01-01T09:00:00-06:00", expectedCount: 50}],
        },
    ],
});

const futureActualResponse = () => axiosResponse({
    facilityId: 1186,
    date: "2999-01-01",
    categories: [],
    totalHours: [],
});

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
        ["a null actual count", {actualCount: null}],
        ["low location coverage", {actualCoverage: 0.5}],
        ["low temporal coverage", {temporalCoverage: 0.5}],
    ])("retains forecast for %s", (_label, overrides) => {
        const merged = forecastParser.mergeActualHoursIntoDays(
            fallbackForecastDays(),
            actualPayload({totalHours: [actualHour(overrides)]}),
        );

        expect(merged[0].totalHours?.[1]).toEqual({
            hourStart: "2026-11-01T01:00:00-06:00",
            expectedCount: 60,
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
                    actualHour({hourStart: "2026-11-01T01:00:00-06:00", actualCount: 41}),
                    actualHour({hourStart: "2026-11-01T07:00:00Z", actualCount: 42}),
                    actualHour({hourStart: "2026-11-01T08:00:00Z", actualCount: 43}),
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
                        hours: [
                            actualHour({hourStart: "2026-11-01T06:00:00Z", actualCount: 31}),
                            actualHour({hourStart: "2026-11-01T01:00:00-06:00", actualCount: 41}),
                            actualHour({hourStart: "2026-11-01T07:00:00Z", actualCount: 42}),
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
                    actualHour({hourStart: "2026-11-01T01:00:00-06:00", actualCount: 41}),
                    actualHour({
                        hourStart: "2026-11-01T07:00:00Z",
                        actualCount: 42,
                        actualCoverage: 0.5,
                    }),
                    actualHour({hourStart: "2026-11-01T08:00:00Z", actualCount: 43}),
                    actualHour({
                        hourStart: "2026-11-01T08:00:00Z",
                        observedCapacity: "malformed",
                    }),
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
                        hours: [
                            actualHour({hourStart: "2026-11-01T06:00:00Z", actualCount: 31}),
                            actualHour({hourStart: "2026-11-01T01:00:00-06:00", actualCount: 41}),
                            actualHour({hourStart: "2026-11-01T07:00:00Z", actualCount: null}),
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
                        hours: [actualHour({hourStart: "2026-11-01T06:00:00Z", actualCount: 31})],
                    },
                    {
                        key: "fitness_floors",
                        hours: [actualHour({hourStart: "2026-11-01T07:00:00Z", actualCount: 41})],
                    },
                    {
                        key: "pool",
                        hours: [actualHour({hourStart: "2026-11-01T07:00:00Z", actualCount: 18})],
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
            observedCapacity: 100,
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
        const forecast = deferred<AxiosResponse>();
        const get = vi.spyOn(axios, "get").mockImplementation((url) => {
            if (url.endsWith("/actual-hours")) {
                return Promise.resolve(futureActualResponse());
            }
            return forecast.promise;
        });
        const request = forecastParser.fetchForecastDays(1186);

        try {
            await Promise.resolve();
            expect(get.mock.calls.map(([url]) => new URL(url, window.location.origin).pathname)).toEqual([
                "/api/forecast/facilities/1186",
                "/api/forecast/facilities/1186/actual-hours",
            ]);
        } finally {
            forecast.resolve(futureForecastResponse());
            await request;
        }
    });

    it.each(["already", "during"] as const)(
        "rejects with CanceledError when the signal is %s aborted despite fulfilled responses",
        async (timing) => {
            const controller = new AbortController();
            if (timing === "already") {
                controller.abort();
            }
            vi.spyOn(axios, "get")
                .mockResolvedValueOnce(futureForecastResponse())
                .mockResolvedValueOnce(futureActualResponse());

            const request = forecastParser.fetchForecastDays(1186, controller.signal);
            if (timing === "during") {
                controller.abort();
            }

            await expect(request).rejects.toMatchObject({name: "CanceledError"});
        },
    );

    it("retains a valid forecast when the optional actual-hour request fails", async () => {
        vi.spyOn(console, "info").mockImplementation(() => undefined);
        vi.spyOn(axios, "get").mockImplementation((url) => {
            if (url.endsWith("/actual-hours")) {
                return Promise.reject(new Error("optional endpoint unavailable"));
            }
            return Promise.resolve(futureForecastResponse());
        });

        await expect(forecastParser.fetchForecastDays(1186)).resolves.toMatchObject({
            days: [{date: "2999-01-01"}],
        });
    });

    it("rejects a forecast failure even when actual hours fulfill", async () => {
        const forecastError = new Error("forecast unavailable");
        vi.spyOn(axios, "get").mockImplementation((url) => {
            if (url.endsWith("/actual-hours")) {
                return Promise.resolve(futureActualResponse());
            }
            return Promise.reject(forecastError);
        });

        await expect(forecastParser.fetchForecastDays(1186)).rejects.toBe(forecastError);
    });
});
