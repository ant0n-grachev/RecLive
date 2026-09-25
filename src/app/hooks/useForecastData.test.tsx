import {act, renderHook, waitFor} from "@testing-library/react";
import {beforeEach, describe, expect, it, vi} from "vitest";
import {fetchForecastDays, type FacilityForecastPayload} from "../../lib/api/forecastParser";
import type {FacilityId} from "../../lib/types/facility";
import {useForecastData} from "./useForecastData";

vi.mock("../../lib/api/forecastParser", () => ({
    fetchForecastDays: vi.fn(),
}));

const mockedFetchForecastDays = vi.mocked(fetchForecastDays);

const deferred = <T,>() => {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
    });
    return {promise, resolve, reject};
};

const forecastPayload: FacilityForecastPayload = {
    days: [{
        dayName: "Thursday",
        date: "2999-01-01",
        totalHours: [{
            hourStart: "2999-01-01T09:00:00-06:00",
            expectedCount: 50,
        }],
    }],
    forecastDayStartHour: 6,
    forecastDayEndHour: 23,
    occupancyThresholds: {lowMax: 34, peakMin: 70},
    sectionOccupancyThresholds: {},
    locationOccupancyThresholds: {},
};

const newerForecastPayload: FacilityForecastPayload = {
    ...forecastPayload,
    days: [{
        dayName: "Friday",
        date: "2999-01-02",
        totalHours: [{
            hourStart: "2999-01-02T10:00:00-06:00",
            expectedCount: 60,
        }],
    }],
};

describe("useForecastData", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("delegates retry ownership to the shared API client", async () => {
        vi.useFakeTimers();
        vi.spyOn(console, "error").mockImplementation(() => undefined);
        mockedFetchForecastDays.mockRejectedValue(new Error("service unavailable"));

        const {result} = renderHook(() => useForecastData({
            facility: 1186,
            refreshKey: 0,
        }));

        await act(async () => {
            await vi.runAllTimersAsync();
        });

        expect(mockedFetchForecastDays).toHaveBeenCalledTimes(1);
        expect(result.current).toMatchObject({
            forecastDays: [],
            forecastError: "Forecast unavailable right now.",
            isForecastLoading: false,
            hasPendingForecastRetry: false,
        });
    });

    it("keeps current forecast data visible when a refresh fails", async () => {
        mockedFetchForecastDays
            .mockResolvedValueOnce(forecastPayload)
            .mockRejectedValueOnce(new Error("refresh failed"));

        const {result, rerender} = renderHook(
            ({refreshKey}) => useForecastData({
                facility: 1186,
                refreshKey,
            }),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(result.current.forecastDays).toEqual(forecastPayload.days));
        rerender({refreshKey: 1});

        await waitFor(() => expect(result.current.hasPendingForecastRetry).toBe(true));
        expect(result.current).toMatchObject({
            forecastDays: forecastPayload.days,
            forecastError: null,
            isForecastLoading: false,
        });
        expect(mockedFetchForecastDays).toHaveBeenCalledTimes(2);
    });

    it("keeps current forecast rows visible while a refresh is pending", async () => {
        const pendingRefresh = deferred<FacilityForecastPayload>();
        mockedFetchForecastDays
            .mockResolvedValueOnce(forecastPayload)
            .mockReturnValueOnce(pendingRefresh.promise);

        const {result, rerender} = renderHook(
            ({refreshKey}) => useForecastData({facility: 1186, refreshKey}),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(result.current.forecastDays).toEqual(forecastPayload.days));
        rerender({refreshKey: 1});

        await waitFor(() => expect(mockedFetchForecastDays).toHaveBeenCalledTimes(2));
        expect(result.current).toMatchObject({
            forecastDays: forecastPayload.days,
            isForecastLoading: true,
            forecastError: null,
        });

        await act(async () => {
            pendingRefresh.resolve(newerForecastPayload);
            await pendingRefresh.promise;
        });
        await waitFor(() => expect(result.current.forecastDays).toEqual(
            newerForecastPayload.days
        ));
    });

    it.each([
        ["facility change", 1656 as const, 0],
        ["refresh", 1186 as const, 1],
    ])("does not let a late result overwrite newer data after a %s", async (
        _label,
        nextFacility,
        nextRefreshKey,
    ) => {
        const staleRequest = deferred<FacilityForecastPayload>();
        const currentRequest = deferred<FacilityForecastPayload>();
        let staleSignal: AbortSignal | undefined;
        mockedFetchForecastDays
            .mockImplementationOnce((_facility, signal) => {
                staleSignal = signal;
                return staleRequest.promise;
            })
            .mockImplementationOnce(() => currentRequest.promise);

        const {result, rerender} = renderHook(
            ({facility, refreshKey}: {facility: FacilityId; refreshKey: number}) => (
                useForecastData({facility, refreshKey})
            ),
            {initialProps: {facility: 1186 as FacilityId, refreshKey: 0}},
        );
        await waitFor(() => expect(mockedFetchForecastDays).toHaveBeenCalledTimes(1));

        rerender({facility: nextFacility, refreshKey: nextRefreshKey});
        await waitFor(() => expect(mockedFetchForecastDays).toHaveBeenCalledTimes(2));
        expect(staleSignal?.aborted).toBe(true);

        await act(async () => {
            currentRequest.resolve(newerForecastPayload);
            await currentRequest.promise;
        });
        await waitFor(() => expect(result.current.forecastDays).toEqual(newerForecastPayload.days));

        await act(async () => {
            staleRequest.resolve(forecastPayload);
            await staleRequest.promise;
        });
        expect(result.current.forecastDays).toEqual(newerForecastPayload.days);
    });
});
