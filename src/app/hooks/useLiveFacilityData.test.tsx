import {act, renderHook, waitFor} from "@testing-library/react";
import {beforeEach, describe, expect, it, vi} from "vitest";
import {fetchFacility} from "../../lib/api/facilityParser";
import {CACHE_KEY, setFacilityCache} from "../../lib/storage/facilityCache";
import type {FacilityId, FacilityPayload, Location} from "../../lib/types/facility";
import {useLiveFacilityData} from "./useLiveFacilityData";

vi.mock("../../lib/api/facilityParser", () => ({
    fetchFacility: vi.fn(),
}));

const mockedFetchFacility = vi.mocked(fetchFacility);

const cachedLocation: Location = {
    facilityId: 1186,
    locationId: 5761,
    locationName: "Nick Power House",
    floor: 0,
    isClosed: false,
    currentCapacity: 30,
    maxCapacity: 50,
    lastUpdated: "2026-08-31T11:55:00Z",
    fetchedAt: "2026-08-31T12:00:00Z",
};

const livePayload: FacilityPayload = {
    facilityId: 1186,
    facilityName: "Nick",
    floors: {0: [cachedLocation]},
    locations: [cachedLocation],
    liveDataSource: "facility_api",
};

const freshLocation: Location = {
    ...cachedLocation,
    currentCapacity: 31,
    lastUpdated: "2026-08-31T12:04:00Z",
    fetchedAt: "2026-08-31T12:05:00Z",
};

const freshPayload: FacilityPayload = {
    ...livePayload,
    floors: {0: [freshLocation]},
    locations: [freshLocation],
};

const bakkeLocation: Location = {
    facilityId: 1656,
    locationId: 8718,
    locationName: "Bakke The Point",
    floor: 1,
    isClosed: false,
    currentCapacity: 12,
    maxCapacity: 40,
    lastUpdated: "2026-08-31T12:04:00Z",
    fetchedAt: "2026-08-31T12:05:00Z",
};

const bakkePayload: FacilityPayload = {
    facilityId: 1656,
    facilityName: "Bakke",
    floors: {1: [bakkeLocation]},
    locations: [bakkeLocation],
    liveDataSource: "facility_api",
};

const deferred = <T,>() => {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((promiseResolve, promiseReject) => {
        resolve = promiseResolve;
        reject = promiseReject;
    });
    return {promise, reject, resolve};
};

describe("useLiveFacilityData", () => {
    beforeEach(() => {
        mockedFetchFacility.mockReset();
    });

    it("publishes the local acceptance time together with a new observation", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:05:00Z");
        mockedFetchFacility.mockResolvedValue(freshPayload);
        const {result} = renderHook(() => useLiveFacilityData({facility: 1186, refreshKey: 0, isOffline: false}));
        await act(async () => { await Promise.resolve(); });
        expect(result.current).toMatchObject({data: freshPayload, acceptedAtMs: Date.parse("2026-08-31T12:05:00Z")});
    });

    it("retains a schema-valid cached snapshot while offline and labels its source", async () => {
        setFacilityCache(1186, livePayload);

        const {result} = renderHook(() => useLiveFacilityData({
            facility: 1186,
            refreshKey: 0,
            isOffline: true,
        }));

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current).toMatchObject({
            data: livePayload,
            liveDataSource: "cache",
            liveOutageState: "cache",
            error: null,
        });
        expect(result.current.cacheTimestampMs).toEqual(expect.any(Number));
        expect(mockedFetchFacility).not.toHaveBeenCalled();
    });

    it("keeps the current in-memory snapshot qualified when going offline without persistent cache", async () => {
        mockedFetchFacility.mockResolvedValue(livePayload);

        const {result, rerender} = renderHook(
            ({isOffline}) => useLiveFacilityData({
                facility: 1186,
                refreshKey: 0,
                isOffline,
            }),
            {initialProps: {isOffline: false}},
        );

        await waitFor(() => expect(result.current).toMatchObject({
            data: livePayload,
            isLoading: false,
            liveDataSource: "facility_api",
            liveOutageState: "none",
        }));
        window.localStorage.removeItem(CACHE_KEY);

        rerender({isOffline: true});

        await waitFor(() => expect(result.current).toMatchObject({
            data: livePayload,
            isLoading: false,
            liveDataSource: "facility_api",
            liveOutageState: "cache",
            hasPendingLiveRetry: false,
            cacheTimestampMs: null,
            error: null,
        }));
        expect(mockedFetchFacility).toHaveBeenCalledTimes(1);
    });

    it("delegates retry ownership to the shared client", async () => {
        vi.useFakeTimers();
        mockedFetchFacility.mockRejectedValue(new Error("service unavailable"));

        const {result} = renderHook(() => useLiveFacilityData({
            facility: 1186,
            refreshKey: 0,
            isOffline: false,
        }));

        await act(async () => {
            await vi.advanceTimersByTimeAsync(10_000);
        });

        expect(mockedFetchFacility).toHaveBeenCalledTimes(1);
        expect(result.current).toMatchObject({
            data: null,
            isLoading: false,
            liveOutageState: "no_cache",
        });
    });

    it("keeps current live data visible when a refresh fails", async () => {
        mockedFetchFacility
            .mockResolvedValueOnce(livePayload)
            .mockRejectedValueOnce(new Error("refresh failed"));

        const {result, rerender} = renderHook(
            ({refreshKey}) => useLiveFacilityData({
                facility: 1186,
                refreshKey,
                isOffline: false,
            }),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(result.current.data).toEqual(livePayload));
        rerender({refreshKey: 1});

        await waitFor(() => expect(result.current.hasPendingLiveRetry).toBe(true));
        expect(result.current).toMatchObject({
            data: livePayload,
            liveDataSource: "facility_api",
            liveOutageState: "none",
            error: null,
        });
        expect(mockedFetchFacility).toHaveBeenCalledTimes(2);
    });

    it("keeps an online cached snapshot qualified while revalidation is pending", async () => {
        const revalidation = deferred<FacilityPayload>();
        setFacilityCache(1186, livePayload);
        mockedFetchFacility.mockReturnValue(revalidation.promise);

        const {result} = renderHook(() => useLiveFacilityData({
            facility: 1186,
            refreshKey: 0,
            isOffline: false,
        }));

        await waitFor(() => expect(result.current.data).toEqual(livePayload));
        expect(result.current).toMatchObject({
            data: livePayload,
            isLoading: true,
            liveDataSource: "cache",
            liveOutageState: "cache",
            error: null,
        });
        expect(result.current.cacheTimestampMs).toEqual(expect.any(Number));

        await act(async () => {
            revalidation.resolve(freshPayload);
            await revalidation.promise;
        });

        await waitFor(() => expect(result.current).toMatchObject({
            data: freshPayload,
            isLoading: false,
            liveDataSource: "facility_api",
            liveOutageState: "none",
            cacheTimestampMs: null,
        }));
    });

    it("retains the same qualified in-memory cache when storage disappears before another failed refresh", async () => {
        const failedRefresh = deferred<FacilityPayload>();
        setFacilityCache(1186, livePayload);
        mockedFetchFacility
            .mockRejectedValueOnce(new Error("initial refresh failed"))
            .mockReturnValueOnce(failedRefresh.promise);

        const {result, rerender} = renderHook(
            ({refreshKey}) => useLiveFacilityData({
                facility: 1186,
                refreshKey,
                isOffline: false,
            }),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(result.current).toMatchObject({
            data: livePayload,
            isLoading: false,
            liveDataSource: "cache",
            liveOutageState: "cache",
            hasPendingLiveRetry: false,
            error: null,
        }));
        const retainedSnapshot = result.current.data;
        const retainedCacheTimestampMs = result.current.cacheTimestampMs;
        expect(retainedSnapshot?.facilityId).toBe(1186);
        expect(retainedCacheTimestampMs).toEqual(expect.any(Number));

        window.localStorage.removeItem(CACHE_KEY);
        rerender({refreshKey: 1});

        await waitFor(() => expect(result.current).toMatchObject({
            data: livePayload,
            isLoading: true,
            liveDataSource: "cache",
            liveOutageState: "cache",
        }));
        expect(result.current.data).toBe(retainedSnapshot);

        await act(async () => {
            failedRefresh.reject(new Error("second refresh failed"));
            await failedRefresh.promise.catch(() => undefined);
        });

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(result.current).toMatchObject({
            data: livePayload,
            liveDataSource: "cache",
            liveOutageState: "cache",
            hasPendingLiveRetry: false,
            cacheTimestampMs: retainedCacheTimestampMs,
            error: null,
        });
        expect(result.current.data).toBe(retainedSnapshot);
        expect(mockedFetchFacility).toHaveBeenCalledTimes(2);
    });

    it("aborts a superseded facility request and ignores its late result", async () => {
        const nickRequest = deferred<FacilityPayload>();
        const bakkeRequest = deferred<FacilityPayload>();
        mockedFetchFacility.mockImplementation((facility) => (
            facility === 1186 ? nickRequest.promise : bakkeRequest.promise
        ));

        const {result, rerender} = renderHook(
            ({facility}: {facility: FacilityId}) => useLiveFacilityData({
                facility,
                refreshKey: 0,
                isOffline: false,
            }),
            {initialProps: {facility: 1186 as FacilityId}},
        );

        await waitFor(() => expect(mockedFetchFacility).toHaveBeenCalledTimes(1));
        const nickSignal = mockedFetchFacility.mock.calls[0]?.[1];
        expect(nickSignal?.aborted).toBe(false);

        rerender({facility: 1656});

        await waitFor(() => expect(mockedFetchFacility).toHaveBeenCalledTimes(2));
        expect(mockedFetchFacility.mock.calls.map(([facility]) => facility)).toEqual([1186, 1656]);
        expect(nickSignal?.aborted).toBe(true);

        await act(async () => {
            bakkeRequest.resolve(bakkePayload);
            await bakkeRequest.promise;
        });
        await waitFor(() => expect(result.current.data).toEqual(bakkePayload));

        await act(async () => {
            nickRequest.resolve(livePayload);
            await nickRequest.promise;
        });
        expect(result.current.data).toEqual(bakkePayload);
    });

    it("aborts a superseded live refresh and ignores its late result", async () => {
        const staleRequest = deferred<FacilityPayload>();
        const currentRequest = deferred<FacilityPayload>();
        mockedFetchFacility
            .mockReturnValueOnce(staleRequest.promise)
            .mockReturnValueOnce(currentRequest.promise);

        const {result, rerender} = renderHook(
            ({refreshKey}) => useLiveFacilityData({
                facility: 1186,
                refreshKey,
                isOffline: false,
            }),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(mockedFetchFacility).toHaveBeenCalledTimes(1));
        const staleSignal = mockedFetchFacility.mock.calls[0]?.[1];
        rerender({refreshKey: 1});

        await waitFor(() => expect(mockedFetchFacility).toHaveBeenCalledTimes(2));
        expect(staleSignal?.aborted).toBe(true);

        await act(async () => {
            currentRequest.resolve(freshPayload);
            await currentRequest.promise;
        });
        await waitFor(() => expect(result.current.data).toEqual(freshPayload));

        await act(async () => {
            staleRequest.resolve(livePayload);
            await staleRequest.promise;
        });
        expect(result.current.data).toEqual(freshPayload);
    });
});
