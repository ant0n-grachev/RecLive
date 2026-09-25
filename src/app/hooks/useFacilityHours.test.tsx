import {act, renderHook, waitFor} from "@testing-library/react";
import {afterEach, beforeEach, describe, expect, it, vi} from "vitest";
import {fetchFacilityHours} from "../../lib/api/facilityScheduleParser";
import type {FacilityScheduleResponse} from "../../lib/api/schemas";
import type {FacilityId} from "../../lib/types/facility";
import {useFacilityHours} from "./useFacilityHours";

vi.mock("../../lib/api/facilityScheduleParser", () => ({
    fetchFacilityHours: vi.fn(),
}));

const mockedFetchFacilityHours = vi.mocked(fetchFacilityHours);

const nickSchedule: FacilityScheduleResponse = {
    generatedAt: "2026-09-01T12:00:00Z",
    sourceSite: "https://recwell.wisc.edu",
    facilityId: 1186,
    facilityName: "Nicholas Recreation Center",
    slug: "nick",
    url: "https://recwell.wisc.edu/nick/",
    resolvedUrl: "https://recwell.wisc.edu/nick/",
    status: "ok",
    source: "wp_json",
    sourceModifiedGmt: "2026-09-01T11:45:00",
    sections: [{
        title: "Building Hours",
        rows: [{label: "Mon-Fri", hours: "6:00 am - 10:00 pm"}],
        note: null,
    }],
    sourceFetchedAt: "2026-09-01T12:00:00Z",
    lastSuccessfulAt: "2026-09-01T12:00:00Z",
    stale: false,
    error: null,
    errorCategory: null,
    updatedAt: "2026-09-01T12:00:00Z",
};

const bakkeSchedule: FacilityScheduleResponse = {
    ...nickSchedule,
    facilityId: 1656,
    facilityName: "Bakke Recreation & Wellbeing Center",
    slug: "bakke",
    url: "https://recwell.wisc.edu/bakke/",
    resolvedUrl: "https://recwell.wisc.edu/bakke/",
};

const deferred = <T,>() => {
    let resolve!: (value: T) => void;
    let reject!: (reason?: unknown) => void;
    const promise = new Promise<T>((resolvePromise, rejectPromise) => {
        resolve = resolvePromise;
        reject = rejectPromise;
    });
    return {promise, resolve, reject};
};

describe("useFacilityHours", () => {
    beforeEach(() => {
        mockedFetchFacilityHours.mockReset();
    });

    afterEach(() => {
        vi.useRealTimers();
    });

    it("delegates retry ownership to the shared request client", async () => {
        vi.useFakeTimers();
        mockedFetchFacilityHours.mockRejectedValue(new Error("schedule unavailable"));

        const {result} = renderHook(() => useFacilityHours({
            facility: 1186,
            refreshKey: 0,
        }));

        await act(async () => {
            await vi.advanceTimersByTimeAsync(10_000);
        });

        expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(1);
        expect(result.current).toMatchObject({
            activeSchedule: null,
            isFacilityHoursLoading: false,
            facilityHoursError: "Schedule unavailable right now.",
            hasPendingScheduleRetry: false,
        });
    });

    it("keeps current schedule data visible when a refresh fails", async () => {
        const refreshRequest = deferred<FacilityScheduleResponse>();
        mockedFetchFacilityHours
            .mockResolvedValueOnce(nickSchedule)
            .mockReturnValueOnce(refreshRequest.promise);

        const {result, rerender} = renderHook(
            ({refreshKey}) => useFacilityHours({facility: 1186, refreshKey}),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(result.current.activeSchedule).toEqual(nickSchedule));
        rerender({refreshKey: 1});
        await waitFor(() => expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(2));
        expect(result.current).toMatchObject({
            activeSchedule: nickSchedule,
            isFacilityHoursLoading: true,
            facilityHoursError: null,
        });

        await act(async () => {
            refreshRequest.reject(new Error("refresh failed"));
            await refreshRequest.promise.catch(() => undefined);
        });

        await waitFor(() => expect(result.current.hasPendingScheduleRetry).toBe(true));
        expect(result.current).toMatchObject({
            activeSchedule: nickSchedule,
            isFacilityHoursLoading: false,
            facilityHoursError: null,
        });
    });

    it("clears a pending retry after switching to a facility with no retained schedule", async () => {
        const bakkeRequest = deferred<FacilityScheduleResponse>();
        mockedFetchFacilityHours
            .mockResolvedValueOnce(nickSchedule)
            .mockRejectedValueOnce(new Error("refresh failed"))
            .mockReturnValueOnce(bakkeRequest.promise);

        const {result, rerender} = renderHook(
            ({facility, refreshKey}: {facility: FacilityId; refreshKey: number}) => (
                useFacilityHours({facility, refreshKey})
            ),
            {initialProps: {facility: 1186 as FacilityId, refreshKey: 0}},
        );

        await waitFor(() => expect(result.current.activeSchedule).toEqual(nickSchedule));
        rerender({facility: 1186, refreshKey: 1});
        await waitFor(() => expect(result.current.hasPendingScheduleRetry).toBe(true));

        rerender({facility: 1656, refreshKey: 1});
        await waitFor(() => expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(3));
        expect(result.current).toMatchObject({
            activeSchedule: null,
            isFacilityHoursLoading: true,
            hasPendingScheduleRetry: false,
        });

        await act(async () => {
            bakkeRequest.reject(new Error("schedule unavailable"));
            await bakkeRequest.promise.catch(() => undefined);
        });
        await waitFor(() => expect(result.current.facilityHoursError).toBe(
            "Schedule unavailable right now."
        ));
        expect(result.current.hasPendingScheduleRetry).toBe(false);
    });

    it("aborts a superseded facility request and ignores its late result", async () => {
        const nickRequest = deferred<FacilityScheduleResponse>();
        const bakkeRequest = deferred<FacilityScheduleResponse>();
        mockedFetchFacilityHours.mockImplementation((facility) => (
            facility === 1186 ? nickRequest.promise : bakkeRequest.promise
        ));

        const {result, rerender} = renderHook(
            ({facility}: {facility: FacilityId}) => useFacilityHours({
                facility,
                refreshKey: 0,
            }),
            {initialProps: {facility: 1186 as FacilityId}},
        );

        await waitFor(() => expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(1));
        const nickSignal = mockedFetchFacilityHours.mock.calls[0]?.[1];
        expect(nickSignal?.aborted).toBe(false);

        rerender({facility: 1656});
        await waitFor(() => expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(2));
        expect(mockedFetchFacilityHours.mock.calls.map(([facility]) => facility)).toEqual([
            1186,
            1656,
        ]);
        expect(nickSignal?.aborted).toBe(true);

        await act(async () => {
            bakkeRequest.resolve(bakkeSchedule);
            await bakkeRequest.promise;
        });
        await waitFor(() => expect(result.current.activeSchedule).toEqual(bakkeSchedule));

        await act(async () => {
            nickRequest.resolve(nickSchedule);
            await nickRequest.promise;
        });
        expect(result.current.activeSchedule).toEqual(bakkeSchedule);
    });

    it("aborts a superseded schedule refresh and ignores its late result", async () => {
        const staleRequest = deferred<FacilityScheduleResponse>();
        const currentRequest = deferred<FacilityScheduleResponse>();
        const refreshedSchedule: FacilityScheduleResponse = {
            ...nickSchedule,
            updatedAt: "2026-09-01T13:00:00Z",
        };
        mockedFetchFacilityHours
            .mockReturnValueOnce(staleRequest.promise)
            .mockReturnValueOnce(currentRequest.promise);

        const {result, rerender} = renderHook(
            ({refreshKey}) => useFacilityHours({facility: 1186, refreshKey}),
            {initialProps: {refreshKey: 0}},
        );

        await waitFor(() => expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(1));
        const staleSignal = mockedFetchFacilityHours.mock.calls[0]?.[1];
        rerender({refreshKey: 1});

        await waitFor(() => expect(mockedFetchFacilityHours).toHaveBeenCalledTimes(2));
        expect(staleSignal?.aborted).toBe(true);

        await act(async () => {
            currentRequest.resolve(refreshedSchedule);
            await currentRequest.promise;
        });
        await waitFor(() => expect(result.current.activeSchedule).toEqual(refreshedSchedule));

        await act(async () => {
            staleRequest.resolve(nickSchedule);
            await staleRequest.promise;
        });
        expect(result.current.activeSchedule).toEqual(refreshedSchedule);
    });
});
