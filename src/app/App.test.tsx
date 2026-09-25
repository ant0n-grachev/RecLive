import {ThemeProvider} from "@mui/material";
import {act, fireEvent, render, screen} from "@testing-library/react";
import {MemoryRouter} from "react-router-dom";
import type {FacilityPayload} from "../lib/types/facility";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import App from "./App";
import {createAppTheme} from "./theme";

const liveHookState = vi.hoisted(() => ({
    data: null as FacilityPayload | null,
    isLoading: false,
    error: null,
    liveDataSource: "facility_api" as const,
    liveOutageState: "none" as const,
    hasPendingLiveRetry: false,
    cacheTimestampMs: null,
    acceptedAtMs: null as number | null,
    prepareRefresh: vi.fn(),
}));

const appHookHarness = vi.hoisted(() => ({
    isOffline: false,
    liveArgs: null as null | {
        facility: 1186 | 1656;
        refreshKey: number;
        isOffline: boolean;
    },
    forecastArgs: null as null | {
        facility: 1186 | 1656;
        refreshKey: number;
    },
    scheduleArgs: null as null | {
        facility: 1186 | 1656;
        refreshKey: number;
    },
    pollingCalls: [] as Array<{
        intervalMs: number;
        onRefresh: () => void;
        enabled: boolean;
        refreshOnVisible?: boolean;
    }>,
    pullToRefreshArgs: null as null | {onRefresh: () => void},
}));

vi.mock("./hooks/useLiveFacilityData", () => ({
    useLiveFacilityData: (args: NonNullable<typeof appHookHarness.liveArgs>) => {
        appHookHarness.liveArgs = args;
        return liveHookState;
    },
}));

vi.mock("./hooks/useForecastData", () => ({
    useForecastData: (args: NonNullable<typeof appHookHarness.forecastArgs>) => {
        appHookHarness.forecastArgs = args;
        return {
            forecastDays: [],
            forecastOccupancyThresholds: null,
            forecastSectionOccupancyThresholds: {},
            forecastLocationOccupancyThresholds: {},
            forecastHourBounds: {startHour: null, endHour: null},
            forecastError: null,
            isForecastLoading: false,
            hasPendingForecastRetry: false,
        };
    },
}));

vi.mock("./hooks/useFacilityHours", () => ({
    useFacilityHours: (args: NonNullable<typeof appHookHarness.scheduleArgs>) => {
        appHookHarness.scheduleArgs = args;
        return {
            activeSchedule: null,
            isFacilityHoursLoading: false,
            facilityHoursError: null,
            hasPendingScheduleRetry: false,
        };
    },
}));

vi.mock("./hooks/useOnlineStatus", () => ({
    useOnlineStatus: () => appHookHarness.isOffline,
}));

vi.mock("./hooks/useVisibilityPolling", () => ({
    useVisibilityPolling: (args: {
        intervalMs: number;
        onRefresh: () => void;
        enabled: boolean;
        refreshOnVisible?: boolean;
    }) => {
        appHookHarness.pollingCalls.push(args);
    },
}));

vi.mock("./hooks/usePullToRefresh", () => ({
    usePullToRefresh: (args: {onRefresh: () => void}) => {
        appHookHarness.pullToRefreshArgs = args;
        return {
            pullDistance: 0,
            isPulling: false,
            isReadyToRefresh: false,
            showIndicator: false,
            resetPullGesture: vi.fn(),
            handleTouchStart: vi.fn(),
            handleTouchMove: vi.fn(),
            handleTouchEnd: vi.fn(),
        };
    },
}));

vi.mock("./components/AlertsPanel", async () => {
    const React = await vi.importActual<typeof import("react")>("react");
    const AlertStateSentinel = () => {
        const [draftVersion, setDraftVersion] = React.useState(0);
        return React.createElement(
            "button",
            {
                type: "button",
                onClick: () => setDraftVersion((value) => value + 1),
            },
            `Alert draft ${draftVersion}`,
        );
    };
    return {
        default: AlertStateSentinel,
    };
});

vi.mock("../facilities/OccupancyHero", () => ({
    default: ({summary, nowTs}: {summary: OccupancySummary; nowTs: number}) => (
        <div data-testid="occupancy-hero" data-now-ts={nowTs}>
            {summary.status}
        </div>
    ),
}));

vi.mock("../facilities/FloorHeatMapCard", () => ({
    default: () => null,
}));

const canonicalPayload = (fetchedAt: string): FacilityPayload => {
    const powerHouse = {
        facilityId: 1186 as const,
        locationId: 5761,
        locationName: "Power House",
        floor: 0,
        isClosed: false,
        currentCapacity: 20,
        maxCapacity: 100,
        lastUpdated: "2026-08-31T11:59:00Z",
        fetchedAt,
    };

    return {
        facilityId: 1186,
        facilityName: "Nick",
        floors: {0: [powerHouse]},
        locations: [powerHouse],
        liveDataSource: "facility_api",
    };
};

const appTree = (props: {
    initialFacility: 1186;
    onFacilityRouteChange?: (facility: 1186 | 1656) => void;
    themeMode: "light";
    onThemeModeChange: () => void;
}) => (
    <ThemeProvider theme={createAppTheme("light")}>
        <MemoryRouter initialEntries={["/nick"]}>
            <App {...props} />
        </MemoryRouter>
    </ThemeProvider>
);

const latestPollingCall = (intervalMs: number) => {
    const call = [...appHookHarness.pollingCalls]
        .reverse()
        .find((candidate) => candidate.intervalMs === intervalMs);
    if (!call) {
        throw new Error(`Missing polling call for ${intervalMs}ms`);
    }
    return call;
};

afterEach(() => {
    vi.unstubAllEnvs();
});

beforeEach(() => {
    vi.clearAllMocks();
    appHookHarness.isOffline = false;
    appHookHarness.liveArgs = null;
    appHookHarness.forecastArgs = null;
    appHookHarness.scheduleArgs = null;
    appHookHarness.pollingCalls = [];
    appHookHarness.pullToRefreshArgs = null;
    liveHookState.data = null;
    liveHookState.isLoading = false;
    liveHookState.error = null;
    liveHookState.liveDataSource = "facility_api";
    liveHookState.liveOutageState = "none";
    liveHookState.hasPendingLiveRetry = false;
    liveHookState.cacheTimestampMs = null;
    liveHookState.acceptedAtMs = null;
    window.matchMedia = vi.fn().mockImplementation((query: string) => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
    }));
});

describe("App occupancy clock", () => {
    it("removes expired occupancy on a clock tick even when no new response arrives", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
        liveHookState.data = canonicalPayload("2026-08-31T12:00:00Z");
        render(appTree({initialFacility: 1186, themeMode: "light", onThemeModeChange: vi.fn()}));
        expect(screen.getByTestId("occupancy-hero")).toHaveTextContent("live");

        await act(async () => { await vi.advanceTimersByTimeAsync(630_000); });
        expect(screen.queryByTestId("occupancy-hero")).not.toBeInTheDocument();
        expect(screen.getByRole("heading", {name: "RecLive is unavailable."})).toBeVisible();
    });

    it("ticks the live clock and removes its timer and debug handlers on unmount", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
        const setInterval = vi.spyOn(window, "setInterval");
        const clearInterval = vi.spyOn(window, "clearInterval");
        liveHookState.data = canonicalPayload("2026-08-31T11:59:00Z");
        const {unmount} = render(appTree({initialFacility: 1186, themeMode: "light", onThemeModeChange: vi.fn()}));
        await act(async () => { await Promise.resolve(); });
        act(() => vi.advanceTimersByTime(30_000));
        expect(screen.getByTestId("occupancy-hero")).toHaveAttribute("data-now-ts", String(Date.parse("2026-08-31T12:00:30Z")));
        const clockIntervalIndex = setInterval.mock.calls.findIndex(([, delay]) => delay === 30_000);
        expect(clockIntervalIndex).toBeGreaterThanOrEqual(0);
        const clockInterval = setInterval.mock.results[clockIntervalIndex].value;
        unmount();
        expect(clearInterval).toHaveBeenCalledWith(clockInterval);
        expect(window.recliveSetDebugNow).toBeUndefined();
        expect(window.recliveDebugDashboardState).toBeUndefined();
    });

    it("ignores persisted debug state and removes debug globals in normal production", () => {
        vi.stubEnv("DEV", false);
        vi.stubEnv("MODE", "production");
        window.localStorage.setItem("reclive:closureOverride", "true");
        window.localStorage.setItem("reclive:debugNow", "2026-08-31T12:00:00Z");
        const getItem = vi.spyOn(Storage.prototype, "getItem");
        window.recliveShowPredictions = () => "stale";
        window.recliveRestoreWarnings = () => "stale";
        window.reclivePredictionOverrideStatus = () => true;
        window.recliveOverrideClosure = () => "stale";
        window.recliveRestoreClosure = () => "stale";
        window.recliveClosureOverrideStatus = () => true;
        window.recliveSetDebugNow = () => "stale";
        window.recliveClearDebugNow = () => "stale";
        window.recliveDebugNowStatus = () => "stale";
        window.recliveDebugDashboardState = () => ({stale: true});

        render(appTree({
            initialFacility: 1186,
            themeMode: "light",
            onThemeModeChange: vi.fn(),
        }));

        expect(getItem.mock.calls.map(([key]) => key)).not.toContain("reclive:closureOverride");
        expect(getItem.mock.calls.map(([key]) => key)).not.toContain("reclive:debugNow");
        expect(window.recliveShowPredictions).toBeUndefined();
        expect(window.recliveRestoreWarnings).toBeUndefined();
        expect(window.reclivePredictionOverrideStatus).toBeUndefined();
        expect(window.recliveOverrideClosure).toBeUndefined();
        expect(window.recliveRestoreClosure).toBeUndefined();
        expect(window.recliveClosureOverrideStatus).toBeUndefined();
        expect(window.recliveSetDebugNow).toBeUndefined();
        expect(window.recliveClearDebugNow).toBeUndefined();
        expect(window.recliveDebugNowStatus).toBeUndefined();
        expect(window.recliveDebugDashboardState).toBeUndefined();
    });

    it("samples the render time when a canonical payload arrives between clock ticks", async () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
        liveHookState.data = null;
        const props = {
            initialFacility: 1186 as const,
            themeMode: "light" as const,
            onThemeModeChange: vi.fn(),
        };
        const {rerender} = render(appTree(props));

        vi.setSystemTime("2026-08-31T12:00:10Z");
        liveHookState.data = canonicalPayload("2026-08-31T12:00:10Z");
        liveHookState.acceptedAtMs = Date.parse("2026-08-31T12:00:10Z");
        rerender(appTree(props));
        await act(async () => {
            await Promise.resolve();
        });

        expect(screen.getByTestId("occupancy-hero")).toHaveTextContent("live");
        expect(screen.getByTestId("occupancy-hero")).toHaveAttribute(
            "data-now-ts",
            String(Date.parse("2026-08-31T12:00:10Z"))
        );
    });

    it("keeps the debug clock override authoritative", () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:05:00Z");
        window.localStorage.setItem("reclive:debugNow", "2026-08-31T12:00:00Z");
        liveHookState.data = canonicalPayload("2026-08-31T12:00:10Z");

        render(appTree({
            initialFacility: 1186,
            themeMode: "light",
            onThemeModeChange: vi.fn(),
        }));

        expect(screen.queryByTestId("occupancy-hero")).not.toBeInTheDocument();
        expect(screen.getByRole("heading", {name: "RecLive is unavailable."})).toBeVisible();
        expect(window.recliveDebugDashboardState?.()).toMatchObject({
            nowTs: Date.parse("2026-08-31T12:00:00Z"), occupancyStatus: "insufficient",
        });
    });
});

describe("App refresh coordination", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
        liveHookState.data = canonicalPayload("2026-08-31T11:59:00Z");
    });
    it("enforces manual-refresh cooldown and clears it when selecting another facility", () => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
        const onFacilityRouteChange = vi.fn();
        render(appTree({initialFacility: 1186, onFacilityRouteChange, themeMode: "light", onThemeModeChange: vi.fn()}));
        const refresh = () => {
            const pull = appHookHarness.pullToRefreshArgs;
            if (!pull) throw new Error("Missing pull refresh action");
            act(() => pull.onRefresh());
        };
        refresh();
        refresh();
        expect(appHookHarness.liveArgs?.refreshKey).toBe(1);
        act(() => vi.advanceTimersByTime(2999));
        refresh();
        expect(appHookHarness.liveArgs?.refreshKey).toBe(1);
        act(() => vi.advanceTimersByTime(1));
        refresh();
        expect(appHookHarness.liveArgs?.refreshKey).toBe(2);
        fireEvent.click(screen.getByRole("button", {name: "Bakke"}));
        expect(onFacilityRouteChange).toHaveBeenCalledExactlyOnceWith(1656);
        expect(window.localStorage.getItem("reclive:selectedFacility")).toBe("1656");
        refresh();
        expect(appHookHarness.liveArgs).toMatchObject({facility: 1656, refreshKey: 3});
        expect(appHookHarness.forecastArgs?.refreshKey).toBe(0);
        expect(appHookHarness.scheduleArgs?.refreshKey).toBe(0);
    });

    it("does not announce the initial live query completing", () => {
        liveHookState.isLoading = true;
        const props = {
            initialFacility: 1186 as const,
            themeMode: "light" as const,
            onThemeModeChange: vi.fn(),
        };
        const {rerender} = render(appTree(props));

        liveHookState.isLoading = false;
        rerender(appTree(props));

        expect(screen.queryByRole("status")).not.toBeInTheDocument();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("announces only a completed manual refresh", async () => {
        const props = {
            initialFacility: 1186 as const,
            themeMode: "light" as const,
            onThemeModeChange: vi.fn(),
        };
        const {rerender} = render(appTree(props));
        const pullToRefresh = appHookHarness.pullToRefreshArgs;
        if (!pullToRefresh) throw new Error("Pull-to-refresh was not configured");

        act(() => pullToRefresh.onRefresh());
        expect(screen.getByRole("status")).toHaveTextContent("Refreshing live occupancy");

        liveHookState.isLoading = true;
        rerender(appTree(props));
        liveHookState.isLoading = false;
        rerender(appTree(props));
        await act(async () => {
            await Promise.resolve();
        });

        expect(screen.getByRole("status")).toHaveTextContent("Live occupancy updated");
    });

    it("announces a retained-data manual refresh failure from retry evidence", async () => {
        liveHookState.data = canonicalPayload("2026-08-31T12:00:00Z");
        const props = {
            initialFacility: 1186 as const,
            themeMode: "light" as const,
            onThemeModeChange: vi.fn(),
        };
        const {rerender} = render(appTree(props));
        await act(async () => {
            await Promise.resolve();
        });
        const pullToRefresh = appHookHarness.pullToRefreshArgs;
        if (!pullToRefresh) throw new Error("Pull-to-refresh was not configured");

        act(() => pullToRefresh.onRefresh());
        liveHookState.isLoading = true;
        rerender(appTree(props));
        liveHookState.hasPendingLiveRetry = true;
        liveHookState.isLoading = false;
        rerender(appTree(props));
        await act(async () => {
            await Promise.resolve();
        });

        expect(screen.getByRole("status")).toHaveTextContent("Couldn't refresh. Try again.");
        expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite");
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    });

    it("does not announce polling completion and clears a pending manual announcement on facility switch", () => {
        const props = {
            initialFacility: 1186 as const,
            themeMode: "light" as const,
            onThemeModeChange: vi.fn(),
        };
        const {rerender} = render(appTree(props));

        act(() => latestPollingCall(90_000).onRefresh());
        liveHookState.isLoading = true;
        rerender(appTree(props));
        liveHookState.isLoading = false;
        rerender(appTree(props));
        expect(screen.queryByRole("status")).not.toBeInTheDocument();

        const pullToRefresh = appHookHarness.pullToRefreshArgs;
        if (!pullToRefresh) throw new Error("Pull-to-refresh was not configured");
        act(() => pullToRefresh.onRefresh());
        expect(screen.getByRole("status")).toHaveTextContent("Refreshing live occupancy");

        fireEvent.click(screen.getByRole("button", {name: "Bakke"}));
        expect(screen.queryByText("Refreshing live occupancy")).not.toBeInTheDocument();
        expect(screen.getByRole("status")).toHaveTextContent("Loading...");
    });

    it("wires independent live, forecast, and schedule polling keys", () => {
        render(appTree({
            initialFacility: 1186,
            themeMode: "light",
            onThemeModeChange: vi.fn(),
        }));

        expect(latestPollingCall(90_000)).toMatchObject({
            enabled: true,
            refreshOnVisible: true,
        });
        expect(latestPollingCall(15 * 60_000)).toMatchObject({
            enabled: true,
            refreshOnVisible: false,
        });
        expect(latestPollingCall(4 * 60 * 60_000)).toMatchObject({
            enabled: true,
            refreshOnVisible: false,
        });
        expect(appHookHarness.liveArgs?.refreshKey).toBe(0);
        expect(appHookHarness.forecastArgs?.refreshKey).toBe(0);
        expect(appHookHarness.scheduleArgs?.refreshKey).toBe(0);

        act(() => latestPollingCall(15 * 60_000).onRefresh());
        expect(appHookHarness.liveArgs?.refreshKey).toBe(0);
        expect(appHookHarness.forecastArgs?.refreshKey).toBe(1);
        expect(appHookHarness.scheduleArgs?.refreshKey).toBe(0);

        act(() => latestPollingCall(4 * 60 * 60_000).onRefresh());
        expect(appHookHarness.liveArgs?.refreshKey).toBe(0);
        expect(appHookHarness.forecastArgs?.refreshKey).toBe(1);
        expect(appHookHarness.scheduleArgs?.refreshKey).toBe(1);

        act(() => latestPollingCall(90_000).onRefresh());
        expect(appHookHarness.liveArgs?.refreshKey).toBe(1);
        expect(appHookHarness.forecastArgs?.refreshKey).toBe(1);
        expect(appHookHarness.scheduleArgs?.refreshKey).toBe(1);
    });

    it("disables all polling domains while offline", () => {
        appHookHarness.isOffline = true;

        render(appTree({
            initialFacility: 1186,
            themeMode: "light",
            onThemeModeChange: vi.fn(),
        }));

        expect(latestPollingCall(90_000).enabled).toBe(false);
        expect(latestPollingCall(15 * 60_000).enabled).toBe(false);
        expect(latestPollingCall(4 * 60 * 60_000).enabled).toBe(false);
        expect(appHookHarness.liveArgs?.isOffline).toBe(true);
    });

    it("pull-to-refresh increments only the live refresh key", () => {
        render(appTree({
            initialFacility: 1186,
            themeMode: "light",
            onThemeModeChange: vi.fn(),
        }));

        const pullToRefresh = appHookHarness.pullToRefreshArgs;
        if (!pullToRefresh) {
            throw new Error("Pull-to-refresh was not configured");
        }
        act(() => pullToRefresh.onRefresh());

        expect(appHookHarness.liveArgs?.refreshKey).toBe(1);
        expect(appHookHarness.forecastArgs?.refreshKey).toBe(0);
        expect(appHookHarness.scheduleArgs?.refreshKey).toBe(0);
        expect(liveHookState.prepareRefresh).toHaveBeenCalledTimes(1);
    });

    it("keeps alert-management state mounted across a live refresh", () => {
        render(appTree({
            initialFacility: 1186,
            themeMode: "light",
            onThemeModeChange: vi.fn(),
        }));

        fireEvent.click(screen.getByRole("button", {name: "Alert draft 0"}));
        expect(screen.getByRole("button", {name: "Alert draft 1"})).toBeInTheDocument();

        act(() => latestPollingCall(90_000).onRefresh());

        expect(screen.getByRole("button", {name: "Alert draft 1"})).toBeInTheDocument();
    });

    it("keeps the active dashboard mounted when a new observation arrives between clock ticks", async () => {
        const props = {initialFacility: 1186 as const, themeMode: "light" as const, onThemeModeChange: vi.fn()};
        const {rerender} = render(appTree(props));
        await act(async () => { await Promise.resolve(); });
        fireEvent.click(screen.getByRole("button", {name: "Alert draft 0"}));

        vi.setSystemTime("2026-08-31T12:00:10Z");
        liveHookState.data = canonicalPayload("2026-08-31T12:00:10Z");
        liveHookState.acceptedAtMs = Date.parse("2026-08-31T12:00:10Z");
        rerender(appTree(props));
        await act(async () => { await Promise.resolve(); });

        expect(screen.getByRole("button", {name: "Alert draft 1"})).toBeInTheDocument();
        expect(screen.queryByText("RecLive is unavailable.")).not.toBeInTheDocument();
    });
});
