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

        expect(screen.getByTestId("occupancy-hero")).toHaveTextContent("insufficient");
        expect(screen.getByTestId("occupancy-hero")).toHaveAttribute(
            "data-now-ts",
            String(Date.parse("2026-08-31T12:00:00Z"))
        );
    });
});

describe("App refresh coordination", () => {
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
});
