import {ThemeProvider} from "@mui/material";
import {act, render, screen} from "@testing-library/react";
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

vi.mock("./hooks/useLiveFacilityData", () => ({
    useLiveFacilityData: () => liveHookState,
}));

vi.mock("./hooks/useForecastData", () => ({
    useForecastData: () => ({
        forecastDays: [],
        forecastOccupancyThresholds: null,
        forecastSectionOccupancyThresholds: {},
        forecastLocationOccupancyThresholds: {},
        forecastHourBounds: {startHour: null, endHour: null},
        forecastError: null,
        isForecastLoading: false,
        hasPendingForecastRetry: false,
    }),
}));

vi.mock("./hooks/useFacilityHours", () => ({
    useFacilityHours: () => ({
        activeSchedule: null,
        isFacilityHoursLoading: false,
        facilityHoursError: null,
        hasPendingScheduleRetry: false,
    }),
}));

vi.mock("./hooks/useOnlineStatus", () => ({
    useOnlineStatus: () => false,
}));

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

describe("App occupancy clock", () => {
    beforeEach(() => {
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
