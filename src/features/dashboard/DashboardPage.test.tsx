import {ThemeProvider} from "@mui/material";
import {act, fireEvent, render, screen, waitFor} from "@testing-library/react";
import type {PaletteMode} from "@mui/material/styles";
import {createAppTheme} from "../../app/theme";
import type {FacilityScheduleResponse} from "../../lib/api/schemas";
import type {DashboardState} from "./dashboardTypes";
import {
    createDashboardStateForTest,
    fixtureLiveByFacility,
    fixtureScheduleByFacility,
} from "../../test/fixtures/dashboard";
import {DashboardPage, type DashboardPageProps} from "./DashboardPage";

const pushApi = vi.hoisted(() => ({
    getPushAvailability: vi.fn(),
    getExistingPushSubscription: vi.fn(),
    isWebPushSupported: vi.fn(),
}));
vi.mock("../../lib/api/pushNotifications", () => pushApi);

const onThemeModeChange = vi.fn<(mode: PaletteMode) => void>();

const createPageProps = (overrides: Partial<DashboardState> = {}): DashboardPageProps => {
    const state = createDashboardStateForTest(overrides);
    return {state, view: state.view, themeMode: "light", onThemeModeChange};
};

const renderPage = (props: DashboardPageProps = createPageProps()) => render(
    <ThemeProvider theme={createAppTheme(props.themeMode)}>
        <DashboardPage {...props}/>
    </ThemeProvider>
);

// Await real lazy imports without advancing the fixture's fake clock or relying on test order.
const settlePageImports = () => act(async () => {
    await vi.dynamicImportSettled();
});

const expectTextOrder = (container: HTMLElement, labels: string[]) => {
    let previousIndex = -1;
    for (const label of labels) {
        const currentIndex = container.textContent?.indexOf(label) ?? -1;
        expect(currentIndex, `Expected ${label} after the preceding dashboard card`).toBeGreaterThan(previousIndex);
        previousIndex = currentIndex;
    }
};

const closedMondaySchedule: FacilityScheduleResponse = {
    ...fixtureScheduleByFacility[1186],
    sections: [{
        title: "Building Hours",
        rows: [
            {label: "Monday", hours: "Closed"},
            {label: "Tuesday", hours: "6am - 11pm"},
        ],
        note: null,
    }],
};

beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "info").mockImplementation(() => {});
    pushApi.getPushAvailability.mockResolvedValue({alertsAvailable: true});
    pushApi.getExistingPushSubscription.mockResolvedValue(null);
    pushApi.isWebPushSupported.mockReturnValue(true);
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

afterEach(() => {
    vi.useRealTimers();
    vi.restoreAllMocks();
});

it("shows the lazy heatmap fallback before the page card resolves", async () => {
    const {container} = renderPage();

    expect(container.querySelector(".MuiCircularProgress-root")).toBeInTheDocument();
    await settlePageImports();
    expect(await screen.findByText("Floor Heat Map")).toBeVisible();
});

it.each([
    [1186, "Nick"],
    [1656, "Bakke"],
] as const)("renders facility %s with its route content and controls", async (facility, label) => {
    vi.useFakeTimers();
    vi.setSystemTime("2026-08-31T13:15:00Z");
    renderPage(createPageProps({
        facility,
        data: fixtureLiveByFacility[facility],
        activeSchedule: fixtureScheduleByFacility[facility],
    }));
    await settlePageImports();

    expect(screen.getByRole("main")).toBeVisible();
    expect(screen.getByRole("button", {name: label})).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("link", {name: label})).toBeVisible();
    expect(screen.getByText("Live Occupancy")).toBeVisible();
    expect(screen.getByRole("button", {name: "Alerts"})).toBeVisible();
    expect(screen.getByText("Forecast Today")).toBeVisible();
    expect(screen.getByRole("button", {name: "Switch to dark theme"})).toBeVisible();
    expect(screen.getByRole("button", {name: "How to add RecLive to your home screen"})).toBeVisible();
    expect(screen.getByText("Floor Heat Map")).toBeVisible();
});

it("preserves the open dashboard card order inside the main landmark", async () => {
    vi.useFakeTimers();
    vi.setSystemTime("2026-08-31T13:15:00Z");
    renderPage();
    await settlePageImports();
    const main = screen.getByRole("main");

    expectTextOrder(main, [
        "Live Occupancy",
        "Schedule Status",
        "Forecast Today",
        "Floor Heat Map",
        "Fitness Floors",
        "Official Hours, Closures & Notices",
        "How to add RecLive to your home screen",
    ]);
});

it("renders loading and error states without stale facility cards", () => {
    const loadingProps = createPageProps({data: null, isLoading: true});
    const {rerender} = renderPage(loadingProps);

    expect(screen.getByText("Loading...")).toBeVisible();
    expect(screen.queryByText("Live Occupancy")).not.toBeInTheDocument();

    const errorProps = createPageProps({data: null, isLoading: false, error: "Live data unavailable."});
    rerender(
        <ThemeProvider theme={createAppTheme(errorProps.themeMode)}>
            <DashboardPage {...errorProps}/>
        </ThemeProvider>
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Live data unavailable.");
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    expect(screen.queryByText("Live Occupancy")).not.toBeInTheDocument();
});

it("keeps closed-mode copy, tomorrow forecast controls, and theme control", () => {
    vi.useFakeTimers();
    vi.setSystemTime("2026-08-31T13:15:00Z");
    renderPage(createPageProps({activeSchedule: closedMondaySchedule}));

    expect(screen.getByText("Closed now according to the official schedule")).toBeVisible();
    expect(screen.getByText("Forecast Tomorrow")).toBeVisible();
    expect(screen.getByRole("button", {name: "Next forecast day"})).toBeDisabled();
    expect(screen.getByRole("button", {name: "Switch to dark theme"})).toBeVisible();
    expect(screen.queryByText("Live Occupancy")).not.toBeInTheDocument();
    expect(screen.queryByText("Floor Heat Map")).not.toBeInTheDocument();
});

it("announces live status while preserving an open alert draft across live refresh", async () => {
    const initialProps = createPageProps({isCrowdAlertOpen: true, liveStatus: "refreshing"});
    const {rerender} = renderPage(initialProps);
    await settlePageImports();
    const threshold = await screen.findByRole("spinbutton", {name: "Alert threshold (%)"});
    fireEvent.change(threshold, {target: {value: "7"}});

    expect(screen.getByRole("status", {hidden: true})).toHaveTextContent("Refreshing live occupancy");
    expect(threshold).toHaveValue(7);

    const refreshedProps = createPageProps({
        isCrowdAlertOpen: true,
        liveStatus: "updated",
        nowTs: initialProps.state.nowTs + 30_000,
        data: {...fixtureLiveByFacility[1186]},
    });
    rerender(
        <ThemeProvider theme={createAppTheme(refreshedProps.themeMode)}>
            <DashboardPage {...refreshedProps}/>
        </ThemeProvider>
    );

    expect(screen.getByRole("status", {hidden: true})).toHaveTextContent("Live occupancy updated");
    await waitFor(() => expect(screen.getByRole("spinbutton", {name: "Alert threshold (%)"})).toHaveValue(7));
});
