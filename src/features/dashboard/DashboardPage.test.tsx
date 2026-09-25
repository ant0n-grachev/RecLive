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

    const errorProps = createPageProps({data: null, isLoading: false, error: "Live data unavailable.", manualRefresh: vi.fn()});
    rerender(
        <ThemeProvider theme={createAppTheme(errorProps.themeMode)}>
            <DashboardPage {...errorProps}/>
        </ThemeProvider>
    );

    expect(screen.getByRole("heading", {name: "RecLive is unavailable."})).toBeVisible();
    expect(screen.queryByLabelText("Current count unavailable")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", {name: "Try again"}));
    expect(errorProps.state.manualRefresh).toHaveBeenCalledOnce();
    expect(screen.queryByText("Loading...")).not.toBeInTheDocument();
    expect(screen.queryByText("Live Occupancy")).not.toBeInTheDocument();
    expect(screen.queryByText("Floor Heat Map")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", {name: "How to add RecLive to your home screen"})).not.toBeInTheDocument();
});

it("replaces expired readings with one unavailable view and recovers without a reload", async () => {
    const expired = createPageProps({nowTs: Date.parse("2026-08-31T12:10:01Z")});
    const {rerender} = renderPage(expired);
    await settlePageImports();

    expect(screen.getByRole("heading", {name: "RecLive is unavailable."})).toBeVisible();
    expect(screen.queryByText("Live Occupancy")).not.toBeInTheDocument();
    expect(screen.queryByText("Forecast Today")).not.toBeInTheDocument();
    expect(screen.queryByText("—")).not.toBeInTheDocument();
    expect(screen.getByRole("button", {name: "Bakke"})).toBeEnabled();

    const recovered = createPageProps();
    rerender(<ThemeProvider theme={createAppTheme("light")}><DashboardPage {...recovered}/></ThemeProvider>);
    await settlePageImports();
    expect(screen.queryByText("RecLive is unavailable.")).not.toBeInTheDocument();
    expect(screen.getByText("Live Occupancy")).toBeVisible();
});

it("does not mistake a confirmed live-source closure for an outage", async () => {
    const locations = fixtureLiveByFacility[1186].locations.map((location) => ({...location, isClosed: true}));
    renderPage(createPageProps({data: {...fixtureLiveByFacility[1186], locations}, activeSchedule: null}));
    await settlePageImports();
    expect(screen.getByRole("heading", {name: "CLOSED"})).toBeVisible();
    expect(screen.queryByText("RecLive is unavailable.")).not.toBeInTheDocument();
});

it("keeps usable readings visible during refresh without flashing the outage view", async () => {
    renderPage(createPageProps({isLoading: true, hasPendingLiveRetry: true}));
    await settlePageImports();
    expect(screen.getByText("Live Occupancy")).toBeVisible();
    expect(screen.queryByText("RecLive is unavailable.")).not.toBeInTheDocument();
});

it.each([
    {isOffline: true, liveOutageState: "cache" as const, liveDataSource: "cache" as const},
    {liveOutageState: "cache" as const, liveDataSource: "cache" as const},
    {liveDataSource: "fallback_api" as const},
    {forecastError: "Forecast unavailable right now."},
])("keeps automatic data diagnostics off the dashboard: %j", async (overrides) => {
    renderPage(createPageProps(overrides));
    await settlePageImports();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(screen.queryByText(/backup feed|saved snapshot|temporarily unavailable|observed-capacity coverage/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", {name: "Alerts"})).toBeVisible();
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
