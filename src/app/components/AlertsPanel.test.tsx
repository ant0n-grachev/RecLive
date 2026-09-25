import {ThemeProvider} from "@mui/material";
import {fireEvent, render, screen, waitFor, within} from "@testing-library/react";
import {createAppTheme} from "../theme";
import AlertsPanel from "./AlertsPanel";
import {liveSummary} from "../../test/fixtures/dashboard";

const pushApi = vi.hoisted(() => ({
    getPushAvailability: vi.fn(),
    getExistingPushSubscription: vi.fn(),
    isWebPushSupported: vi.fn(),
}));
vi.mock("../../lib/api/pushNotifications", () => pushApi);

beforeEach(() => {
    pushApi.getPushAvailability.mockResolvedValue({alertsAvailable: true});
    pushApi.getExistingPushSubscription.mockResolvedValue(null);
    pushApi.isWebPushSupported.mockReturnValue(true);
});

const renderPanel = (useDesktopModal: boolean, onClose = vi.fn(), isTouchCapable = false) => render(
    <ThemeProvider theme={createAppTheme("light")}>
        <AlertsPanel
            open
            onClose={onClose}
            facility={1186}
            sections={[{key: "overall", label: "Entire Facility", summary: liveSummary}]}
            useDesktopModal={useDesktopModal}
            isStandalonePwa={false}
            isTouchCapable={isTouchCapable}
        />
    </ThemeProvider>
);

it.each([
    ["desktop modal", true],
    ["mobile drawer", false],
])("labels the %s as the Alerts dialog", async (_label, useDesktopModal) => {
    renderPanel(useDesktopModal);

    const dialog = await screen.findByRole("dialog", {name: "Alerts"});
    expect(dialog).toBeVisible();
    expect(await within(dialog).findByRole("region", {name: "Manage alerts"})).toBeVisible();
    expect(await within(dialog).findByText("No active browser subscription was found.")).toBeVisible();
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    expect(within(dialog).getByRole("combobox", {name: "Gym area"})).toBeVisible();
    expect(screen.getByRole("button", {name: "Close alerts"})).toBeVisible();
});

it("keeps closing with the outer panel and gates only creation on touch mobile", async () => {
    const onClose = vi.fn();
    renderPanel(false, onClose, true);
    await screen.findByRole("region", {name: "Manage alerts"});
    expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveAttribute("aria-disabled", "true");
    expect(screen.getByRole("button", {name: "Set alert"})).toBeDisabled();
    await waitFor(() => expect(screen.getByText("No active browser subscription was found.")).toBeVisible());
    fireEvent.click(screen.getByRole("button", {name: "Close alerts"}));
    expect(onClose).toHaveBeenCalledTimes(1);
});
