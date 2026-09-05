import {ThemeProvider} from "@mui/material";
import {render, screen} from "@testing-library/react";
import {createAppTheme} from "../theme";
import AlertsPanel from "./AlertsPanel";

vi.mock("../../facilities/CrowdAlertSubscriptionCard", () => ({
    default: () => <section aria-label="Manage alerts">Alert controls</section>,
}));

const renderPanel = (useDesktopModal: boolean) => render(
    <ThemeProvider theme={createAppTheme("light")}>
        <AlertsPanel
            open
            onClose={vi.fn()}
            facility={1186}
            sections={[]}
            useDesktopModal={useDesktopModal}
            isStandalonePwa={false}
            isTouchCapable={false}
        />
    </ThemeProvider>
);

it.each([
    ["desktop modal", true],
    ["mobile drawer", false],
])("labels the %s as the Alerts dialog", async (_label, useDesktopModal) => {
    renderPanel(useDesktopModal);

    expect(await screen.findByRole("dialog", {name: "Alerts"})).toBeVisible();
    expect(screen.getByRole("button", {name: "Close alerts"})).toBeVisible();
});
