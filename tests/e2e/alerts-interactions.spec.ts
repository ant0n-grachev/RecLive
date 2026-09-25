import AxeBuilder from "@axe-core/playwright";
import {expect, test} from "@playwright/test";
import {installDashboardApiMocks} from "./support/apiMocks";

for (const facility of [{path: "/nick", name: "Nick"}, {path: "/bakke", name: "Bakke"}]) {
    test(`${facility.name} inner alert body creates and cancels within one labelled dialog`, async ({page}, testInfo) => {
        await page.emulateMedia({reducedMotion: "reduce"});
        await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
        await installDashboardApiMocks(page);
        // This UI test supplies a deterministic existing browser owner. Worker
        // registration/readiness remains covered separately by the PWA tests.
        await page.addInitScript(() => {
            Object.defineProperty(Notification, "permission", {get: () => "granted"});
            Object.defineProperty(navigator.serviceWorker, "ready", {
                get: () => Promise.resolve({
                    pushManager: {
                        getSubscription: async () => ({toJSON: () => ({
                            endpoint: "https://push.example.test/alert-fixture",
                            keys: {p256dh: "fixture-key", auth: "fixture-auth"},
                        })}),
                    },
                }),
            });
        });
        const pushRequests: string[] = [];
        page.on("request", (request) => {
            const path = new URL(request.url()).pathname;
            if (path.startsWith("/api/push/")) pushRequests.push(path);
        });
        await page.goto(facility.path);
        await expect(page.getByRole("progressbar", {name: "Current occupancy percentage"})).toBeVisible();
        await page.getByRole("button", {name: "Alerts", exact: true}).click();
        const dialog = page.getByRole("dialog", {name: "Alerts", exact: true});
        await expect(dialog.getByText("No active alerts for this browser.")).toBeVisible();
        await expect(page.getByRole("dialog")).toHaveCount(1);
        await expect(dialog.getByRole("region", {name: "Manage alerts"})).toBeVisible();
        const threshold = dialog.getByRole("spinbutton", {name: "Alert threshold (%)"});
        await threshold.fill("1");
        await dialog.getByRole("button", {name: "Set alert", exact: true}).click();
        await expect(dialog.getByText("Alert set successfully.")).toBeVisible();
        await expect(dialog).toBeVisible();
        await expect(dialog.getByRole("listitem")).toHaveCount(1);
        await expect(dialog.getByRole("status")).toHaveText("Occupancy alert created.");
        expect(pushRequests.filter((path) => path === "/api/push/rules/list")).toHaveLength(1);
        expect(pushRequests.filter((path) => path === "/api/push/subscribe")).toHaveLength(1);
        expect(pushRequests).not.toContain("/api/push/public-key");

        const accessibility = await new AxeBuilder({page})
            .include('[role="dialog"][aria-labelledby="alerts-panel-title"]')
            .analyze();
        expect(accessibility.violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""))).toEqual([]);
        await page.screenshot({path: testInfo.outputPath(`${facility.name.toLowerCase()}-active-alert.png`), fullPage: true});

        await dialog.getByRole("button", {name: `Cancel alert for ${facility.name} Entire Facility at 1%`, exact: true}).click();
        await expect(dialog.getByText("Alert cancelled.")).toHaveAttribute("role", "status");
        await expect(dialog.getByRole("listitem")).toHaveCount(0);
        await expect(dialog).toBeVisible();
        await dialog.getByRole("button", {name: "Close alerts"}).click();
        await expect(dialog).toBeHidden();
    });
}
