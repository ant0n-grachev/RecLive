import {expect, test} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import {installDashboardApiMocks} from "./support/apiMocks";

test.beforeEach(async ({page}) => {
    await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
    await installDashboardApiMocks(page);
});

for (const routePath of ["/nick", "/bakke"]) {
    test(`${routePath} loads a visible dashboard shell without critical axe violations`, async ({page}) => {
        await page.goto(routePath);
        await expect(page.locator("main")).not.toBeEmpty();
        await expect(page).toHaveURL(new RegExp(`${routePath}$`));
        await expect(page.getByText("Live Occupancy", {exact: true})).toBeVisible();
        await expect(page.getByRole("button", {name: "Alerts", exact: true})).toBeVisible();
        await expect(page.getByText("Train smarter. Skip the crowd.").first()).toBeVisible();
        await expect(page.getByRole("progressbar", {name: "Current occupancy percentage"})).toBeVisible();

        const violations = (await new AxeBuilder({page}).withTags(["wcag2a", "wcag2aa"]).analyze()).violations
            .filter((violation) => violation.impact === "critical");
        expect(violations).toEqual([]);

        await page.getByRole("button", {name: "Alerts", exact: true}).click();
        const alertsDialog = page.getByRole("dialog", {name: "Alerts", exact: true});
        await expect(alertsDialog).toBeVisible();
        const manageAlerts = alertsDialog.getByRole("region", {name: "Manage alerts"});
        await expect(manageAlerts).toBeVisible();
        await expect(manageAlerts.getByRole("heading", {name: "Manage alerts"})).toBeVisible();
    });
}

test("dashboard API fixture rejects a duplicated API path prefix", async ({page}) => {
    await page.goto("/nick");
    const status = await page.evaluate(async () => (
        await fetch("/api/api/live-counts")
    ).status);
    expect(status).toBe(404);
});

test("dashboard API fixture accepts only the canonical actual-hours path", async ({page}) => {
    await page.goto("/nick");
    const statuses = await page.evaluate(async () => Promise.all([
        fetch("/api/forecast/facilities/1186/actual-hours").then((response) => response.status),
        fetch("/api/api/forecast/facilities/1186/actual-hours").then((response) => response.status),
    ]));
    expect(statuses).toEqual([200, 404]);
});
