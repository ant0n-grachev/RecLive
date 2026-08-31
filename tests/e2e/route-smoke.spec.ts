import {expect, test} from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

const liveRows = [{LocationId: 1, IsClosed: false, LastCount: 2, LastUpdatedDateAndTime: "2026-08-31T12:00:00Z"}];
const emptyForecast = {generatedAt: "2026-08-31T12:00:00Z", facilities: []};
const emptySchedule = {generatedAt: "2026-08-31T12:00:00Z", facilities: []};

test.beforeEach(async ({page}) => {
    await page.route("**/api/live-counts", (route) => route.fulfill({json: liveRows}));
    await page.route("**/api/forecast/**", (route) => route.fulfill({json: emptyForecast}));
    await page.route("**/api/facility-hours/**", (route) => route.fulfill({json: emptySchedule}));
});

for (const routePath of ["/nick", "/bakke"]) {
    test(`${routePath} loads a visible dashboard shell without critical axe violations`, async ({page}) => {
        await page.goto(routePath);
        await expect(page.locator("#root")).not.toBeEmpty();
        await expect(page).toHaveURL(new RegExp(`${routePath}$`));
        await expect(page.getByText("Train smarter. Skip the crowd.").first()).toBeVisible();

        const violations = (await new AxeBuilder({page}).withTags(["wcag2a", "wcag2aa"]).analyze()).violations
            .filter((violation) => violation.impact === "critical");
        expect(violations).toEqual([]);
    });
}
