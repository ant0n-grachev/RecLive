import {expect, test} from "@playwright/test";
import {installDashboardApiMocks} from "./support/apiMocks";

const viewports = [
    {name: "desktop", width: 1280, height: 900},
    {name: "mobile", width: 390, height: 844},
] as const;

for (const viewport of viewports) {
    test(`${viewport.name} forecast chart preserves expansion, keyboard details, and filtering`, async ({page}, testInfo) => {
        const consoleProblems: string[] = [];
        const pageErrors: string[] = [];
        page.on("console", (message) => {
            if (["warning", "error"].includes(message.type())) consoleProblems.push(message.text());
        });
        page.on("pageerror", (error) => pageErrors.push(error.message));

        await page.setViewportSize({width: viewport.width, height: viewport.height});
        await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
        await page.emulateMedia({reducedMotion: "reduce"});
        await installDashboardApiMocks(page, {forecastExpectedPct: 0.5});
        await page.goto("/nick");

        await expect(page).toHaveURL(/\/nick$/);
        await expect(page).toHaveTitle(/Nick/i);
        await expect(page.getByText("Forecast Today")).toBeVisible();
        await expect(page.getByRole("button", {name: "Show crowd chart"})).toBeVisible();
        await expect(page.locator("vite-error-overlay")).toHaveCount(0);

        await page.getByRole("button", {name: "LOW", exact: true}).click();
        await expect(page.getByText("No matching intervals.")).toBeVisible();
        await page.getByRole("button", {name: "LOW", exact: true}).click();
        await expect(page.getByText("9:00 AM – 10:00 AM")).toBeVisible();

        await page.getByRole("button", {name: "Show crowd chart"}).click();
        const chart = page.getByRole("img", {name: "People by hour"});
        await expect(chart).toBeVisible();
        const bar = chart.getByRole("button", {name: "9:00 AM – 10:00 AM, 50 people"});
        await bar.focus();
        await page.keyboard.press("Enter");
        await expect(page.getByText("9:00 AM – 10:00 AM: 50")).toBeVisible();

        await page.screenshot({
            path: testInfo.outputPath(`nick-${viewport.name}-expanded-selected.png`),
            fullPage: true,
        });

        await page.getByRole("button", {name: "Hide crowd chart"}).click();
        await expect(chart).toBeHidden();
        expect(pageErrors).toEqual([]);
        expect(consoleProblems).toEqual([]);
    });
}
