import AxeBuilder from "@axe-core/playwright";
import {expect, test, type Locator, type Page} from "@playwright/test";
import {installDashboardApiMocks, removeDashboardApiMocks} from "./support/apiMocks";

const routeCases = [
    {path: "/nick", label: "Nick", total: /30\s*\/\s*1028/},
    {path: "/bakke", label: "Bakke", total: /24\s*\/\s*1173/},
] as const;

const viewportCases = [
    {name: "desktop", width: 1280, height: 900},
    {name: "mobile", width: 390, height: 844},
] as const;

const automaticDiagnosticCopy = /saved snapshot|backup feed|observed-capacity coverage|No internet connection right now|Live and prediction services are temporarily unavailable|Prediction services are temporarily unavailable|Current counts may be delayed/i;

test.beforeEach(async ({page}) => {
    await page.emulateMedia({reducedMotion: "reduce"});
});

for (const {level, expectedPct} of [
    {level: "LOW", expectedPct: 0.2}, {level: "MEDIUM", expectedPct: 0.5}, {level: "PEAK", expectedPct: 0.9},
]) {
    for (const mode of ["light", "dark"] as const) {
        test(`populated ${level} crowd caption passes contrast in ${mode}`, async ({page}) => {
            await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
            await installDashboardApiMocks(page, {forecastExpectedPct: expectedPct});
            await page.addInitScript((theme) => localStorage.setItem("reclive:themeMode", theme), mode);
            await page.goto("/nick");
            await expect(page.getByText(`${level} CROWD`, {exact: true})).toBeVisible();
            const results = await new AxeBuilder({page}).include("main").withRules(["color-contrast"]).analyze();
            expect(results.violations).toEqual([]);
            expect(results.incomplete.filter((rule) => rule.id === "color-contrast").flatMap((rule) => rule.nodes)
                .filter((node) => node.html.includes(`${level} CROWD`))).toEqual([]);
        });
    }
}

const seriousOrCritical = (violations: ReadonlyArray<{impact: string | null}>) =>
    violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""));

const waitForSettledDashboard = async (page: Page, total: RegExp) => {
    await expect(page.getByRole("heading", {name: total})).toBeVisible();
    await expect(page.getByRole("progressbar", {name: "Current occupancy percentage"})).toBeVisible();
    await expect(page.getByRole("button", {name: "Show map", exact: true})).toBeVisible();
    await expect(page.locator('main [role="progressbar"]')).toHaveCount(1);
};

const expectMinimumTarget = async (locator: Locator, minimum: number) => {
    const box = await locator.boundingBox();
    const accessibleName = await locator.getAttribute("aria-label") ?? await locator.textContent() ?? "unnamed target";
    expect(box).not.toBeNull();
    expect(box!.width, `${accessibleName} width`).toBeGreaterThanOrEqual(minimum);
    expect(box!.height, `${accessibleName} height`).toBeGreaterThanOrEqual(minimum);
};

const expectAllVisibleButtonsAtLeast44 = async (page: Page) => {
    const buttons = await page.locator("button").evaluateAll((elements) => elements
        .filter((element) => element.checkVisibility())
        .map((element) => {
            const box = element.getBoundingClientRect();
            return {
                name: element.getAttribute("aria-label") ?? element.textContent?.trim() ?? "unnamed button",
                width: box.width,
                height: box.height,
            };
        }));
    expect(buttons.length).toBeGreaterThan(0);
    for (const button of buttons) {
        expect(button.width, `${button.name} width`).toBeGreaterThanOrEqual(44);
        expect(button.height, `${button.name} height`).toBeGreaterThanOrEqual(44);
    }
};

const expectNoHorizontalOverflow = async (page: Page) => {
    expect(await page.evaluate(() => (
        document.documentElement.scrollWidth <= document.documentElement.clientWidth
    ))).toBe(true);
};

const expectQuietUnavailableOccupancy = async (page: Page) => {
    await expect(page.getByRole("heading", {name: "RecLive is unavailable."})).toHaveCount(1);
    await expect(page.getByRole("button", {name: "Try again", exact: true})).toBeVisible();
    await expect(page.getByLabel("Current count unavailable")).toHaveCount(0);
    await expect(page.getByRole("progressbar", {name: "Current occupancy percentage"})).toHaveCount(0);
    await expect(page.getByText("Live Occupancy", {exact: true})).toHaveCount(0);
    await expect(page.getByText("Forecast Today", {exact: true})).toHaveCount(0);
    await expect(page.getByText("Floor Heat Map", {exact: true})).toHaveCount(0);
    await expect(page.getByRole("button", {name: "Alerts", exact: true})).toHaveCount(0);
    await expect(page.getByText("Train smarter. Skip the crowd.", {exact: true})).toHaveCount(0);
    await expect(page.getByText("—", {exact: true})).toHaveCount(0);
    await expect(page.getByText(automaticDiagnosticCopy)).toHaveCount(0);
    await expect(page.getByRole("alert")).toHaveCount(0);
};

for (const routeCase of routeCases) {
    for (const viewport of viewportCases) {
        test(`${routeCase.label} ${viewport.name} renders without overlays or runtime errors`, async ({page}, testInfo) => {
            const consoleProblems: string[] = [];
            const pageErrors: string[] = [];
            page.on("console", (message) => {
                if (["warning", "error"].includes(message.type())) consoleProblems.push(message.text());
            });
            page.on("pageerror", (error) => pageErrors.push(error.message));

            await page.setViewportSize({width: viewport.width, height: viewport.height});
            await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
            await installDashboardApiMocks(page);
            await page.goto(routeCase.path);

            await expect(page).toHaveURL(new RegExp(`${routeCase.path}$`));
            await expect(page).toHaveTitle(new RegExp(routeCase.label, "i"));
            await expect(page.locator("main")).toBeVisible();
            await expect(page.locator("main")).not.toBeEmpty();
            await waitForSettledDashboard(page, routeCase.total);
            await expect(page.locator("vite-error-overlay")).toHaveCount(0);
            await expect(page.getByText(/Internal Server Error|Failed to fetch dynamically imported module/)).toHaveCount(0);

            await page.screenshot({
                path: testInfo.outputPath(`${routeCase.label.toLowerCase()}-${viewport.name}-first-viewport.png`),
                fullPage: false,
            });

            const alertsButton = page.getByRole("button", {name: "Alerts", exact: true});
            await expectMinimumTarget(alertsButton, 44);
            await alertsButton.click();
            const alertsDialog = page.getByRole("dialog", {name: "Alerts", exact: true});
            await expect(alertsDialog).toBeVisible();
            await expect(alertsDialog.getByRole("region", {name: "Manage alerts"})).toBeVisible();
            await expectMinimumTarget(alertsDialog.getByRole("button", {name: "Close alerts"}), 44);
            await page.screenshot({
                path: testInfo.outputPath(`${routeCase.label.toLowerCase()}-${viewport.name}-alerts-open.png`),
                fullPage: true,
            });
            await alertsDialog.getByRole("button", {name: "Close alerts"}).click();
            await expect(alertsDialog).toBeHidden();

            await expectAllVisibleButtonsAtLeast44(page);
            await page.screenshot({
                path: testInfo.outputPath(`${routeCase.label.toLowerCase()}-${viewport.name}.png`),
                fullPage: true,
            });

            expect(pageErrors).toEqual([]);
            expect(consoleProblems).toEqual([]);
        });
    }
}

for (const stateCase of [
    {name: "partial", liveCountsMode: "partial" as const, viewport: viewportCases[1]},
    {name: "missing", liveCountsMode: "missing" as const, viewport: viewportCases[0]},
]) {
    test(`Nick ${stateCase.name} facility becomes one quiet unavailable view at ${stateCase.viewport.name} size`, async ({page}, testInfo) => {
        const consoleProblems: string[] = [];
        const pageErrors: string[] = [];
        page.on("console", (message) => {
            if (["warning", "error"].includes(message.type())) consoleProblems.push(message.text());
        });
        page.on("pageerror", (error) => pageErrors.push(error.message));

        await page.setViewportSize({width: stateCase.viewport.width, height: stateCase.viewport.height});
        await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
        await installDashboardApiMocks(page, {liveCountsMode: stateCase.liveCountsMode});
        await page.goto("/nick");

        await expect(page).toHaveURL(/\/nick$/);
        await expect(page).toHaveTitle(/Nick/i);
        await expect(page.locator("main")).toBeVisible();
        await expect(page.locator("main")).not.toBeEmpty();
        await expect(page.getByRole("button", {name: "Nick", exact: true})).toHaveAttribute("aria-pressed", "true");
        await expectQuietUnavailableOccupancy(page);
        await expect(page.locator("vite-error-overlay")).toHaveCount(0);
        await expect(page.getByText(/Internal Server Error|Failed to fetch dynamically imported module/)).toHaveCount(0);
        await expectNoHorizontalOverflow(page);
        const accessibilityResults = await new AxeBuilder({page}).include("main").analyze();
        expect(seriousOrCritical(accessibilityResults.violations)).toEqual([]);
        await page.screenshot({
            path: testInfo.outputPath(`nick-${stateCase.name}-${stateCase.viewport.name}.png`),
            fullPage: false,
        });

        expect(pageErrors).toEqual([]);
        expect(consoleProblems).toEqual([]);
    });
}

for (const routeCase of routeCases) {
    for (const viewport of viewportCases) {
        test(`${routeCase.label} ${viewport.name} dashboard, alert form, and heat-map dialog pass axe`, async ({page}) => {
            await page.setViewportSize({width: viewport.width, height: viewport.height});
            await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
            await installDashboardApiMocks(page);
            await page.goto(routeCase.path);
            await waitForSettledDashboard(page, routeCase.total);

            const dashboardResults = await new AxeBuilder({page}).include("main").analyze();
            expect(seriousOrCritical(dashboardResults.violations)).toEqual([]);

            await page.getByRole("button", {name: "Alerts", exact: true}).click();
            const alertsDialog = page.getByRole("dialog", {name: "Alerts", exact: true});
            await expect(alertsDialog.getByRole("region", {name: "Manage alerts"})).toBeVisible();
            const alertsResults = await new AxeBuilder({page})
                .include('[role="dialog"][aria-labelledby="alerts-panel-title"]')
                .analyze();
            expect(seriousOrCritical(alertsResults.violations)).toEqual([]);
            await alertsDialog.getByRole("button", {name: "Close alerts"}).click();
            await expect(alertsDialog).toBeHidden();

            await page.getByRole("button", {name: "Show map", exact: true}).click();
            const zone = page.locator('main polygon[role="button"]').first();
            await zone.focus();
            await page.keyboard.press("Enter");
            const heatmapDialog = page.locator("#heatmap-zone-dialog");
            await expect(heatmapDialog).toBeVisible();
            await expect(heatmapDialog).toHaveAttribute("aria-labelledby", /.+/);
            const heatmapResults = await new AxeBuilder({page}).include("#heatmap-zone-dialog").analyze();
            expect(seriousOrCritical(heatmapResults.violations)).toEqual([]);
        });
    }
}

for (const routeCase of routeCases) {
    test(`${routeCase.label} exposes every configured heat-map floor with usable zone targets`, async ({page}, testInfo) => {
        await page.setViewportSize({width: 390, height: 844});
        await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
        await installDashboardApiMocks(page);
        await page.goto(routeCase.path);
        await waitForSettledDashboard(page, routeCase.total);
        await page.getByRole("button", {name: "Show map", exact: true}).click();

        const expectedFloors = routeCase.path === "/nick"
            ? ["Lower", "Floor 1", "Floor 2", "Floor 3", "Floor 4"]
            : ["Floor 1", "Floor 2", "Floor 3", "Floor 4"];

        for (const floor of expectedFloors) {
            await page.getByRole("button", {name: floor, exact: true}).click();
            const zones = page.locator('main polygon[role="button"]');
            await expect(zones.first()).toBeVisible();
            const count = await zones.count();
            expect(count).toBeGreaterThan(0);
            for (let index = 0; index < count; index += 1) {
                await expectMinimumTarget(zones.nth(index), 24);
            }
        }

        const focusedZone = routeCase.path === "/nick"
            ? page.getByRole("button", {name: /^Racquetball:/})
            : page.locator('main polygon[role="button"]').first();
        await focusedZone.focus();
        await page.screenshot({
            path: testInfo.outputPath(`${routeCase.label.toLowerCase()}-mobile-heatmap-floor4-focus.png`),
            fullPage: true,
        });
    });
}

test("heat-map dialog supports Enter, Space, Escape, close, and click-away with focus restoration", async ({page}, testInfo) => {
    await page.setViewportSize({width: 1280, height: 900});
    await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
    await installDashboardApiMocks(page);
    await page.goto("/nick");
    await waitForSettledDashboard(page, routeCases[0].total);
    await page.getByRole("button", {name: "Show map", exact: true}).click();
    const zone = page.locator('main polygon[aria-label^="Power House:"]');

    await zone.focus();
    await page.keyboard.press("Enter");
    expect(await zone.evaluate((element) => ({
        stroke: getComputedStyle(element).stroke,
        strokeWidth: getComputedStyle(element).strokeWidth,
        strokeOpacity: getComputedStyle(element).strokeOpacity,
        vectorEffect: getComputedStyle(element).vectorEffect,
    }))).toEqual({
        stroke: "rgb(0, 0, 0)",
        strokeWidth: "3px",
        strokeOpacity: "1",
        vectorEffect: "non-scaling-stroke",
    });
    await expect(page.getByRole("dialog", {name: "Power House details"})).toBeVisible();
    await page.screenshot({
        path: testInfo.outputPath("nick-desktop-heatmap-dialog.png"),
        fullPage: true,
    });
    await page.keyboard.press("Escape");
    await expect(page.getByRole("dialog", {name: "Power House details"})).toBeHidden();
    await expect(zone).toBeFocused();

    await page.keyboard.press("Space");
    const close = page.getByRole("button", {name: "Close Power House details"});
    await expectMinimumTarget(close, 44);
    await close.click();
    await expect(zone).toBeFocused();

    await page.keyboard.press("Enter");
    await expect(page.getByRole("dialog", {name: "Power House details"})).toBeVisible();
    await page.getByText("Floor Heat Map", {exact: true}).click();
    await expect(page.getByRole("dialog", {name: "Power House details"})).toBeHidden();
    await expect(zone).toBeFocused();

    await page.getByRole("button", {name: "Switch to dark theme"}).click();
    await zone.focus();
    await page.keyboard.press("ArrowRight");
    expect(await zone.evaluate((element) => getComputedStyle(element).stroke)).toBe("rgb(255, 255, 255)");
    await page.screenshot({
        path: testInfo.outputPath("nick-desktop-heatmap-focus-dark.png"),
        fullPage: true,
    });
    const darkMainResults = await new AxeBuilder({page}).include("main").analyze();
    expect(seriousOrCritical(darkMainResults.violations)).toEqual([]);
});

test("generated worker preserves a quiet cached shell, expires counts, and recovers online", async ({page, context}) => {
    await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
    await installDashboardApiMocks(page);
    await page.goto("/nick");
    await waitForSettledDashboard(page, routeCases[0].total);

    await page.evaluate(async () => navigator.serviceWorker.ready);
    if (!await page.evaluate(() => Boolean(navigator.serviceWorker.controller))) {
        await page.reload();
        await waitForSettledDashboard(page, routeCases[0].total);
    }
    expect(await page.evaluate(() => Boolean(navigator.serviceWorker.controller))).toBe(true);

    const cachedApiUrls = await page.evaluate(async () => {
        const urls: string[] = [];
        for (const cacheName of await caches.keys()) {
            const cache = await caches.open(cacheName);
            for (const request of await cache.keys()) {
                if (new URL(request.url).pathname.startsWith("/api/")) urls.push(request.url);
            }
        }
        return urls;
    });
    expect(cachedApiUrls).toEqual([]);

    await removeDashboardApiMocks(page);
    await context.setOffline(true);
    await page.reload();
    await expect(page).toHaveURL(/\/nick$/);
    await expect(page).toHaveTitle(/Nick/i);
    await expect(page.locator("main")).toBeVisible();
    await expect(page.locator("main")).not.toBeEmpty();
    await expect(page.getByRole("button", {name: "Nick", exact: true})).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("heading", {name: routeCases[0].total})).toBeVisible();
    await expect(page.getByText(automaticDiagnosticCopy)).toHaveCount(0);

    await page.clock.setFixedTime(new Date("2026-08-31T12:11:00Z"));
    await page.reload();
    await expect(page.locator("main")).toBeVisible();
    await expect(page.getByRole("heading", {name: routeCases[0].total})).toHaveCount(0);
    await expectQuietUnavailableOccupancy(page);

    await page.evaluate(() => window.localStorage.removeItem("reclive:facilityCache"));
    await page.goto("/bakke");
    await expect(page).toHaveURL(/\/bakke$/);
    await expect(page).toHaveTitle(/Bakke/i);
    await expect(page.locator("main")).toBeVisible();
    await expect(page.locator("main")).not.toBeEmpty();
    await expect(page.getByRole("button", {name: "Bakke", exact: true})).toHaveAttribute("aria-pressed", "true");
    await expectQuietUnavailableOccupancy(page);
    await expect(page.getByRole("button", {name: "Try again", exact: true})).toBeVisible();
    await expectNoHorizontalOverflow(page);

    await installDashboardApiMocks(page, {observedAt: "2026-08-31T12:11:00Z"});
    await context.setOffline(false);
    await page.reload();
    await waitForSettledDashboard(page, routeCases[1].total);
    await expect(page.getByRole("button", {name: "Bakke", exact: true})).toHaveAttribute("aria-pressed", "true");
});

test("normal production localhost never exposes debug globals", async ({page}) => {
    await page.addInitScript(() => {
        window.localStorage.setItem("reclive:closureOverride", "true");
        window.localStorage.setItem("reclive:debugNow", "2031-01-01T12:00:00Z");
    });
    await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
    await installDashboardApiMocks(page);
    await page.goto("/nick?debugNow=2030-01-01T12:00:00Z&overrideClosure=true");
    await waitForSettledDashboard(page, routeCases[0].total);
    await expect(page.getByText("Updated today at 7:00 AM", {exact: true})).toBeVisible();

    expect(await page.evaluate(() => ({
        dashboard: typeof window.recliveDebugDashboardState,
        prediction: typeof window.recliveShowPredictions,
        closure: typeof window.recliveOverrideClosure,
        clock: typeof window.recliveSetDebugNow,
        heatmap: typeof window.heatmapdebug,
    }))).toEqual({
        dashboard: "undefined",
        prediction: "undefined",
        closure: "undefined",
        clock: "undefined",
        heatmap: "undefined",
    });
});
