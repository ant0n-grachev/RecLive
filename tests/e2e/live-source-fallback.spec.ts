import {join} from "node:path";
import {expect, test, type Page} from "@playwright/test";
import {installDashboardApiMocks} from "./support/apiMocks";

const fixedNow = "2026-08-31T12:00:00Z";
const screenshotDir = process.env.RECLIVE_E2E_SCREENSHOT_DIR;

const expectHealthyDashboard = async (page: Page) => {
    await expect(page.getByText("Live Occupancy", {exact: true})).toBeVisible();
    await expect(page.getByRole("progressbar", {name: "Current occupancy percentage"})).toBeVisible();
    await expect(page.getByRole("heading", {name: "RecLive is unavailable."})).toHaveCount(0);
};

const expectUnavailableDashboard = async (page: Page) => {
    const main = page.getByRole("main");
    await expect(main.getByRole("heading", {name: "RecLive is unavailable."})).toHaveCount(1);
    await expect(main.getByRole("button", {name: "Try again"})).toBeVisible();
    await expect(main.getByRole("button", {name: "Nick", exact: true})).toBeVisible();
    await expect(main.getByRole("button", {name: "Bakke", exact: true})).toBeVisible();

    await expect(main.getByText("Live Occupancy", {exact: true})).toHaveCount(0);
    await expect(main.getByText("Schedule Status", {exact: true})).toHaveCount(0);
    await expect(main.getByText("Forecast Today", {exact: true})).toHaveCount(0);
    await expect(main.getByText("Floor Heat Map", {exact: true})).toHaveCount(0);
    await expect(main.getByRole("button", {name: "Alerts", exact: true})).toHaveCount(0);
    await expect(main.getByRole("button", {name: "How to add RecLive to your home screen"})).toHaveCount(0);
    await expect(main.getByText("Train smarter. Skip the crowd.", {exact: true})).toHaveCount(0);
    await expect(main.getByText("—", {exact: true})).toHaveCount(0);
    await expect(page.getByRole("dialog")).toHaveCount(0);
};

test.beforeEach(async ({page}) => {
    await page.emulateMedia({reducedMotion: "reduce"});
    await page.clock.setFixedTime(new Date(fixedNow));
});

for (const facility of [
    {path: "/nick", label: "Nick"},
    {path: "/bakke", label: "Bakke"},
] as const) {
    test(`${facility.label} uses the official public source without touching the failed backup`, async ({page}) => {
        if (facility.path === "/nick") await page.setViewportSize({width: 390, height: 844});
        const mocks = await installDashboardApiMocks(page, {
            officialCountsMode: "fresh",
            liveCountsMode: "failed",
        });

        await page.goto(facility.path);
        await expectHealthyDashboard(page);
        expect(mocks.officialRequests).toHaveLength(1);
        expect(mocks.backupRequests).toHaveLength(0);
        expect(mocks.liveRequestOrder).toEqual(["official"]);

        if (facility.path === "/nick") {
            const fetchedAt = await page.evaluate(() => {
                const raw = localStorage.getItem("reclive:facilityCache");
                const cache = raw ? JSON.parse(raw) as Record<string, {
                    payload: {locations: Array<{locationId: number; fetchedAt: string | null}>};
                }> : {};
                return cache["1186"]?.payload.locations.find((row) => row.locationId === 5761)?.fetchedAt;
            });
            expect(fetchedAt).toBe(new Date(fixedNow).toISOString());
            if (screenshotDir) {
                await page.screenshot({path: join(screenshotDir, "reclive-healthy-phone.png"), fullPage: true});
            }
        }
    });
}

test("falls back to a fresh server snapshot only after the official source fails", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "failed",
        liveCountsMode: "fresh",
    });

    await page.goto("/nick");
    await expectHealthyDashboard(page);
    expect(mocks.liveRequestOrder).toEqual(["official", "backup"]);
});

test("shows one quiet unavailable view when both live sources fail and preserves facility switching", async ({page}) => {
    await page.setViewportSize({width: 1280, height: 900});
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "failed",
        liveCountsMode: "failed",
    });

    await page.goto("/nick");
    await expectUnavailableDashboard(page);
    expect(mocks.liveRequestOrder).toEqual(["official", "backup"]);
    if (screenshotDir) {
        await page.screenshot({path: join(screenshotDir, "reclive-outage-desktop.png"), fullPage: true});
    }

    const switchRequestStart = mocks.liveRequestOrder.length;
    await page.getByRole("button", {name: "Bakke", exact: true}).click();
    await expect(page).toHaveURL(/\/bakke$/);
    await expect(page.getByRole("button", {name: "Bakke", exact: true})).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByRole("button", {name: "Nick", exact: true})).toHaveAttribute("aria-pressed", "false");
    await expectUnavailableDashboard(page);
    await expect(page.getByRole("button", {name: "Try again", exact: true})).toBeEnabled();
    await expect.poll(() => mocks.liveRequestOrder.slice(switchRequestStart).slice(-2))
        .toEqual(["official", "backup"]);

    const switchRequestOrder = mocks.liveRequestOrder.slice(switchRequestStart);
    // The keyed route remount may cancel one intermediate request after its official attempt.
    const cancelledOrCompletedIntermediateAttempt = switchRequestOrder.slice(0, -2);
    expect([[], ["official"], ["official", "backup"]])
        .toContainEqual(cancelledOrCompletedIntermediateAttempt);
});

test("rejects a stale backup snapshot when the official source is unavailable", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "failed",
        liveCountsMode: "stale",
    });

    await page.goto("/nick");
    await expectUnavailableDashboard(page);
    expect(mocks.liveRequestOrder).toEqual(["official", "backup"]);
});

test("manual Try again recovers the unavailable view without a page reload", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "failed",
        liveCountsMode: "failed",
    });
    await page.goto("/nick");
    await expectUnavailableDashboard(page);

    mocks.setOfficialMode("fresh");
    await page.getByRole("button", {name: "Try again"}).click();
    await expectHealthyDashboard(page);
    expect(mocks.liveRequestOrder).toEqual(["official", "backup", "official"]);
});

test("returning to a visible tab automatically recovers the unavailable view", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "failed",
        liveCountsMode: "failed",
    });
    await page.goto("/nick");
    await expectUnavailableDashboard(page);

    mocks.setOfficialMode("fresh");
    await page.evaluate(() => {
        Object.defineProperty(document, "visibilityState", {configurable: true, value: "hidden"});
        document.dispatchEvent(new Event("visibilitychange"));
        Object.defineProperty(document, "visibilityState", {configurable: true, value: "visible"});
        document.dispatchEvent(new Event("visibilitychange"));
    });

    await expectHealthyDashboard(page);
    expect(mocks.liveRequestOrder).toEqual(["official", "backup", "official"]);
});

test("keeps a recent cached snapshot during a transient dual-source outage", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "fresh",
        liveCountsMode: "failed",
    });
    await page.goto("/nick");
    await expectHealthyDashboard(page);

    mocks.setOfficialMode("failed");
    await page.reload();
    await expectHealthyDashboard(page);
    expect(mocks.liveRequestOrder).toEqual(["official", "official", "backup"]);
});

test("replaces a cached snapshot older than ten minutes with the unavailable view", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "fresh",
        liveCountsMode: "failed",
    });
    await page.goto("/nick");
    await expectHealthyDashboard(page);

    mocks.setOfficialMode("failed");
    await page.clock.setFixedTime(new Date("2026-08-31T12:10:01Z"));
    await page.reload();
    await expectUnavailableDashboard(page);
});

test("keeps the facility live at sufficient coverage and hides the missing optional section", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "coverage",
        liveCountsMode: "failed",
    });

    await page.goto("/nick");
    await expectHealthyDashboard(page);
    await expect(page.getByText("Fitness Floors", {exact: true})).toBeVisible();
    await expect(page.getByText("Racquetball Courts", {exact: true})).toHaveCount(0);
    await expect(page.getByText("—", {exact: true})).toHaveCount(0);
    expect(mocks.backupRequests).toHaveLength(0);
});

test("retains a confirmed official closure without consulting the backup", async ({page}) => {
    const mocks = await installDashboardApiMocks(page, {
        officialCountsMode: "closed",
        liveCountsMode: "failed",
    });

    await page.goto("/nick");
    await expect(page.getByText("CLOSED", {exact: true}).first()).toBeVisible();
    await expect(page.getByRole("heading", {name: "RecLive is unavailable."})).toHaveCount(0);
    expect(mocks.liveRequestOrder).toEqual(["official"]);
});
