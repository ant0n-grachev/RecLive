import type {BrowserContext, Page, Route} from "@playwright/test";

const observedAt = "2026-08-31T12:00:00Z";
const fixtureDate = "2026-08-31";
const vapidPublicKey =
    "BGsX0fLhLEJH-Lzm5WOkQPJ3A32BLeszoPShOUXYmMKWT-NC4v4af5uO5-tKfA-eFivOM1drMV7Oy7ZAaDe_UfU";

const nickLocationIds = [5761, 5764, 5760, 7089, 5762, 5758, 7090, 5766, 5753, 5754, 5763];
const bakkeLocationIds = [8718, 8717, 8720, 8698, 8716, 10550, 8705, 8712, 8700, 8714, 8701, 8699, 8696, 8694, 8695];
const partialFreshLocationIds = new Set([5761, 5764, 5760, 5762, 8717, 8700, 10550, 8716]);
type LiveCountsMode = "fresh" | "partial" | "missing";

const liveRow = (locationId: number, mode: LiveCountsMode, liveObservedAt: string) => ({
    LocationId: locationId,
    IsClosed: false,
    LastCount: mode === "missing" ? null : locationId === 5761 ? 30 : locationId === 8717 ? 24 : 0,
    LastUpdatedDateAndTime: mode === "missing" ? null : liveObservedAt,
    FetchedAt: mode === "fresh" || (mode === "partial" && partialFreshLocationIds.has(locationId))
        ? liveObservedAt
        : null,
});

const liveCounts = (mode: LiveCountsMode, liveObservedAt: string) => ({
    ingestion: {
        lastSuccessfulFetchAt: liveObservedAt,
        ageSeconds: 0,
        status: "healthy",
    },
    rows: [...nickLocationIds, ...bakkeLocationIds].map(
        (locationId) => liveRow(locationId, mode, liveObservedAt)
    ),
});

const facilityName = (id: 1186 | 1656) => id === 1656
    ? "Bakke Recreation & Wellbeing Center"
    : "Nicholas Recreation Center";

const forecast = (id: 1186 | 1656, expectedPct?: number) => ({
    facilityId: id,
    facilityName: facilityName(id),
    forecastDayStartHour: 6,
    forecastDayEndHour: 23,
    occupancyThresholds: {lowMax: 34, peakMin: 70},
    sectionOccupancyThresholds: {},
    locationOccupancyThresholds: {},
    weeklyForecast: [{
        dayName: "Monday",
        date: fixtureDate,
        categories: [],
        totalHours: [{
            hour: 9,
            hourStart: "2026-08-31T09:00:00-05:00",
            expectedCount: 50,
            ...(expectedPct === undefined ? {} : {expectedPct}),
            spikeAdjusted: true,
        }],
        avoidWindows: [],
        bestWindows: [],
        crowdBands: [],
    }],
});

const actualHours = (id: 1186 | 1656) => ({
    facilityId: id,
    date: fixtureDate,
    categories: [],
    totalHours: [],
});

const schedule = (id: 1186 | 1656) => ({
    generatedAt: observedAt,
    sourceSite: "https://recwell.example.test",
    facilityId: id,
    facilityName: facilityName(id),
    slug: id === 1656 ? "bakke" : "nick",
    url: `https://recwell.example.test/${id === 1656 ? "bakke" : "nick"}/`,
    resolvedUrl: `https://recwell.example.test/${id === 1656 ? "bakke" : "nick"}/`,
    status: "ok",
    source: "direct_html",
    sections: [{
        title: "Building Hours",
        rows: [{label: "Monday", hours: "6:00 am - 10:00 pm"}],
        note: null,
    }],
    sourceFetchedAt: observedAt,
    lastSuccessfulAt: observedAt,
    stale: false,
    error: null,
    errorCategory: null,
    updatedAt: observedAt,
});

type RouteHandler = (route: Route) => Promise<void>;

const apiHandlers = new WeakMap<BrowserContext, RouteHandler>();
const externalHandlers = new WeakMap<BrowserContext, RouteHandler>();

const fixtureNotFound = (route: Route) => route.fulfill({
    status: 404,
    json: {detail: "fixture route not found"},
});

const facilityIdFromPath = (pathname: string): 1186 | 1656 | null => {
    const match = pathname.match(/\/facilities\/(1186|1656)(?:\/actual-hours)?$/);
    return match?.[1] === "1186" ? 1186 : match?.[1] === "1656" ? 1656 : null;
};

export async function installDashboardApiMocks(
    page: Page,
    options: {forecastExpectedPct?: number; liveCountsMode?: LiveCountsMode; observedAt?: string} = {}
): Promise<void> {
    const context = page.context();
    const previousApiHandler = apiHandlers.get(context);
    if (previousApiHandler) await context.unroute("**/api/**", previousApiHandler);

    if (!externalHandlers.has(context)) {
        const externalHandler: RouteHandler = async (route) => {
            const url = new URL(route.request().url());
            if (
                (url.protocol === "http:" || url.protocol === "https:")
                && url.hostname !== "127.0.0.1"
                && url.hostname !== "localhost"
            ) {
                await route.abort("blockedbyclient");
                return;
            }
            await route.continue();
        };
        externalHandlers.set(context, externalHandler);
        await context.route("**/*", externalHandler);
    }

    const apiHandler: RouteHandler = async (route) => {
        const request = route.request();
        const pathname = new URL(request.url()).pathname;
        const method = request.method();

        if (method === "GET" && pathname === "/api/live-counts") {
            await route.fulfill({
                json: liveCounts(options.liveCountsMode ?? "fresh", options.observedAt ?? observedAt),
            });
            return;
        }

        const actualHoursMatch = pathname.match(
            /^\/api\/forecast\/facilities\/(1186|1656)\/actual-hours$/
        );
        const actualHoursFacilityId = actualHoursMatch?.[1] === "1186"
            ? 1186
            : actualHoursMatch?.[1] === "1656" ? 1656 : null;
        if (method === "GET" && actualHoursFacilityId) {
            await route.fulfill({json: actualHours(actualHoursFacilityId)});
            return;
        }

        const facilityId = facilityIdFromPath(pathname);
        if (method === "GET" && facilityId && pathname.startsWith("/api/forecast/facilities/")) {
            await route.fulfill({json: forecast(facilityId, options.forecastExpectedPct)});
            return;
        }
        if (method === "GET" && facilityId && pathname.startsWith("/api/facility-hours/facilities/")) {
            await route.fulfill({json: schedule(facilityId)});
            return;
        }

        if (method === "GET" && pathname === "/api/push/availability") {
            await route.fulfill({json: {
                apiAvailable: true,
                dbAvailable: true,
                alertsAvailable: true,
                reason: null,
                storeBackend: "db",
            }});
            return;
        }
        if (method === "GET" && pathname === "/api/push/public-key") {
            await route.fulfill({json: {publicKey: vapidPublicKey}});
            return;
        }
        if (method === "POST" && pathname === "/api/push/rules/list") {
            await route.fulfill({json: {status: "ok", rules: []}});
            return;
        }
        if (method === "POST" && pathname === "/api/push/subscribe") {
            const body = request.postDataJSON() as {
                facilityId: 1186 | 1656;
                sectionKey: string;
                threshold: number;
            };
            await route.fulfill({json: {
                status: "ok",
                created: true,
                rule: {
                    id: 7,
                    facilityId: body.facilityId,
                    sectionKey: body.sectionKey,
                    threshold: body.threshold,
                    createdAt: observedAt,
                    expiresAt: "2026-09-01T12:00:00Z",
                    status: "pending",
                },
            }});
            return;
        }
        if (method === "POST" && pathname === "/api/push/rules/cancel-all") {
            await route.fulfill({json: {status: "ok", cancelled: 0}});
            return;
        }
        if (method === "DELETE" && /^\/api\/push\/rules\/[1-9]\d*$/.test(pathname)) {
            await route.fulfill({json: {status: "ok", cancelled: 1}});
            return;
        }

        await fixtureNotFound(route);
    };

    apiHandlers.set(context, apiHandler);
    await context.route("**/api/**", apiHandler);
}

export async function removeDashboardApiMocks(page: Page): Promise<void> {
    const context = page.context();
    const handler = apiHandlers.get(context);
    if (!handler) return;
    await context.unroute("**/api/**", handler);
    apiHandlers.delete(context);
}
