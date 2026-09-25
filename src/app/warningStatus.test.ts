import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import {resolveDashboardWarning, type WarningResolverInput} from "./warningStatus";

const input = (occupancyStatus: OccupancySummary["status"]): WarningResolverInput => ({
    hasAnyError: false,
    isOffline: false,
    liveOutageState: "none",
    liveDataSource: "facility_api",
    forecastError: null,
    isScheduledClosedNow: false,
    isScheduledOpenButDataNotLive: false,
    occupancyStatus,
});

describe("resolveDashboardWarning occupancy coverage", () => {
    it("labels partial observations without hiding a valid forecast", () => {
        expect(resolveDashboardWarning(input("partial"))).toMatchObject({
            kind: "partial_live",
            hidePredictions: false,
        });
    });

    it.each(["insufficient", "unknown"] as const)(
        "reports %s live occupancy as unavailable without suppressing a valid forecast",
        (status) => {
            expect(resolveDashboardWarning(input(status))).toMatchObject({
                kind: "occupancy_unavailable",
                hidePredictions: false,
            });
        }
    );

    it("does not warn for a complete live summary", () => {
        expect(resolveDashboardWarning(input("live"))).toEqual({
            kind: "none",
            text: null,
            hidePredictions: false,
        });
    });

    const priorityCases = [
        {
            label: "offline cached snapshot",
            overrides: {
                isOffline: true,
                liveOutageState: "cache",
                forecastError: "lower-priority forecast outage",
            },
            kind: "offline_cache",
            hidePredictions: true,
            contradictoryCopy: /live occupancy is paused/i,
        },
        {
            label: "connected total-outage cached snapshot",
            overrides: {
                liveOutageState: "cache",
                forecastError: "lower-priority forecast outage",
            },
            kind: "total_outage_cache",
            hidePredictions: true,
            contradictoryCopy: /occupancy is not live/i,
        },
        {
            label: "forecast outage",
            overrides: {forecastError: "forecast unavailable"},
            kind: "prediction_unavailable",
            hidePredictions: true,
            contradictoryCopy: /Live occupancy is still shown/i,
        },
        {
            label: "fallback feed",
            overrides: {
                liveDataSource: "fallback_api",
                forecastError: "lower-priority forecast outage",
            },
            kind: "facility_fallback",
            hidePredictions: true,
            contradictoryCopy: /occupancy may be slightly delayed/i,
        },
        {
            label: "scheduled-open data gap",
            overrides: {
                isScheduledOpenButDataNotLive: true,
                forecastError: "lower-priority forecast outage",
            },
            kind: "scheduled_open_not_live",
            hidePredictions: false,
            contradictoryCopy: /Current counts may be delayed/i,
        },
    ] as const;

    describe.each(priorityCases)("$label priority copy", ({overrides, kind, hidePredictions, contradictoryCopy}) => {
        it.each(["insufficient", "unknown"] as const)(
            "states explicitly that %s occupancy is unavailable without changing priority",
            (status) => {
                const result = resolveDashboardWarning({...input(status), ...overrides});

                expect(result).toMatchObject({kind, hidePredictions});
                expect(result.text).toContain("Live occupancy unavailable");
                expect(result.text).not.toMatch(contradictoryCopy);
            }
        );

        it.each(["live", "partial"] as const)(
            "does not call observed %s occupancy unavailable",
            (status) => {
                const result = resolveDashboardWarning({...input(status), ...overrides});

                expect(result).toMatchObject({kind, hidePredictions});
                expect(result.text).not.toContain("Live occupancy unavailable");
            }
        );
    });

    it.each(priorityCases.slice(2))(
        "$label copy truthfully describes a closed occupancy summary",
        ({overrides, kind, hidePredictions, contradictoryCopy}) => {
            const result = resolveDashboardWarning({...input("closed"), ...overrides});

            expect(result).toMatchObject({kind, hidePredictions});
            expect(result.text).toMatch(/\bclosed\b/i);
            expect(result.text).not.toMatch(contradictoryCopy);
        }
    );

    it("does not claim forecasts are available for a closed scheduled-open gap with a forecast error", () => {
        const result = resolveDashboardWarning({
            ...input("closed"),
            isScheduledOpenButDataNotLive: true,
            forecastError: "forecast unavailable",
        });

        expect(result).toMatchObject({
            kind: "scheduled_open_not_live",
            hidePredictions: false,
        });
        expect(result.text).toMatch(/\bclosed\b/i);
        expect(result.text).not.toMatch(/forecasts remain available/i);
    });

    it("keeps an expected closure silent ahead of occupancy-unavailable copy", () => {
        expect(resolveDashboardWarning({
            ...input("closed"),
            isScheduledClosedNow: true,
            forecastError: "lower-priority forecast outage",
        })).toEqual({
            kind: "scheduled_closed",
            text: null,
            hidePredictions: true,
        });
    });
});
