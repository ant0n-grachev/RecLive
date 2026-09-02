import {HttpResponse, http} from "msw";
import {describe, expect, it} from "vitest";
import {server} from "../../test/msw/server";
import {fetchFacilityHours} from "./facilityScheduleParser";

const scheduleEndpoint = "*/api/facility-hours/facilities/1186";

const validSchedule = {
    generatedAt: "2026-09-01T12:00:00Z",
    sourceSite: "https://recwell.wisc.edu",
    facilityId: 1186,
    facilityName: "Nicholas Recreation Center",
    slug: "nick",
    url: "https://recwell.wisc.edu/nick/",
    resolvedUrl: "https://recwell.wisc.edu/nick/",
    status: "stale",
    source: "wp_json",
    sourceModifiedGmt: "2026-09-01T11:45:00",
    sections: [{
        title: "Building Hours",
        rows: [{label: "Mon-Fri", hours: "6:00 am - 10:00 pm"}],
        note: "Holiday hours may vary.",
    }],
    sourceFetchedAt: "2026-09-01T12:00:00Z",
    lastSuccessfulAt: "2026-09-01T11:55:00Z",
    stale: true,
    error: "Upstream refresh failed",
    errorCategory: "upstream_timeout",
    updatedAt: "2026-09-01T12:00:00Z",
} as const;

describe("fetchFacilityHours", () => {
    it("preserves validated schedule provenance and Phase 7 metadata", async () => {
        server.use(http.get(scheduleEndpoint, () => HttpResponse.json(validSchedule)));

        await expect(fetchFacilityHours(1186)).resolves.toEqual(validSchedule);
    });

    it("rejects a schema-invalid schedule instead of coercing it", async () => {
        server.use(http.get(scheduleEndpoint, () => HttpResponse.json({
            ...validSchedule,
            facilityId: "1186",
        })));

        await expect(fetchFacilityHours(1186)).rejects.toMatchObject({kind: "schema"});
    });

    it("rejects a valid schedule bound to a different facility", async () => {
        server.use(http.get(scheduleEndpoint, () => HttpResponse.json({
            ...validSchedule,
            facilityId: 1656,
            facilityName: "Bakke Recreation & Wellbeing Center",
            slug: "bakke",
            url: "https://recwell.wisc.edu/bakke/",
            resolvedUrl: "https://recwell.wisc.edu/bakke/",
        })));

        await expect(fetchFacilityHours(1186)).rejects.toMatchObject({kind: "schema"});
    });
});
