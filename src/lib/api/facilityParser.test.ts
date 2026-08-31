import {http, HttpResponse, type JsonBodyType} from "msw";
import {env} from "../config/env";
import {server} from "../../test/msw/server";
import {fetchFacility} from "./facilityParser";

const row = (overrides: Record<string, unknown> = {}) => ({
    LocationId: 5761,
    IsClosed: false,
    LastCount: 47,
    LastUpdatedDateAndTime: null,
    FetchedAt: "2026-08-31T12:00:00Z",
    ...overrides,
});

const mockLiveCounts = (payload: JsonBodyType) => {
    const requestUrl = new URL(env.liveCountsUrl, window.location.origin);
    server.use(http.get(`${requestUrl.origin}${requestUrl.pathname}`, () => HttpResponse.json(payload)));
};

describe("fetchFacility live-count parsing", () => {
    it("prefers canonical rows and maps their FetchedAt provenance", async () => {
        mockLiveCounts({
            rows: [row()],
            data: [row({LastCount: 99, FetchedAt: "2026-08-31T11:00:00Z"})],
        });

        await expect(fetchFacility(1186)).resolves.toMatchObject({
            locations: expect.arrayContaining([
                expect.objectContaining({
                    locationId: 5761,
                    currentCapacity: 47,
                    fetchedAt: "2026-08-31T12:00:00Z",
                }),
            ]),
        });
    });

    it.each([
        ["array", [row()]],
        ["data envelope", {data: [row()]}],
    ])("keeps %s payload freshness unknown", async (_label, payload) => {
        mockLiveCounts(payload);

        await expect(fetchFacility(1186)).resolves.toMatchObject({
            locations: expect.arrayContaining([
                expect.objectContaining({
                    locationId: 5761,
                    currentCapacity: 47,
                    fetchedAt: null,
                }),
            ]),
        });
    });
});
