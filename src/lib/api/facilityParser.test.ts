import {http, HttpResponse, type JsonBodyType} from "msw";
import {env} from "../config/env";
import {server} from "../../test/msw/server";
import {fetchFacility} from "./facilityParser";

const LIVE_COUNTS_URL = env.apiBaseUrl
    ? `${env.apiBaseUrl}/api/live-counts`
    : `${window.location.origin}/api/live-counts`;

const healthyIngestion = {
    lastSuccessfulFetchAt: "2026-08-31T12:00:00Z",
    ageSeconds: 0,
    status: "healthy" as const,
};

const row = (overrides: Record<string, unknown> = {}) => ({
    LocationId: 5761,
    IsClosed: false,
    LastCount: 47,
    LastUpdatedDateAndTime: null,
    FetchedAt: "2026-08-31T12:00:00Z",
    ...overrides,
});

const mockLiveCounts = (payload: JsonBodyType) => {
    server.use(http.get(LIVE_COUNTS_URL, () => HttpResponse.json(payload)));
};

describe("fetchFacility live-count parsing", () => {
    it("maps a schema-valid canonical row without changing the facility layout", async () => {
        mockLiveCounts({
            ingestion: healthyIngestion,
            rows: [row()],
        });

        await expect(fetchFacility(1186)).resolves.toMatchObject({
            facilityId: 1186,
            liveDataSource: "facility_api",
            locations: expect.arrayContaining([
                expect.objectContaining({
                    facilityId: 1186,
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

    it("keeps a missing live count nullable instead of turning it into zero", async () => {
        mockLiveCounts({
            ingestion: healthyIngestion,
            rows: [row({LastCount: null})],
        });

        await expect(fetchFacility(1186)).resolves.toMatchObject({
            locations: expect.arrayContaining([
                expect.objectContaining({
                    locationId: 5761,
                    currentCapacity: null,
                }),
            ]),
        });
    });

    it.each([
        ["malformed row", {
            ingestion: healthyIngestion,
            rows: [row({LocationId: "5761"})],
        }],
        ["empty feed", {
            ingestion: healthyIngestion,
            rows: [],
        }],
    ])("rejects a %s through the shared response schema", async (_label, payload) => {
        mockLiveCounts(payload);

        await expect(fetchFacility(1186)).rejects.toMatchObject({kind: "schema"});
    });

    it("does not retry the same live URL as its own fallback", async () => {
        let requests = 0;
        server.use(http.get(LIVE_COUNTS_URL, () => {
            requests += 1;
            return new HttpResponse(null, {status: 404});
        }));

        await expect(fetchFacility(1186)).rejects.toMatchObject({status: 404});
        expect(requests).toBe(1);
    });
});
