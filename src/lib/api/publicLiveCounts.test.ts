import {delay, http, HttpResponse} from "msw";
import {server} from "../../test/msw/server";
import {fetchPublicLiveCounts} from "./publicLiveCounts";

const OFFICIAL_COUNTS_URL = "https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount";

const publicRow = (overrides: Record<string, unknown> = {}) => ({
    FacilityId: 1186,
    FacilityName: "Nicholas Recreation Center",
    LocationId: 5761,
    LocationName: "Nick Power House",
    IsClosed: false,
    LastCount: 47,
    CountOfParticipants: 47,
    LastUpdatedDateAndTime: "2026-08-31T11:59:00Z",
    CountCapacityColorEnabled: true,
    MaxCapacityRange: 100,
    MaxColor: "#000000",
    MidColor: "#000000",
    MinCapacityRange: 0,
    MinColor: "#000000",
    PercetageCapacity: 47,
    SubLocations: null,
    TotalCapacity: 100,
    ...overrides,
});

describe("fetchPublicLiveCounts", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
    });

    it("validates public observations and gives them one receipt timestamp", async () => {
        server.use(http.get(OFFICIAL_COUNTS_URL, () => HttpResponse.json([
            publicRow({FetchedAt: "2020-01-01T00:00:00Z"}),
            publicRow({
                LocationId: 5764,
                LastCount: 12,
                LastUpdatedDateAndTime: "2026-08-31T11:59:00",
            }),
        ])));

        await expect(fetchPublicLiveCounts()).resolves.toEqual([
            {
                FacilityId: 1186,
                LocationId: 5761,
                IsClosed: false,
                LastCount: 47,
                LastUpdatedDateAndTime: "2026-08-31T11:59:00Z",
                FetchedAt: "2026-08-31T12:00:00.000Z",
            },
            {
                FacilityId: 1186,
                LocationId: 5764,
                IsClosed: false,
                LastCount: 12,
                LastUpdatedDateAndTime: null,
                FetchedAt: "2026-08-31T12:00:00.000Z",
            },
        ]);
    });

    it("uses the published identifier without credentials and bypasses browser caches", async () => {
        let requestContract: Record<string, unknown> | null = null;
        server.use(http.get(OFFICIAL_COUNTS_URL, ({request}) => {
            const url = new URL(request.url);
            requestContract = {
                cache: request.cache,
                credentials: request.credentials,
                queryKeys: [...url.searchParams.keys()],
                hasIdentifier: Boolean(url.searchParams.get("AccountAPIKey")),
                authorization: request.headers.get("authorization"),
                cookie: request.headers.get("cookie"),
            };
            return HttpResponse.json([publicRow()]);
        }));

        await fetchPublicLiveCounts();

        expect(requestContract).toEqual({
            cache: "no-store",
            credentials: "omit",
            queryKeys: ["AccountAPIKey"],
            hasIdentifier: true,
            authorization: null,
            cookie: null,
        });
    });

    it("returns a sanitized schema error without reflecting the URL or payload", async () => {
        server.use(http.get(OFFICIAL_COUNTS_URL, () => HttpResponse.json([
            publicRow({LocationId: "payload-sentinel"}),
        ])));

        const error = await fetchPublicLiveCounts().catch((cause: unknown) => cause);

        expect(error).toMatchObject({kind: "schema"});
        expect(String(error)).not.toContain("payload-sentinel");
        expect(String(error)).not.toContain("goboardapi");
    });

    it("distinguishes its deadline from a caller cancellation", async () => {
        server.use(http.get(OFFICIAL_COUNTS_URL, async () => {
            await delay("infinite");
            return HttpResponse.json([publicRow()]);
        }));

        const pending = fetchPublicLiveCounts({timeoutMs: 4_000});
        const rejection = expect(pending).rejects.toMatchObject({kind: "timeout"});
        await vi.advanceTimersByTimeAsync(4_001);

        await rejection;
    });
});
