import axios, {AxiosError} from "axios";
import {delay, http, HttpResponse, type JsonBodyType} from "msw";
import {env} from "../config/env";
import {server} from "../../test/msw/server";
import {fetchFacility} from "./facilityParser";

const OFFICIAL_COUNTS_URL = "https://goboardapi.azurewebsites.net/api/FacilityCount/GetCountsByAccount";
const LIVE_COUNTS_URL = env.apiBaseUrl
    ? `${env.apiBaseUrl}/api/live-counts`
    : `${window.location.origin}/api/live-counts`;

const NICK_LOCATION_IDS = [5761, 5764, 5760, 7089, 5762, 5758, 7090, 5766, 5753, 5754, 5763];

const healthyIngestion = {
    lastSuccessfulFetchAt: "2026-08-31T12:00:00Z",
    ageSeconds: 0,
    status: "healthy" as const,
};

const officialRow = (locationId: number, overrides: Record<string, unknown> = {}) => ({
    FacilityId: locationId >= 8000 ? 1656 : 1186,
    FacilityName: locationId >= 8000 ? "Bakke Recreation & Wellbeing Center" : "Nicholas Recreation Center",
    LocationId: locationId,
    LocationName: `Location ${locationId}`,
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

const backupRow = (locationId: number, fetchedAt: string | null, overrides: Record<string, unknown> = {}) => ({
    LocationId: locationId,
    IsClosed: false,
    LastCount: 47,
    LastUpdatedDateAndTime: "2026-08-31T11:59:00Z",
    FetchedAt: fetchedAt,
    ...overrides,
});

const officialRows = (ids = NICK_LOCATION_IDS, overrides: Record<string, unknown> = {}) => (
    ids.map((locationId) => officialRow(locationId, overrides))
);

const backupRows = (fetchedAt: string | null, ids = NICK_LOCATION_IDS) => (
    ids.map((locationId) => backupRow(locationId, fetchedAt))
);

const mockOfficial = (payload: JsonBodyType, status = 200) => {
    server.use(http.get(OFFICIAL_COUNTS_URL, () => HttpResponse.json(payload, {status})));
};

const mockBackup = (payload: JsonBodyType, onRequest?: () => void) => {
    server.use(http.get(LIVE_COUNTS_URL, () => {
        onRequest?.();
        return HttpResponse.json(payload);
    }));
};

describe("fetchFacility official-first live-count parsing", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime("2026-08-31T12:00:00Z");
    });

    it("uses a sufficiently complete official response without calling the backup", async () => {
        let backupRequests = 0;
        mockOfficial(officialRows());
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });

        const result = await fetchFacility(1186);

        expect(result).toMatchObject({
            facilityId: 1186,
            liveDataSource: "official_api",
            locations: expect.arrayContaining([
                expect.objectContaining({
                    locationId: 5761,
                    currentCapacity: 47,
                    fetchedAt: "2026-08-31T12:00:00.000Z",
                }),
            ]),
        });
        expect(backupRequests).toBe(0);
    });

    it.each([
        ["HTTP failure", () => HttpResponse.json({error: "unavailable"}, {status: 503})],
        ["invalid JSON", () => HttpResponse.text("not-json")],
        ["schema failure", () => HttpResponse.json([{LocationId: "5761"}])],
        ["empty response", () => HttpResponse.json([])],
        ["insufficient selected-facility coverage", () => HttpResponse.json([officialRow(5761)])],
    ])("uses the server backup once after official %s", async (_label, officialResponse) => {
        let backupRequests = 0;
        server.use(http.get(OFFICIAL_COUNTS_URL, officialResponse));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });

        await expect(fetchFacility(1186)).resolves.toMatchObject({
            facilityId: 1186,
            liveDataSource: "facility_api",
            locations: expect.arrayContaining([
                expect.objectContaining({locationId: 5761, currentCapacity: 47}),
            ]),
        });
        expect(backupRequests).toBe(1);
    });

    it("uses the server backup after an official network failure", async () => {
        let backupRequests = 0;
        server.use(http.get(OFFICIAL_COUNTS_URL, () => HttpResponse.error()));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });

        await expect(fetchFacility(1186)).resolves.toMatchObject({liveDataSource: "facility_api"});
        expect(backupRequests).toBe(1);
    });

    it("uses the server backup once after the official request times out", async () => {
        let backupRequests = 0;
        server.use(http.get(OFFICIAL_COUNTS_URL, async () => {
            await delay("infinite");
            return HttpResponse.json(officialRows());
        }));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });

        const pending = fetchFacility(1186);
        await vi.advanceTimersByTimeAsync(4_001);

        await expect(pending).resolves.toMatchObject({liveDataSource: "facility_api"});
        expect(backupRequests).toBe(1);
    });

    it("does not let cross-facility rows with matching location IDs qualify", async () => {
        let backupRequests = 0;
        mockOfficial(officialRows(NICK_LOCATION_IDS, {FacilityId: 1656}));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });

        await expect(fetchFacility(1186)).resolves.toMatchObject({liveDataSource: "facility_api"});
        expect(backupRequests).toBe(1);
    });

    it("accepts an official response that confirms every configured location is closed", async () => {
        let backupRequests = 0;
        mockOfficial(officialRows(NICK_LOCATION_IDS, {IsClosed: true, LastCount: null}));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });

        const result = await fetchFacility(1186);

        expect(result.liveDataSource).toBe("official_api");
        expect(result.locations).toHaveLength(NICK_LOCATION_IDS.length);
        expect(result.locations.every((location) => (
            location.isClosed === true
            && location.currentCapacity === null
            && location.fetchedAt === "2026-08-31T12:00:00.000Z"
        ))).toBe(true);
        expect(backupRequests).toBe(0);
    });

    it("stamps successful official observations at receipt time, ignoring supplied freshness claims", async () => {
        mockOfficial(officialRows(NICK_LOCATION_IDS, {FetchedAt: "2020-01-01T00:00:00Z"}));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")});

        const result = await fetchFacility(1186);

        expect(result.locations.map((location) => location.fetchedAt)).toEqual(
            NICK_LOCATION_IDS.map(() => "2026-08-31T12:00:00.000Z"),
        );
    });

    it.each([
        ["stale", "2026-08-31T11:49:59Z"],
        ["future", "2026-08-31T12:00:01Z"],
    ])("rejects a %s backup instead of restamping its rows", async (_label, fetchedAt) => {
        mockOfficial([]);
        mockBackup({ingestion: healthyIngestion, rows: backupRows(fetchedAt)});

        await expect(fetchFacility(1186)).rejects.toMatchObject({kind: "schema"});
    });

    it("does not call the backup when the caller aborts the official request", async () => {
        let backupRequests = 0;
        server.use(http.get(OFFICIAL_COUNTS_URL, async () => {
            await delay("infinite");
            return HttpResponse.json(officialRows());
        }));
        mockBackup({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")}, () => {
            backupRequests += 1;
        });
        const controller = new AbortController();

        const pending = fetchFacility(1186, controller.signal);
        controller.abort();

        await expect(pending).rejects.toMatchObject({kind: "aborted"});
        expect(backupRequests).toBe(0);
    });

    it("reports the single backup failure after the official response fails", async () => {
        let backupRequests = 0;
        mockOfficial([], 200);
        server.use(http.get(LIVE_COUNTS_URL, () => {
            backupRequests += 1;
            return HttpResponse.json({error: "unavailable"}, {status: 503});
        }));

        await expect(fetchFacility(1186)).rejects.toMatchObject({kind: "http", status: 503});
        expect(backupRequests).toBe(1);
    });

    it("times out the one backup attempt after five seconds", async () => {
        mockOfficial([]);
        const backupRequest = vi.spyOn(axios, "request").mockRejectedValue(
            new AxiosError("deadline exceeded", "ECONNABORTED"),
        );

        await expect(fetchFacility(1186)).rejects.toMatchObject({kind: "timeout"});
        expect(backupRequest).toHaveBeenCalledTimes(1);
        expect(backupRequest).toHaveBeenCalledWith(expect.objectContaining({timeout: 5_000}));
    });

    it("cancels an in-flight backup without retrying it", async () => {
        let backupRequests = 0;
        let markBackupStarted: (() => void) | undefined;
        const backupStarted = new Promise<void>((resolve) => {
            markBackupStarted = resolve;
        });
        mockOfficial([]);
        server.use(http.get(LIVE_COUNTS_URL, async () => {
            backupRequests += 1;
            markBackupStarted?.();
            await delay("infinite");
            return HttpResponse.json({ingestion: healthyIngestion, rows: backupRows("2026-08-31T12:00:00Z")});
        }));
        const controller = new AbortController();

        const pending = fetchFacility(1186, controller.signal);
        const rejection = expect(pending).rejects.toMatchObject({kind: "aborted"});
        await backupStarted;
        controller.abort();

        await rejection;
        expect(backupRequests).toBe(1);
    });
});
