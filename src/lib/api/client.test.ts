import axios, {AxiosError} from "axios";
import type {AxiosResponse} from "axios";
import {delay, HttpResponse, http} from "msw";
import {describe, expect, it, vi} from "vitest";
import {z} from "zod";
import {server} from "../../test/msw/server";
import {
    ApiError,
    parseRetryAfterMs,
    requestJson,
    shouldRetryApiError,
    uniqueApiUrls,
} from "./client";

vi.mock("../config/env", () => ({
    env: {apiBaseUrl: "https://api.example.test/base"},
}));

const EXAMPLE_URL = "https://api.example.test/base/api/example";
const okSchema = z.object({ok: z.literal(true)}).strict();

const axiosTextResponse = (data: string): AxiosResponse<string> => ({
    data,
    status: 200,
    statusText: "OK",
    headers: {},
    config: {headers: {} as AxiosResponse["config"]["headers"]},
});

describe("requestJson retry policy", () => {
    it("defaults GET requests to three attempts", async () => {
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return new HttpResponse(null, {status: 503});
        }));

        await expect(requestJson("/api/example", okSchema, {
            sleep: async () => undefined,
            jitter: () => 0,
        })).rejects.toMatchObject({kind: "http", status: 503});
        expect(calls).toBe(3);
    });

    it.each([
        ["POST", http.post],
        ["DELETE", http.delete],
    ] as const)("defaults %s requests to one attempt", async (method, handler) => {
        let calls = 0;
        server.use(handler(EXAMPLE_URL, () => {
            calls += 1;
            return new HttpResponse(null, {status: 503});
        }));

        await expect(requestJson("/api/example", okSchema, {
            method,
            sleep: async () => undefined,
        })).rejects.toMatchObject({kind: "http", status: 503});
        expect(calls).toBe(1);
    });

    it("allows a POST endpoint to opt into bounded retries", async () => {
        let calls = 0;
        server.use(http.post(EXAMPLE_URL, () => {
            calls += 1;
            return calls === 1
                ? new HttpResponse(null, {status: 503})
                : HttpResponse.json({ok: true});
        }));

        await expect(requestJson("/api/example", okSchema, {
            method: "POST",
            attempts: 2,
            sleep: async () => undefined,
        })).resolves.toEqual({ok: true});
        expect(calls).toBe(2);
    });

    it("caps an excessive explicit attempt count", async () => {
        const sleep = vi.fn(async () => undefined);
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return new HttpResponse(null, {status: 503});
        }));

        await expect(requestJson("/api/example", okSchema, {
            attempts: 99,
            jitter: () => 1,
            sleep,
        })).rejects.toMatchObject({status: 503});
        expect(calls).toBe(5);
        expect(sleep).toHaveBeenCalledTimes(4);
        expect(sleep).toHaveBeenLastCalledWith(6_000, undefined);
    });

    it.each([0, -1, 1.5, Number.NaN, Number.POSITIVE_INFINITY])(
        "rejects invalid explicit attempts %s before making a request",
        async (attempts) => {
            let calls = 0;
            server.use(http.get(EXAMPLE_URL, () => {
                calls += 1;
                return HttpResponse.json({ok: true});
            }));

            await expect(requestJson("/api/example", okSchema, {attempts}))
                .rejects.toMatchObject({kind: "client", message: "Invalid API request options"});
            expect(calls).toBe(0);
        },
    );

    it.each([408, 429, 500, 503, 599])("retries transient HTTP %s", async (status) => {
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return calls === 1
                ? new HttpResponse(null, {status})
                : HttpResponse.json({ok: true});
        }));

        await expect(requestJson("/api/example", okSchema, {
            attempts: 2,
            sleep: async () => undefined,
        })).resolves.toEqual({ok: true});
        expect(calls).toBe(2);
    });

    it.each([400, 401, 403, 404, 409, 422, 499, 600])(
        "does not retry HTTP %s",
        async (status) => {
            let calls = 0;
            server.use(http.get(EXAMPLE_URL, () => {
                calls += 1;
                return new HttpResponse(null, {status});
            }));

            await expect(requestJson("/api/example", okSchema, {
                attempts: 3,
                sleep: async () => undefined,
            })).rejects.toMatchObject({kind: "http", status});
            expect(calls).toBe(1);
        },
    );

    it("retries a network failure", async () => {
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return calls === 1 ? HttpResponse.error() : HttpResponse.json({ok: true});
        }));

        await expect(requestJson("/api/example", okSchema, {
            attempts: 2,
            sleep: async () => undefined,
        })).resolves.toEqual({ok: true});
        expect(calls).toBe(2);
    });

    it("normalizes and retries an Axios timeout", async () => {
        vi.spyOn(axios, "request")
            .mockRejectedValueOnce(new AxiosError("private timeout detail", "ECONNABORTED"))
            .mockResolvedValueOnce(axiosTextResponse('{"ok":true}'));

        await expect(requestJson("/api/example", okSchema, {
            attempts: 2,
            sleep: async () => undefined,
        })).resolves.toEqual({ok: true});
    });

    it("treats an unknown pre-dispatch failure as a non-retryable client error", async () => {
        const request = vi.spyOn(axios, "request").mockImplementation(() => {
            throw new TypeError("circular-private-sentinel");
        });

        const error = await requestJson("/api/example", okSchema, {
            attempts: 3,
            sleep: async () => undefined,
        }).catch((cause: unknown) => cause);

        expect(error).toMatchObject({
            kind: "client",
            message: "API request could not be prepared",
        });
        expect((error as Error).message).not.toContain("circular-private-sentinel");
        expect(request).toHaveBeenCalledTimes(1);
    });

    it("does not retry a response-less Axios configuration error", async () => {
        const request = vi.spyOn(axios, "request").mockRejectedValue(
            new AxiosError(
                "private bad-option detail",
                "ERR_BAD_OPTION_VALUE",
                undefined,
                {},
            ),
        );

        await expect(requestJson("/api/example", okSchema, {
            attempts: 3,
            sleep: async () => undefined,
        })).rejects.toMatchObject({
            kind: "client",
            message: "API request could not be prepared",
        });
        expect(request).toHaveBeenCalledTimes(1);
    });

    it("does not retry invalid JSON", async () => {
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return new HttpResponse("not-json", {
                headers: {"Content-Type": "application/json"},
            });
        }));

        await expect(requestJson("/api/example", okSchema, {
            attempts: 3,
            sleep: async () => undefined,
        })).rejects.toMatchObject({
            kind: "invalid_json",
            message: "API returned invalid JSON",
        });
        expect(calls).toBe(1);
    });

    it("does not retry schema failures or expose schema details", async () => {
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return HttpResponse.json({ok: "private-schema-sentinel"});
        }));

        const error = await requestJson("/api/example", okSchema, {
            attempts: 3,
            sleep: async () => undefined,
        }).catch((cause: unknown) => cause);

        expect(error).toMatchObject({
            kind: "schema",
            message: "API response did not match its contract",
        });
        expect(String((error as Error).message)).not.toContain("private-schema-sentinel");
        expect(calls).toBe(1);
    });
});

describe("bounded retry timing", () => {
    it("honors a bounded Retry-After seconds header", async () => {
        const sleep = vi.fn(async () => undefined);
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return calls === 1
                ? new HttpResponse(null, {status: 429, headers: {"Retry-After": "2"}})
                : HttpResponse.json({ok: true});
        }));

        await requestJson("/api/example", okSchema, {attempts: 2, sleep});
        expect(sleep).toHaveBeenCalledWith(2_000, undefined);
    });

    it("caps an excessive Retry-After header", async () => {
        const sleep = vi.fn(async () => undefined);
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return calls === 1
                ? new HttpResponse(null, {status: 503, headers: {"Retry-After": "999999"}})
                : HttpResponse.json({ok: true});
        }));

        await requestJson("/api/example", okSchema, {attempts: 2, sleep});
        expect(sleep).toHaveBeenCalledWith(60_000, undefined);
    });

    it.each([
        [() => -100, 250],
        [() => 100, 750],
        [() => Number.NaN, 500],
        [() => Number.POSITIVE_INFINITY, 500],
    ])("bounds jitter output", async (jitter, expectedDelay) => {
        const sleep = vi.fn(async () => undefined);
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return calls === 1
                ? new HttpResponse(null, {status: 503})
                : HttpResponse.json({ok: true});
        }));

        await requestJson("/api/example", okSchema, {attempts: 2, jitter, sleep});
        expect(sleep).toHaveBeenCalledWith(expectedDelay, undefined);
    });

    it("caps an excessive timeout before passing it to the transport", async () => {
        const request = vi.spyOn(axios, "request")
            .mockResolvedValueOnce(axiosTextResponse('{"ok":true}'));

        await requestJson("/api/example", okSchema, {timeoutMs: 999_999});

        expect(request).toHaveBeenCalledWith(expect.objectContaining({timeout: 30_000}));
    });

    it.each([0, -1, Number.NaN, Number.POSITIVE_INFINITY])(
        "rejects invalid timeout %s before invoking the transport",
        async (timeoutMs) => {
            const request = vi.spyOn(axios, "request");

            await expect(requestJson("/api/example", okSchema, {timeoutMs}))
                .rejects.toMatchObject({kind: "client", message: "Invalid API request options"});
            expect(request).not.toHaveBeenCalled();
        },
    );

    it("parses seconds and dates and bounds Retry-After values", () => {
        const now = Date.parse("2026-09-01T12:00:00Z");

        expect(parseRetryAfterMs("2", now)).toBe(2_000);
        expect(parseRetryAfterMs("Tue, 01 Sep 2026 12:00:05 GMT", now)).toBe(5_000);
        expect(parseRetryAfterMs("Tue, 01 Sep 2026 11:59:59 GMT", now)).toBe(0);
        expect(parseRetryAfterMs("999999", now)).toBe(60_000);
        expect(parseRetryAfterMs("Mon, 01 Sep 2036 12:00:00 GMT", now)).toBe(60_000);
        expect(parseRetryAfterMs("1.5", now)).toBeNull();
        expect(parseRetryAfterMs("not-a-delay", now)).toBeNull();
        expect(parseRetryAfterMs(undefined, now)).toBeNull();
    });

    it.each([
        "-1",
        "+1",
        "Sep 1",
        "Mon, 31 Feb 2026 12:00:00 GMT",
        "Mon, 01 Sep 2026 12:00:00 GMT",
    ])("rejects a noncanonical Retry-After date %#", (value) => {
        expect(parseRetryAfterMs(value, Date.parse("2026-09-01T12:00:00Z"))).toBeNull();
    });
});

describe("safe URL construction", () => {
    it("appends a canonical API path beneath the configured base path", async () => {
        let seenUrl = "";
        server.use(http.get(EXAMPLE_URL, ({request}) => {
            seenUrl = request.url;
            return HttpResponse.json({ok: true});
        }));

        await expect(requestJson("api//example/", okSchema, {
            params: {facilityId: 1186},
        })).resolves.toEqual({ok: true});
        expect(seenUrl).toBe(`${EXAMPLE_URL}?facilityId=1186`);
    });

    it.each([
        "https://evil.example/api/example",
        "http://evil.example/api/example",
        "//evil.example/api/example",
        "../api/example",
        "/api/../private",
        "/api/%2e%2e/private",
        "/api/%252e%252e/private",
        "\\\\evil.example\\api\\example",
        "/not-api/example",
        "/api/example?next=https://evil.example",
        "/api/example#private",
    ])("rejects unsafe caller path %# without making a request", async (path) => {
        const request = vi.spyOn(axios, "request");

        const error = await requestJson(path, okSchema, {attempts: 5})
            .catch((cause: unknown) => cause);

        expect(error).toMatchObject({
            kind: "client",
            message: "Invalid API request path",
            status: null,
            retryAfterMs: null,
        });
        expect((error as Error).message).not.toContain(path);
        expect(request).not.toHaveBeenCalled();
    });

    it("canonicalizes and deduplicates equivalent safe API paths", () => {
        expect(uniqueApiUrls([
            "/api/live-counts",
            "api/live-counts",
            "/api//live-counts/",
            "/api/forecast/facilities/1186",
        ])).toEqual([
            "/api/live-counts",
            "/api/forecast/facilities/1186",
        ]);
    });

    it.each([
        "https://evil.example/api/live-counts",
        "//evil.example/api/live-counts",
        "/api/../live-counts",
    ])("does not admit unsafe candidate URL %#", (path) => {
        expect(() => uniqueApiUrls(["/api/live-counts", path]))
            .toThrow(expect.objectContaining({kind: "client"}));
    });
});

describe("safe errors and cancellation", () => {
    it("exposes explicit safe ApiError fields", () => {
        const error = new ApiError("http", "API returned HTTP 503", 503, 2_000);

        expect(error).toBeInstanceOf(Error);
        expect(error).toMatchObject({
            name: "ApiError",
            kind: "http",
            message: "API returned HTTP 503",
            status: 503,
            retryAfterMs: 2_000,
        });
    });

    it("never exposes response bodies or request URLs in HTTP errors", async () => {
        const privateBody = "private-response-sentinel";
        server.use(http.get(EXAMPLE_URL, () => new HttpResponse(privateBody, {status: 500})));

        const error = await requestJson("/api/example", okSchema, {attempts: 1})
            .catch((cause: unknown) => cause);

        expect(error).toMatchObject({
            kind: "http",
            status: 500,
            message: "API returned HTTP 500",
        });
        expect((error as Error).message).not.toContain(privateBody);
        expect((error as Error).message).not.toContain(EXAMPLE_URL);
    });

    it("normalizes timeout details to a fixed safe message", async () => {
        vi.spyOn(axios, "request").mockRejectedValueOnce(
            new AxiosError("https://private.example/secret timeout sentinel", "ETIMEDOUT"),
        );

        await expect(requestJson("/api/example", okSchema, {attempts: 1}))
            .rejects.toMatchObject({kind: "timeout", message: "Request timed out"});
    });

    it("marks only transient normalized errors as retryable", () => {
        expect(shouldRetryApiError(new ApiError("network", "Network request failed"))).toBe(true);
        expect(shouldRetryApiError(new ApiError("timeout", "Request timed out"))).toBe(true);
        expect(shouldRetryApiError(new ApiError("http", "API returned HTTP 408", 408))).toBe(true);
        expect(shouldRetryApiError(new ApiError("http", "API returned HTTP 429", 429))).toBe(true);
        expect(shouldRetryApiError(new ApiError("http", "API returned HTTP 500", 500))).toBe(true);
        expect(shouldRetryApiError(new ApiError("http", "API returned HTTP 599", 599))).toBe(true);

        expect(shouldRetryApiError(new ApiError("aborted", "Request aborted"))).toBe(false);
        expect(shouldRetryApiError(new ApiError("client", "Invalid API request"))).toBe(false);
        expect(shouldRetryApiError(new ApiError("invalid_json", "API returned invalid JSON"))).toBe(false);
        expect(shouldRetryApiError(new ApiError("schema", "Invalid response"))).toBe(false);
        expect(shouldRetryApiError(new ApiError("http", "API returned HTTP 422", 422))).toBe(false);
        expect(shouldRetryApiError(new ApiError("http", "API returned HTTP 600", 600))).toBe(false);
    });

    it("aborts before the first request without retrying", async () => {
        const controller = new AbortController();
        controller.abort();
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return HttpResponse.json({ok: true});
        }));

        await expect(requestJson("/api/example", okSchema, {
            attempts: 3,
            signal: controller.signal,
        })).rejects.toMatchObject({kind: "aborted", message: "Request aborted"});
        expect(calls).toBe(0);
    });

    it("aborts an in-flight request without starting another attempt", async () => {
        const controller = new AbortController();
        let enteredHandler: (() => void) | undefined;
        const entered = new Promise<void>((resolve) => {
            enteredHandler = resolve;
        });
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, async () => {
            calls += 1;
            enteredHandler?.();
            await delay("infinite");
            return HttpResponse.json({ok: true});
        }));

        const expectation = expect(requestJson("/api/example", okSchema, {
            attempts: 3,
            signal: controller.signal,
        })).rejects.toMatchObject({kind: "aborted"});
        await entered;
        controller.abort();

        await expectation;
        expect(calls).toBe(1);
    });

    it("aborts retry backoff even when injected sleep ignores the signal", async () => {
        const controller = new AbortController();
        let enteredSleep: (() => void) | undefined;
        const sleeping = new Promise<void>((resolve) => {
            enteredSleep = resolve;
        });
        const sleep = (): Promise<void> => new Promise(() => {
            enteredSleep?.();
        });
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return new HttpResponse(null, {status: 503});
        }));

        const expectation = expect(requestJson("/api/example", okSchema, {
            attempts: 3,
            signal: controller.signal,
            sleep,
        })).rejects.toMatchObject({kind: "aborted"});
        await sleeping;
        controller.abort();

        await expectation;
        expect(calls).toBe(1);
    });

    it("lets abort win when injected sleep resolution races cancellation", async () => {
        const controller = new AbortController();
        let enteredSleep: (() => void) | undefined;
        let resolveSleep: (() => void) | undefined;
        const sleeping = new Promise<void>((resolve) => {
            enteredSleep = resolve;
        });
        const sleep = (): Promise<void> => new Promise((resolve) => {
            resolveSleep = resolve;
            enteredSleep?.();
        });
        let calls = 0;
        server.use(http.get(EXAMPLE_URL, () => {
            calls += 1;
            return new HttpResponse(null, {status: 503});
        }));

        const expectation = expect(requestJson("/api/example", okSchema, {
            attempts: 3,
            signal: controller.signal,
            sleep,
        })).rejects.toMatchObject({kind: "aborted"});
        await sleeping;
        resolveSleep?.();
        controller.abort();

        await expectation;
        expect(calls).toBe(1);
    });
});
