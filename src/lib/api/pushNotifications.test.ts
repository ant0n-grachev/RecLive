import {http, HttpResponse, type JsonBodyType} from "msw";
import {server} from "../../test/msw/server";
import * as pushNotifications from "./pushNotifications";

interface PushRuleFixture {
    id: number;
    facilityId: number;
    sectionKey: string;
    threshold: number;
    createdAt: string;
    expiresAt: string;
    status: "pending";
}

interface SubscribePayloadFixture {
    subscription: PushSubscriptionJSON;
    facilityId: number;
    sectionKey: string;
    threshold: number;
    ttlSeconds?: number;
}

interface PushApiContract {
    subscribePushRule?: (payload: SubscribePayloadFixture) => Promise<{
        created: boolean;
        rule: PushRuleFixture;
    }>;
    listPushRules?: (subscription: PushSubscriptionJSON) => Promise<PushRuleFixture[]>;
    cancelPushRule?: (id: number, subscription: PushSubscriptionJSON) => Promise<void>;
    cancelAllPushRules?: (subscription: PushSubscriptionJSON) => Promise<number>;
    getPushAvailability: () => Promise<{
        apiAvailable: boolean;
        dbAvailable: boolean;
        alertsAvailable: boolean;
        reason: string | null;
    }>;
}

const api = pushNotifications as unknown as PushApiContract;

const requireApi = <Name extends keyof PushApiContract>(
    name: Name
): NonNullable<PushApiContract[Name]> => {
    const implementation = api[name];
    expect(implementation).toEqual(expect.any(Function));
    return implementation as NonNullable<PushApiContract[Name]>;
};

const endpoint = "https://push.reclive-notify.net/private-endpoint-token";
const p256dh = "private-p256dh-token";
const auth = "private-auth-token";
const serverSentinel = "private-server-sentinel";

const validSubscription: PushSubscriptionJSON = {
    endpoint,
    expirationTime: 1_800_000_000_000,
    keys: {p256dh, auth},
};

const canonicalSubscription = {
    endpoint,
    keys: {p256dh, auth},
};

const validPayload: SubscribePayloadFixture = {
    subscription: validSubscription,
    facilityId: 1186,
    sectionKey: "overall",
    threshold: 40,
};

const validRule: PushRuleFixture = {
    id: 7,
    facilityId: 1186,
    sectionKey: "overall",
    threshold: 40,
    createdAt: "2026-09-01T12:00:00Z",
    expiresAt: "2026-09-02T12:00:00Z",
    status: "pending",
};

const secondRule: PushRuleFixture = {
    id: 9,
    facilityId: 1656,
    sectionKey: "fitness floors",
    threshold: 25,
    createdAt: "2026-09-01T08:00:00-04:00",
    expiresAt: "2026-09-02T08:00:00-04:00",
    status: "pending",
};

const subscribeEnvelope = (
    overrides: Record<string, unknown> = {}
): Record<string, unknown> => ({
    status: "ok",
    created: true,
    rule: validRule,
    ...overrides,
});

const omitKey = (value: Record<string, unknown>, key: string): Record<string, unknown> => (
    Object.fromEntries(Object.entries(value).filter(([entryKey]) => entryKey !== key))
);

const expectFixedError = async (promise: Promise<unknown>, message: string): Promise<void> => {
    await expect(promise).rejects.toEqual(new Error(message));
};

describe("server-backed push rule wire contract", () => {
    it("subscribes with the exact canonical JSON request and returns a created safe rule", async () => {
        let observedRequest: {
            method: string;
            path: string;
            contentType: string | null;
            body: unknown;
        } | null = null;
        server.use(http.post("*/api/push/subscribe", async ({request}) => {
            observedRequest = {
                method: request.method,
                path: new URL(request.url).pathname,
                contentType: request.headers.get("content-type"),
                body: await request.json(),
            };
            return HttpResponse.json(subscribeEnvelope());
        }));

        const result = await requireApi("subscribePushRule")({
            ...validPayload,
            ttlSeconds: 86_400,
            runtimeOnlyExtra: serverSentinel,
        } as SubscribePayloadFixture);

        expect(observedRequest).toEqual({
            method: "POST",
            path: "/api/push/subscribe",
            contentType: "application/json",
            body: {
                subscription: canonicalSubscription,
                facilityId: 1186,
                sectionKey: "overall",
                threshold: 40,
                ttlSeconds: 86_400,
            },
        });
        expect(result).toEqual({created: true, rule: validRule});
        expect(JSON.stringify(result)).not.toContain(endpoint);
        expect(JSON.stringify(result)).not.toContain(p256dh);
        expect(JSON.stringify(result)).not.toContain(auth);
    });

    it("preserves the server's false created result without adding private fields", async () => {
        server.use(http.post("*/api/push/subscribe", () => HttpResponse.json(
            subscribeEnvelope({created: false})
        )));

        await expect(requireApi("subscribePushRule")(validPayload)).resolves.toEqual({
            created: false,
            rule: validRule,
        });
    });

    it("lists rules in deterministic server order with the exact canonical request", async () => {
        let observedRequest: {
            method: string;
            path: string;
            contentType: string | null;
            body: unknown;
        } | null = null;
        server.use(http.post("*/api/push/rules/list", async ({request}) => {
            observedRequest = {
                method: request.method,
                path: new URL(request.url).pathname,
                contentType: request.headers.get("content-type"),
                body: await request.json(),
            };
            return HttpResponse.json({status: "ok", rules: [secondRule, validRule]});
        }));

        const result = await requireApi("listPushRules")(validSubscription);

        expect(observedRequest).toEqual({
            method: "POST",
            path: "/api/push/rules/list",
            contentType: "application/json",
            body: {subscription: canonicalSubscription},
        });
        expect(result).toEqual([secondRule, validRule]);
    });

    it("cancels one positive safe rule id with the exact DELETE request", async () => {
        let observedRequest: {
            method: string;
            path: string;
            contentType: string | null;
            body: unknown;
        } | null = null;
        server.use(http.delete("*/api/push/rules/7", async ({request}) => {
            observedRequest = {
                method: request.method,
                path: new URL(request.url).pathname,
                contentType: request.headers.get("content-type"),
                body: await request.json(),
            };
            return HttpResponse.json({status: "ok", cancelled: 1});
        }));

        await expect(requireApi("cancelPushRule")(7, validSubscription)).resolves.toBeUndefined();
        expect(observedRequest).toEqual({
            method: "DELETE",
            path: "/api/push/rules/7",
            contentType: "application/json",
            body: {subscription: canonicalSubscription},
        });
    });

    it("cancels all rules with the exact canonical POST and returns a safe count", async () => {
        let observedRequest: {
            method: string;
            path: string;
            contentType: string | null;
            body: unknown;
        } | null = null;
        server.use(http.post("*/api/push/rules/cancel-all", async ({request}) => {
            observedRequest = {
                method: request.method,
                path: new URL(request.url).pathname,
                contentType: request.headers.get("content-type"),
                body: await request.json(),
            };
            return HttpResponse.json({status: "ok", cancelled: 2});
        }));

        await expect(requireApi("cancelAllPushRules")(validSubscription)).resolves.toBe(2);
        expect(observedRequest).toEqual({
            method: "POST",
            path: "/api/push/rules/cancel-all",
            contentType: "application/json",
            body: {subscription: canonicalSubscription},
        });
    });
});

describe("browser subscription expirationTime canonicalization", () => {
    it.each([
        ["absent", {endpoint, keys: {p256dh, auth}}],
        ["undefined", {endpoint, expirationTime: undefined, keys: {p256dh, auth}}],
        ["null", {endpoint, expirationTime: null, keys: {p256dh, auth}}],
        ["zero", {endpoint, expirationTime: 0, keys: {p256dh, auth}}],
        ["finite fractional", {endpoint, expirationTime: 0.5, keys: {p256dh, auth}}],
        ["finite numeric", {endpoint, expirationTime: 1_800_000_000_000, keys: {p256dh, auth}}],
    ])("accepts %s and always omits it from ownership requests", async (_name, subscription) => {
        let observedBody: unknown;
        server.use(http.post("*/api/push/rules/list", async ({request}) => {
            observedBody = await request.json();
            return HttpResponse.json({status: "ok", rules: []});
        }));

        await expect(requireApi("listPushRules")(
            subscription as PushSubscriptionJSON
        )).resolves.toEqual([]);
        expect(observedBody).toEqual({subscription: canonicalSubscription});
    });

    it.each([
        ["boolean", true],
        ["string", "1800000000000"],
        ["negative", -1],
        ["NaN", Number.NaN],
        ["positive infinity", Number.POSITIVE_INFINITY],
        ["negative infinity", Number.NEGATIVE_INFINITY],
    ])("rejects %s locally with the fixed private error", async (_name, expirationTime) => {
        let requestCount = 0;
        server.use(http.all("*/api/push/*", () => {
            requestCount += 1;
            return HttpResponse.json({status: "ok", rules: []});
        }));
        const subscription = {
            endpoint,
            expirationTime,
            keys: {p256dh, auth},
        } as unknown as PushSubscriptionJSON;

        await expectFixedError(
            requireApi("listPushRules")(subscription),
            "Push subscription unavailable."
        );
        expect(requestCount).toBe(0);
    });
});

describe("strict push rule response parsing", () => {
    const malformedSubscribeResponses: Array<[string, JsonBodyType]> = [
        ["missing envelope field", {status: "ok", rule: validRule}],
        ["extra envelope field", subscribeEnvelope({debug: serverSentinel}) as JsonBodyType],
        ["wrong envelope status", subscribeEnvelope({status: "error"}) as JsonBodyType],
        ["number used as boolean", subscribeEnvelope({created: 1}) as JsonBodyType],
        ["missing rule field", subscribeEnvelope({
            rule: omitKey(validRule as unknown as Record<string, unknown>, "expiresAt"),
        }) as JsonBodyType],
        ["extra private rule field", subscribeEnvelope({
            rule: {...validRule, endpoint: serverSentinel},
        }) as JsonBodyType],
        ["boolean rule id", subscribeEnvelope({rule: {...validRule, id: true}}) as JsonBodyType],
        ["zero rule id", subscribeEnvelope({rule: {...validRule, id: 0}}) as JsonBodyType],
        ["fractional rule id", subscribeEnvelope({rule: {...validRule, id: 1.5}}) as JsonBodyType],
        ["unsafe rule id", subscribeEnvelope({
            rule: {...validRule, id: Number.MAX_SAFE_INTEGER + 1},
        }) as JsonBodyType],
        ["wrong facility", subscribeEnvelope({rule: {...validRule, facilityId: 9999}}) as JsonBodyType],
        ["boolean facility", subscribeEnvelope({rule: {...validRule, facilityId: true}}) as JsonBodyType],
        ["threshold below range", subscribeEnvelope({rule: {...validRule, threshold: 0}}) as JsonBodyType],
        ["threshold above range", subscribeEnvelope({rule: {...validRule, threshold: 101}}) as JsonBodyType],
        ["fractional threshold", subscribeEnvelope({rule: {...validRule, threshold: 1.5}}) as JsonBodyType],
        ["boolean threshold", subscribeEnvelope({rule: {...validRule, threshold: true}}) as JsonBodyType],
        ["nonpending status", subscribeEnvelope({rule: {...validRule, status: "sent"}}) as JsonBodyType],
        ["empty section", subscribeEnvelope({rule: {...validRule, sectionKey: ""}}) as JsonBodyType],
        ["upper-case section", subscribeEnvelope({rule: {...validRule, sectionKey: "Fitness Floors"}}) as JsonBodyType],
        ["leading section space", subscribeEnvelope({rule: {...validRule, sectionKey: " overall"}}) as JsonBodyType],
        ["repeated section spaces", subscribeEnvelope({
            rule: {...validRule, sectionKey: "fitness  floors"},
        }) as JsonBodyType],
        ["oversized section", subscribeEnvelope({rule: {...validRule, sectionKey: "x".repeat(81)}}) as JsonBodyType],
        ["timezone-less creation", subscribeEnvelope({
            rule: {...validRule, createdAt: "2026-09-01T12:00:00"},
        }) as JsonBodyType],
        ["invalid calendar date", subscribeEnvelope({
            rule: {...validRule, createdAt: "2026-02-30T12:00:00Z"},
        }) as JsonBodyType],
        ["nonfinite date text", subscribeEnvelope({
            rule: {...validRule, createdAt: "Infinity"},
        }) as JsonBodyType],
        ["equal timestamps", subscribeEnvelope({
            rule: {...validRule, expiresAt: validRule.createdAt},
        }) as JsonBodyType],
        ["reversed timestamps", subscribeEnvelope({
            rule: {...validRule, expiresAt: "2026-08-31T12:00:00Z"},
        }) as JsonBodyType],
    ];

    it.each(malformedSubscribeResponses)("rejects a subscribe response with %s", async (_name, body) => {
        server.use(http.post("*/api/push/subscribe", () => HttpResponse.json(body)));

        await expectFixedError(
            requireApi("subscribePushRule")(validPayload),
            "Could not save this alert right now."
        );
    });

    it("accepts a canonical section containing one legitimate space and explicit offsets", async () => {
        server.use(http.post("*/api/push/subscribe", () => HttpResponse.json({
            status: "ok",
            created: true,
            rule: secondRule,
        })));

        await expect(requireApi("subscribePushRule")({
            ...validPayload,
            facilityId: 1656,
            sectionKey: "fitness floors",
            threshold: 25,
        })).resolves.toEqual({created: true, rule: secondRule});
    });

    it("rejects an entire list when one row is malformed", async () => {
        server.use(http.post("*/api/push/rules/list", () => HttpResponse.json({
            status: "ok",
            rules: [validRule, {...secondRule, status: "failed"}],
        })));

        await expectFixedError(
            requireApi("listPushRules")(validSubscription),
            "Could not load alerts right now."
        );
    });

    it("rejects duplicate list rule ids instead of returning ambiguous actions", async () => {
        server.use(http.post("*/api/push/rules/list", () => HttpResponse.json({
            status: "ok",
            rules: [validRule, {...secondRule, id: validRule.id}],
        })));

        await expectFixedError(
            requireApi("listPushRules")(validSubscription),
            "Could not load alerts right now."
        );
    });

    it.each([
        ["missing rules", {status: "ok"}],
        ["extra list field", {status: "ok", rules: [], debug: serverSentinel}],
        ["wrong list status", {status: "error", rules: []}],
        ["non-array rules", {status: "ok", rules: {}}],
    ])("rejects a list envelope with %s", async (_name, body) => {
        server.use(http.post("*/api/push/rules/list", () => HttpResponse.json(body)));

        await expectFixedError(
            requireApi("listPushRules")(validSubscription),
            "Could not load alerts right now."
        );
    });

    it.each([
        ["zero", 0],
        ["two", 2],
        ["boolean", true],
        ["string", "1"],
    ])("rejects cancel-one count %s", async (_name, cancelled) => {
        server.use(http.delete("*/api/push/rules/7", () => HttpResponse.json({
            status: "ok",
            cancelled,
        })));

        await expectFixedError(
            requireApi("cancelPushRule")(7, validSubscription),
            "Could not cancel this alert right now."
        );
    });

    it.each([
        ["cancel one missing status", "*/api/push/rules/7", "DELETE", {cancelled: 1}],
        ["cancel one wrong status", "*/api/push/rules/7", "DELETE", {status: "error", cancelled: 1}],
        ["cancel one missing count", "*/api/push/rules/7", "DELETE", {status: "ok"}],
        ["cancel all missing status", "*/api/push/rules/cancel-all", "POST", {cancelled: 0}],
        ["cancel all wrong status", "*/api/push/rules/cancel-all", "POST", {status: "error", cancelled: 0}],
        ["cancel all missing count", "*/api/push/rules/cancel-all", "POST", {status: "ok"}],
    ])("rejects %s", async (_name, path, method, body) => {
        server.use(method === "DELETE"
            ? http.delete(path, () => HttpResponse.json(body))
            : http.post(path, () => HttpResponse.json(body)));

        const operation = method === "DELETE"
            ? requireApi("cancelPushRule")(7, validSubscription)
            : requireApi("cancelAllPushRules")(validSubscription);
        await expectFixedError(
            operation,
            method === "DELETE"
                ? "Could not cancel this alert right now."
                : "Could not cancel alerts right now."
        );
    });

    it.each([
        ["negative", -1],
        ["fractional", 1.5],
        ["boolean", true],
        ["unsafe", Number.MAX_SAFE_INTEGER + 1],
    ])("rejects cancel-all count %s", async (_name, cancelled) => {
        server.use(http.post("*/api/push/rules/cancel-all", () => HttpResponse.json({
            status: "ok",
            cancelled,
        })));

        await expectFixedError(
            requireApi("cancelAllPushRules")(validSubscription),
            "Could not cancel alerts right now."
        );
    });

    it.each([
        ["cancel one", "*/api/push/rules/7", "DELETE"],
        ["cancel all", "*/api/push/rules/cancel-all", "POST"],
    ])("rejects an extra field in the %s envelope", async (_name, path, method) => {
        const handler = method === "DELETE"
            ? http.delete(path, () => HttpResponse.json({status: "ok", cancelled: 1, debug: serverSentinel}))
            : http.post(path, () => HttpResponse.json({status: "ok", cancelled: 0, debug: serverSentinel}));
        server.use(handler);

        const promise = method === "DELETE"
            ? requireApi("cancelPushRule")(7, validSubscription)
            : requireApi("cancelAllPushRules")(validSubscription);
        const message = method === "DELETE"
            ? "Could not cancel this alert right now."
            : "Could not cancel alerts right now.";
        await expectFixedError(promise, message);
        await promise.catch((error: unknown) => {
            expect(String(error)).not.toContain(serverSentinel);
        });
    });
});

describe("safe request and failure handling", () => {
    it.each([
        ["missing endpoint", {keys: {p256dh, auth}}],
        ["empty endpoint", {endpoint: "", keys: {p256dh, auth}}],
        ["non-HTTPS endpoint", {endpoint: "http://push.example.test/a", keys: {p256dh, auth}}],
        ["missing keys", {endpoint}],
        ["empty p256dh", {endpoint, keys: {p256dh: "", auth}}],
        ["non-string auth", {endpoint, keys: {p256dh, auth: 7}}],
    ])("rejects %s locally with the same private generic error", async (_name, subscription) => {
        let requestCount = 0;
        server.use(http.all("*/api/push/*", () => {
            requestCount += 1;
            return HttpResponse.json({status: "ok"});
        }));

        await expectFixedError(
            requireApi("listPushRules")(subscription as unknown as PushSubscriptionJSON),
            "Push subscription unavailable."
        );
        expect(requestCount).toBe(0);
    });

    it.each([
        ["wrong facility", {...validPayload, facilityId: 9999}],
        ["empty section", {...validPayload, sectionKey: ""}],
        ["noncanonical section", {...validPayload, sectionKey: "fitness  floors"}],
        ["oversized section", {...validPayload, sectionKey: "x".repeat(81)}],
        ["zero threshold", {...validPayload, threshold: 0}],
        ["threshold above range", {...validPayload, threshold: 101}],
        ["fractional threshold", {...validPayload, threshold: 1.5}],
        ["zero TTL", {...validPayload, ttlSeconds: 0}],
        ["TTL above maximum", {...validPayload, ttlSeconds: 604_801}],
        ["fractional TTL", {...validPayload, ttlSeconds: 1.5}],
        ["unsafe TTL", {...validPayload, ttlSeconds: Number.MAX_SAFE_INTEGER + 1}],
    ])("rejects a locally invalid subscribe payload: %s", async (_name, payload) => {
        let requestCount = 0;
        server.use(http.post("*/api/push/subscribe", () => {
            requestCount += 1;
            return HttpResponse.json(subscribeEnvelope());
        }));

        await expectFixedError(
            requireApi("subscribePushRule")(payload),
            "Could not save this alert right now."
        );
        expect(requestCount).toBe(0);
    });

    it.each([
        ["null payload", null, "Could not save this alert right now."],
        ["array payload", [], "Could not save this alert right now."],
        ["missing subscription", {
            facilityId: 1186,
            sectionKey: "overall",
            threshold: 40,
        }, "Push subscription unavailable."],
    ])("rejects a malformed top-level subscribe %s with a fixed error", async (_name, payload, message) => {
        let requestCount = 0;
        server.use(http.post("*/api/push/subscribe", () => {
            requestCount += 1;
            return HttpResponse.json(subscribeEnvelope());
        }));

        await expectFixedError(
            requireApi("subscribePushRule")(payload as unknown as SubscribePayloadFixture),
            message
        );
        expect(requestCount).toBe(0);
    });

    it("rejects unsafe cancel ids locally without making a request", async () => {
        let requestCount = 0;
        server.use(http.all("*/api/push/*", () => {
            requestCount += 1;
            return HttpResponse.json({status: "ok"});
        }));

        for (const id of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
            await expectFixedError(
                requireApi("cancelPushRule")(id, validSubscription),
                "Could not cancel this alert right now."
            );
        }
        expect(requestCount).toBe(0);
    });

    it.each([
        ["subscribe", "*/api/push/subscribe", "POST", "Could not save this alert right now."],
        ["list", "*/api/push/rules/list", "POST", "Could not load alerts right now."],
        ["cancel one", "*/api/push/rules/7", "DELETE", "Could not cancel this alert right now."],
        ["cancel all", "*/api/push/rules/cancel-all", "POST", "Could not cancel alerts right now."],
    ])("does not read or expose the non-2xx %s response", async (_name, path, method, message) => {
        const response = () => HttpResponse.json({
            detail: serverSentinel,
            endpoint,
            p256dh,
            auth,
        }, {status: 422});
        server.use(method === "DELETE" ? http.delete(path, response) : http.post(path, response));

        const operation = method === "DELETE"
            ? requireApi("cancelPushRule")(7, validSubscription)
            : path.endsWith("/subscribe")
                ? requireApi("subscribePushRule")(validPayload)
                : path.endsWith("/list")
                    ? requireApi("listPushRules")(validSubscription)
                    : requireApi("cancelAllPushRules")(validSubscription);
        await expectFixedError(operation, message);
        await operation.catch((error: unknown) => {
            const serialized = String(error);
            expect(serialized).not.toContain(serverSentinel);
            expect(serialized).not.toContain(endpoint);
            expect(serialized).not.toContain(p256dh);
            expect(serialized).not.toContain(auth);
        });
    });

    it.each([
        ["subscribe", "*/api/push/subscribe", "Could not save this alert right now."],
        ["list", "*/api/push/rules/list", "Could not load alerts right now."],
    ])("converts malformed %s JSON into its fixed safe error", async (_name, path, message) => {
        server.use(http.post(path, () => new HttpResponse("{")));

        const operation = path.endsWith("/subscribe")
            ? requireApi("subscribePushRule")(validPayload)
            : requireApi("listPushRules")(validSubscription);
        await expectFixedError(operation, message);
    });

    it.each([
        ["cancel one", "*/api/push/rules/7", "DELETE", "Could not cancel this alert right now."],
        ["cancel all", "*/api/push/rules/cancel-all", "POST", "Could not cancel alerts right now."],
    ])("converts malformed %s JSON into its fixed safe error", async (_name, path, method, message) => {
        server.use(method === "DELETE"
            ? http.delete(path, () => new HttpResponse("{"))
            : http.post(path, () => new HttpResponse("{")));

        const operation = method === "DELETE"
            ? requireApi("cancelPushRule")(7, validSubscription)
            : requireApi("cancelAllPushRules")(validSubscription);
        await expectFixedError(operation, message);
    });

    it.each([
        ["list", "*/api/push/rules/list", "POST", "Could not load alerts right now."],
        ["cancel one", "*/api/push/rules/7", "DELETE", "Could not cancel this alert right now."],
        ["cancel all", "*/api/push/rules/cancel-all", "POST", "Could not cancel alerts right now."],
    ])("converts a %s transport failure into a fixed safe error", async (_name, path, method, message) => {
        server.use(method === "DELETE"
            ? http.delete(path, () => HttpResponse.error())
            : http.post(path, () => HttpResponse.error()));

        const operation = method === "DELETE"
            ? requireApi("cancelPushRule")(7, validSubscription)
            : path.endsWith("/list")
                ? requireApi("listPushRules")(validSubscription)
                : requireApi("cancelAllPushRules")(validSubscription);
        await expectFixedError(operation, message);
    });
});

describe("push availability parsing", () => {
    it.each([
        [{
            apiAvailable: true,
            dbAvailable: false,
            alertsAvailable: false,
            reason: "push_rules_db_unavailable",
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: false,
            alertsAvailable: false,
            reason: "push_rules_db_unavailable",
        }],
        [{
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_vapid_unconfigured",
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_vapid_unconfigured",
        }],
        [{
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_identity_unconfigured",
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: "push_identity_unconfigured",
        }],
        [{
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "db",
        }, {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
        }],
    ])("returns only the safe four-field value for a consistent envelope", async (wire, expected) => {
        server.use(http.get("*/api/push/availability", () => HttpResponse.json(wire)));

        await expect(api.getPushAvailability()).resolves.toEqual(expected);
    });

    it.each([
        ["missing field", {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
        }],
        ["extra field", {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "db",
            debug: serverSentinel,
        }],
        ["false API availability", {
            apiAvailable: false,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "db",
        }],
        ["wrong backend", {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: null,
            storeBackend: "memory",
        }],
        ["boolean-like string", {
            apiAvailable: true,
            dbAvailable: "true",
            alertsAvailable: true,
            reason: null,
            storeBackend: "db",
        }],
        ["available with a reason", {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: true,
            reason: "push_identity_unconfigured",
            storeBackend: "db",
        }],
        ["unavailable without a reason", {
            apiAvailable: true,
            dbAvailable: true,
            alertsAvailable: false,
            reason: null,
            storeBackend: "db",
        }],
        ["alerts while database is down", {
            apiAvailable: true,
            dbAvailable: false,
            alertsAvailable: true,
            reason: "push_rules_db_unavailable",
            storeBackend: "db",
        }],
    ])("rejects a malformed or inconsistent availability envelope: %s", async (_name, wire) => {
        server.use(http.get("*/api/push/availability", () => HttpResponse.json(wire)));

        await expectFixedError(
            api.getPushAvailability(),
            "Push availability endpoint unavailable"
        );
    });

    it("does not expose a non-2xx availability body", async () => {
        server.use(http.get("*/api/push/availability", () => HttpResponse.json({
            detail: serverSentinel,
            endpoint,
        }, {status: 503})));

        await expectFixedError(
            api.getPushAvailability(),
            "Push availability endpoint unavailable"
        );
    });
});
