import {afterAll, beforeAll, describe, expect, it, vi} from "vitest";

const workbox = vi.hoisted(() => ({
    registerRoute: vi.fn(),
    precacheAndRoute: vi.fn(),
    cleanupOutdatedCaches: vi.fn(),
    clientsClaim: vi.fn(),
}));

vi.mock("workbox-core", () => ({clientsClaim: workbox.clientsClaim}));
vi.mock("workbox-expiration", () => ({
    ExpirationPlugin: class ExpirationPlugin {},
}));
vi.mock("workbox-precaching", () => ({
    createHandlerBoundToURL: vi.fn(() => vi.fn()),
    precacheAndRoute: workbox.precacheAndRoute,
    cleanupOutdatedCaches: workbox.cleanupOutdatedCaches,
}));
vi.mock("workbox-routing", () => ({
    NavigationRoute: class NavigationRoute {},
    registerRoute: workbox.registerRoute,
}));
vi.mock("workbox-strategies", () => ({
    CacheFirst: class CacheFirst {},
}));

type Matcher = (input: {request: {url: string; destination: string}}) => boolean;
type WorkerListener = (event: never) => void;

const workerListeners = new Map<string, WorkerListener>();
const skipWaiting = vi.fn(async () => undefined);
const showNotification = vi.fn(async () => undefined);
const matchAll = vi.fn(async () => []);
const openWindow = vi.fn(async () => null);
const originalDescriptors = new Map<PropertyKey, PropertyDescriptor | undefined>();
let precacheManifest: unknown;
let cleanupCallCount = 0;
let clientsClaimCallCount = 0;
let registeredRoutes: unknown[] = [];

const installWorkerGlobal = (property: PropertyKey, value: unknown) => {
    originalDescriptors.set(property, Object.getOwnPropertyDescriptor(self, property));
    Object.defineProperty(self, property, {configurable: true, value});
};

beforeAll(async () => {
    installWorkerGlobal("__WB_MANIFEST", [{url: "/index.html", revision: "test"}]);
    installWorkerGlobal("skipWaiting", skipWaiting);
    installWorkerGlobal("registration", {showNotification});
    installWorkerGlobal("clients", {matchAll, openWindow});
    installWorkerGlobal("addEventListener", vi.fn((type: string, listener: WorkerListener) => {
        workerListeners.set(type, listener);
    }));

    await import("./sw");

    precacheManifest = workbox.precacheAndRoute.mock.calls[0]?.[0];
    cleanupCallCount = workbox.cleanupOutdatedCaches.mock.calls.length;
    clientsClaimCallCount = workbox.clientsClaim.mock.calls.length;
    registeredRoutes = workbox.registerRoute.mock.calls.map(([candidate]) => candidate);
});

afterAll(() => {
    for (const [property, descriptor] of originalDescriptors) {
        if (descriptor) Object.defineProperty(self, property, descriptor);
        else Reflect.deleteProperty(self, property);
    }
});

describe("service worker", () => {
    it("precaches the injected manifest and claims clients", () => {
        expect(precacheManifest).toEqual([{url: "/index.html", revision: "test"}]);
        expect(cleanupCallCount).toBe(1);
        expect(clientsClaimCallCount).toBe(1);
    });

    it("registers same-origin non-API map/image and font policies", () => {
        const matchers = registeredRoutes
            .filter((candidate): candidate is Matcher => typeof candidate === "function");

        expect(matchers).toHaveLength(2);
        const matches = (request: {url: string; destination: string}) => (
            matchers.some((matcher) => matcher({request}))
        );
        const origin = self.location.origin;

        expect(matches({url: `${origin}/floor-maps/nick.png`, destination: "image"})).toBe(true);
        expect(matches({url: `${origin}/assets/app.woff2`, destination: "font"})).toBe(true);
        expect(matches({url: `${origin}/api/floor-map.png`, destination: "image"})).toBe(false);
        expect(matches({url: `${origin}/api/app.woff2`, destination: "font"})).toBe(false);
        expect(matches({url: "https://cdn.example/nick.png", destination: "image"})).toBe(false);
        expect(matches({url: `${origin}/api/live-counts`, destination: ""})).toBe(false);
    });

    it("activates a waiting worker only for the update message", async () => {
        const listener = workerListeners.get("message");
        expect(listener).toBeDefined();

        listener?.({data: {type: "OTHER"}} as never);
        expect(skipWaiting).not.toHaveBeenCalled();

        listener?.({data: {type: "SKIP_WAITING"}} as never);
        await Promise.resolve();
        expect(skipWaiting).toHaveBeenCalledOnce();
    });

    it("shows bounded push data with app identity without collapsing distinct alerts", async () => {
        const waitUntil = vi.fn((promise: Promise<unknown>) => promise);
        const listener = workerListeners.get("push");
        expect(listener).toBeDefined();

        listener?.({
            data: {json: () => ({title: "x".repeat(81), body: 42, url: "https://evil.example/x"})},
            waitUntil,
        } as never);
        await waitUntil.mock.calls[0]?.[0];

        expect(showNotification).toHaveBeenCalledWith("x".repeat(80), {
            body: "Your occupancy alert is ready.",
            icon: "/icons/icon-192.png",
            badge: "/icons/icon-192.png",
            data: {url: "/"},
        });
    });

    it("closes notifications and opens only a same-origin target", async () => {
        const close = vi.fn();
        const waitUntil = vi.fn((promise: Promise<unknown>) => promise);
        const listener = workerListeners.get("notificationclick");
        expect(listener).toBeDefined();

        listener?.({
            notification: {close, data: {url: "/\\evil.example/x"}},
            waitUntil,
        } as never);
        await waitUntil.mock.calls[0]?.[0];

        expect(close).toHaveBeenCalledOnce();
        expect(openWindow).toHaveBeenCalledWith(`${self.location.origin}/`);
    });
});
