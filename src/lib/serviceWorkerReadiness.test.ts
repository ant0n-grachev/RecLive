import {afterEach, beforeEach, expect, it, vi} from "vitest";

const descriptor = Object.getOwnPropertyDescriptor(navigator, "serviceWorker");
const installReady = (ready: Promise<ServiceWorkerRegistration>) => {
    Object.defineProperty(navigator, "serviceWorker", {configurable: true, value: {ready}});
};
const worker = {active: {}} as ServiceWorkerRegistration;

beforeEach(() => {vi.resetModules(); vi.useFakeTimers();});
afterEach(() => {
    if (descriptor) Object.defineProperty(navigator, "serviceWorker", descriptor);
    else Reflect.deleteProperty(navigator, "serviceWorker");
    vi.useRealTimers();
});

it("propagates registration failure to all waiters without raw errors, then recovers on registration", async () => {
    const readiness = await import("./serviceWorkerReadiness");
    let resolve!: (value: ServiceWorkerRegistration) => void;
    installReady(new Promise((done) => {resolve = done;}));
    const first = readiness.waitForActiveServiceWorker();
    const second = readiness.waitForActiveServiceWorker();
    expect(vi.getTimerCount()).toBe(1);
    const results = Promise.allSettled([first, second]);
    readiness.reportServiceWorkerRegistrationError();
    for (const result of await results) {
        expect(result.status).toBe("rejected");
        if (result.status === "rejected") expect(result.reason.message).toBe(
            "Alerts are unavailable right now. Close and reopen Alerts to try again.");
    }
    expect(vi.getTimerCount()).toBe(0);
    await expect(readiness.waitForActiveServiceWorker()).rejects.toThrow("Alerts are unavailable");
    readiness.reportServiceWorkerRegistered();
    const recovered = readiness.waitForActiveServiceWorker();
    resolve(worker);
    await expect(recovered).resolves.toBe(worker);
    expect(vi.getTimerCount()).toBe(0);
});

it("uses an already-active worker even after an update registration error", async () => {
    const readiness = await import("./serviceWorkerReadiness");
    installReady(Promise.resolve(worker));
    readiness.reportServiceWorkerRegistrationError();
    await expect(readiness.waitForActiveServiceWorker()).resolves.toBe(worker);
    expect(vi.getTimerCount()).toBe(0);
});

it("times out shared readiness and allows reopening once a worker becomes ready", async () => {
    const readiness = await import("./serviceWorkerReadiness");
    let resolve!: (value: ServiceWorkerRegistration) => void;
    installReady(new Promise((done) => {resolve = done;}));
    const failed = expect(readiness.waitForActiveServiceWorker()).rejects.toThrow("Alerts are unavailable");
    await vi.advanceTimersByTimeAsync(10_000);
    await failed;
    expect(vi.getTimerCount()).toBe(0);
    resolve(worker);
    await expect(readiness.waitForActiveServiceWorker()).resolves.toBe(worker);
    expect(vi.getTimerCount()).toBe(0);
});

it("sanitizes a throwing readiness getter without leaving timers or future rejections", async () => {
    const readiness = await import("./serviceWorkerReadiness");
    const serviceWorker = {};
    Object.defineProperty(serviceWorker, "ready", {get: () => {throw new Error("synthetic browser detail");}});
    Object.defineProperty(navigator, "serviceWorker", {configurable: true, value: serviceWorker});
    let safeFailure = false;
    try {
        await readiness.waitForActiveServiceWorker();
    } catch (error) {
        safeFailure = error instanceof Error && error.message ===
            "Alerts are unavailable right now. Close and reopen Alerts to try again.";
    }
    try {
        expect(vi.getTimerCount()).toBe(0);
        expect(safeFailure).toBe(true);
    } finally {
        vi.clearAllTimers();
    }
});
