// The PWA hook in Root is the sole registration owner. Alert consumers share
// one bounded wait, using the same 10-second budget as an API request.
const READINESS_TIMEOUT_MS = 10_000;
const READINESS_ERROR = "Alerts are unavailable right now. Close and reopen Alerts to try again.";
let registrationFailed = false;
let pending: Promise<ServiceWorkerRegistration> | null = null;
const failureListeners = new Set<() => void>();

export const reportServiceWorkerRegistrationError = (): void => {
    registrationFailed = true;
    for (const fail of failureListeners) fail();
};

export const reportServiceWorkerRegistered = (): void => {
    registrationFailed = false;
};

export const waitForActiveServiceWorker = (): Promise<ServiceWorkerRegistration> => {
    if (pending) return pending;
    let ready: Promise<ServiceWorkerRegistration>;
    try {
        ready = navigator.serviceWorker.ready;
    } catch {
        // Browser access can fail synchronously; allocate no wait resources yet.
        return Promise.reject(new Error(READINESS_ERROR));
    }
    let timer: ReturnType<typeof setTimeout>;
    let fail: () => void;
    const unavailable = new Promise<never>((_resolve, reject) => {
        fail = () => reject(new Error(READINESS_ERROR));
        failureListeners.add(fail);
        timer = setTimeout(fail, READINESS_TIMEOUT_MS);
    });
    // An already-ready worker remains usable even if an update registration
    // failed. A later ready worker also makes reopening/retrying recoverable.
    const candidates = [ready, unavailable];
    if (registrationFailed) candidates.push(Promise.reject(new Error(READINESS_ERROR)));
    pending = Promise.race(candidates)
        .catch(() => {throw new Error(READINESS_ERROR);})
        .finally(() => {
            clearTimeout(timer);
            failureListeners.delete(fail);
            pending = null;
        });
    return pending;
};
