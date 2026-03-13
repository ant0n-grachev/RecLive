const DEFAULT_TITLE = "RecLive Alert";
const DEFAULT_BODY = "Occupancy update available.";
const DEFAULT_URL = "/";
const APP_CACHE_NAME = "reclive-app-v1";
const APP_SHELL_URLS = [
    "/manifest.webmanifest",
    "/icons/icon-192.png",
    "/icons/icon-512.png",
];

const cacheUrl = async (cache, url) => {
    try {
        const response = await fetch(url, {cache: "reload"});
        if (!response.ok) return null;
        await cache.put(url, response.clone());
        return response;
    } catch {
        return null;
    }
};

const cacheAppShell = async () => {
    const cache = await self.caches.open(APP_CACHE_NAME);
    const shellResponse = await cacheUrl(cache, DEFAULT_URL);

    await Promise.all(
        APP_SHELL_URLS.map(async (url) => {
            await cacheUrl(cache, url);
        })
    );

    if (!shellResponse) return;

    const html = await shellResponse.clone().text();
    const assetUrls = new Set(
        [...html.matchAll(/(?:href|src)="([^"]+)"/g)]
            .map((match) => new URL(match[1], self.location.origin))
            .filter((url) => url.origin === self.location.origin && url.pathname.startsWith("/assets/"))
            .map((url) => url.href)
    );

    await Promise.all([...assetUrls].map((url) => cacheUrl(cache, url)));
};

const deleteOldCaches = async () => {
    const keys = await self.caches.keys();
    await Promise.all(
        keys
            .filter((key) => key !== APP_CACHE_NAME)
            .map((key) => self.caches.delete(key))
    );
};

self.addEventListener("install", (event) => {
    event.waitUntil(cacheAppShell().then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
    event.waitUntil(deleteOldCaches().then(() => self.clients.claim()));
});

self.addEventListener("fetch", (event) => {
    const {request} = event;
    if (request.method !== "GET") return;

    const url = new URL(request.url);
    if (url.origin !== self.location.origin || url.pathname.startsWith("/api/")) {
        return;
    }

    if (request.mode === "navigate") {
        event.respondWith(
            fetch(request)
                .then(async (response) => {
                    const cache = await self.caches.open(APP_CACHE_NAME);
                    cache.put(request, response.clone());
                    return response;
                })
                .catch(async () => (
                    await self.caches.match(request)
                    ?? await self.caches.match(DEFAULT_URL)
                    ?? Response.error()
                ))
        );
        return;
    }

    const cacheableDestinations = new Set(["font", "image", "manifest", "script", "style"]);
    if (!cacheableDestinations.has(request.destination)) return;

    event.respondWith(
        self.caches.match(request).then((cached) => {
            if (cached) return cached;

            return fetch(request).then(async (response) => {
                if (response.ok) {
                    const cache = await self.caches.open(APP_CACHE_NAME);
                    cache.put(request, response.clone());
                }
                return response;
            });
        })
    );
});

self.addEventListener("push", (event) => {
    let payload = {};

    if (event.data) {
        try {
            payload = event.data.json();
        } catch {
            try {
                payload = {body: event.data.text()};
            } catch {
                payload = {};
            }
        }
    }

    const title = payload.title || DEFAULT_TITLE;
    const body = payload.body || DEFAULT_BODY;
    const url = payload.url || DEFAULT_URL;

    event.waitUntil(
        self.registration.showNotification(title, {
            body,
            icon: "/icons/icon-192.png",
            badge: "/icons/icon-192.png",
            data: {url},
        })
    );
});

self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    const targetUrl = event.notification?.data?.url || DEFAULT_URL;

    event.waitUntil(
        self.clients.matchAll({type: "window", includeUncontrolled: true}).then(async (clients) => {
            const destination = new URL(targetUrl, self.location.origin).href;

            for (const client of clients) {
                try {
                    const clientOrigin = new URL(client.url).origin;
                    if (clientOrigin !== self.location.origin) continue;

                    if ("navigate" in client) {
                        await client.navigate(destination);
                    }
                    await client.focus();
                    return;
                } catch {
                    // Fall back to opening a fresh window when client navigation fails.
                }
            }

            await self.clients.openWindow(destination);
        })
    );
});
