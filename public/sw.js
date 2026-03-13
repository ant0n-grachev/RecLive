const DEFAULT_TITLE = "RecLive Alert";
const DEFAULT_BODY = "Occupancy update available.";
const DEFAULT_URL = "/";

self.addEventListener("install", (event) => {
    event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
    event.waitUntil(self.clients.claim());
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
