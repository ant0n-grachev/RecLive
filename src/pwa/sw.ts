/// <reference lib="webworker" />

import {clientsClaim} from "workbox-core";
import {ExpirationPlugin} from "workbox-expiration";
import {
    cleanupOutdatedCaches,
    createHandlerBoundToURL,
    precacheAndRoute,
    type PrecacheEntry,
} from "workbox-precaching";
import {NavigationRoute, registerRoute} from "workbox-routing";
import {CacheFirst} from "workbox-strategies";
import {runtimeAssetCacheKind} from "./cachePolicy";
import {openOrFocusSameOrigin} from "./notificationTarget";
import {safePushPayload} from "./pushPayload";

declare const self: ServiceWorkerGlobalScope & typeof globalThis & {
    __WB_MANIFEST: Array<PrecacheEntry | string>;
};

precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();
clientsClaim();

self.addEventListener("message", (event) => {
    if (event.data?.type === "SKIP_WAITING") void self.skipWaiting();
});

const appShellHandler = createHandlerBoundToURL("/index.html");
registerRoute(new NavigationRoute(appShellHandler, {denylist: [/^\/api\//]}));

registerRoute(
    ({request}) => runtimeAssetCacheKind(request, self.location.origin) === "image",
    new CacheFirst({
        cacheName: "reclive-images",
        plugins: [new ExpirationPlugin({
            maxEntries: 80,
            maxAgeSeconds: 60 * 60 * 24 * 30,
        })],
    }),
);

registerRoute(
    ({request}) => runtimeAssetCacheKind(request, self.location.origin) === "font",
    new CacheFirst({
        cacheName: "reclive-fonts",
        plugins: [new ExpirationPlugin({
            maxEntries: 20,
            maxAgeSeconds: 60 * 60 * 24 * 365,
        })],
    }),
);

self.addEventListener("push", (event) => {
    const payload = safePushPayload(event.data);
    event.waitUntil(self.registration.showNotification(payload.title, {
        body: payload.body,
        icon: "/icons/icon-192.png",
        badge: "/icons/icon-192.png",
        data: {url: payload.url},
    }));
});

self.addEventListener("notificationclick", (event) => {
    event.notification.close();
    event.waitUntil(openOrFocusSameOrigin(event.notification.data?.url, {
        origin: self.location.origin,
        matchAll: (options) => self.clients.matchAll(options),
        openWindow: (url) => self.clients.openWindow(url),
    }));
});
