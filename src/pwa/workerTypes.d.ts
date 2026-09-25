/// <reference lib="webworker" />

import type {PrecacheEntry} from "workbox-precaching";

export {};

declare global {
    interface ServiceWorkerGlobalScope {
        __WB_MANIFEST: Array<PrecacheEntry | string>;
    }
}
