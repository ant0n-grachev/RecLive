# RecLive PWA and Accessibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Ship an installable, offline-capable RecLive dashboard whose live-state changes, heat-map zones, alert feedback, and PWA update flow are operable and understandable with keyboard and assistive technology.

**Architecture:** Vite PWA's inject-manifest build emits a typed service worker. Pure push and notification-target helpers keep worker behaviour testable without browser globals. React owns update, live-status, and heat-map dialog state; it announces meaningful changes through one concrete live-region component. The PWA caches only app-shell/static assets and safe same-origin images, never live API responses.

**Tech Stack:** React, TypeScript, Vite, vite-plugin-pwa/Workbox, Material UI, Vitest/Testing Library, Playwright, axe-core.

## Global constraints

- Preserve current routes, occupancy data contracts, refresh semantics, and the Phase 3 `OccupancySummary` shape.
- The master plan's phase-level commit policy is authoritative: task commit snippets describe staging/review scope only; make the single Phase 8 source commit after Task 5.
- Cache neither `/api/` responses nor live count/forecast responses. The worker must only open or focus same-origin windows.
- Vite mode `local-debug` is the sole production-build debug opt-in; it adds no third public `VITE_` variable. Debug controls are enabled when `import.meta.env.DEV` is true, or when `import.meta.env.MODE === "local-debug"` **and** the hostname is exactly `localhost` or `127.0.0.1`. A normal production build served on localhost remains non-debug.
- Do not log push subscription endpoints, raw notification payloads, or client URLs.
- Use the existing `tests/backend/`, colocated `src/**/*.test.ts(x)`, and `tests/e2e/` conventions. This phase adds no backend tests.
- Production build tests must inspect emitted behaviour/literals and asset references, not source function names or formatting that minification can remove.

## File map

- Modify: `package.json`, `package-lock.json`, `vite.config.ts`, `src/vite-env.d.ts`, `src/main.tsx`, `src/app/App.tsx`, `src/app/components/AlertsPanel.tsx`, `src/app/theme.tsx`, `src/facilities/FloorHeatMapCard.tsx`, `src/facilities/CrowdAlertSubscriptionCard.tsx`, `index.html`, `tests/e2e/route-smoke.spec.ts`.
- Delete: `public/sw.js` after the generated worker is wired.
- Create: `src/app/debugOverrides.ts`, `src/app/debugOverrides.test.ts`, `src/pwa/cachePolicy.ts`, `src/pwa/cachePolicy.test.ts`, `src/pwa/pushPayload.ts`, `src/pwa/pushPayload.test.ts`, `src/pwa/notificationTarget.ts`, `src/pwa/notificationTarget.test.ts`, `src/pwa/sw.ts`, `src/pwa/sw.test.ts`, `src/pwa/workerTypes.d.ts`, `src/facilities/PwaUpdatePrompt.tsx`, `src/facilities/PwaUpdatePrompt.test.tsx`, `src/facilities/LiveStatusAnnouncer.tsx`, `src/facilities/LiveStatusAnnouncer.test.tsx`, `src/facilities/FloorHeatMapCard.test.tsx`, `tests/e2e/support/apiMocks.ts`, `tests/e2e/pwa-build.spec.ts`, `tests/e2e/accessibility.spec.ts`.

## Task 1: Gate development overrides with an explicit local production flag

**Files:**
- Create: `src/app/debugOverrides.ts`
- Create: `src/app/debugOverrides.test.ts`
- Modify: `src/app/App.tsx`

1. Write the failing tests in `src/app/debugOverrides.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import {
  debugControlsEnabled,
  loadDebugOverrides,
} from "./debugOverrides";

describe("debug overrides", () => {
  it("does not read query or persisted overrides while controls are disabled", () => {
    const getItem = vi.fn(() => "true");
    expect(loadDebugOverrides(false, { search: "?debugNow=2026-08-31T09:00&overrideClosure=1", storage: { getItem } }))
      .toEqual({ closureOverrideEnabled: false, debugNowValue: null });
    expect(getItem).not.toHaveBeenCalled();
  });

  it("requires the explicit local-debug build mode outside development", () => {
    const production = { DEV: false, MODE: "production" };
    expect(debugControlsEnabled(production, "localhost")).toBe(false);
    expect(debugControlsEnabled(production, "127.0.0.1")).toBe(false);
    expect(debugControlsEnabled({ DEV: false, MODE: "local-debug" }, "localhost")).toBe(true);
    expect(debugControlsEnabled({ DEV: false, MODE: "local-debug" }, "recwell.wisc.edu")).toBe(false);
    expect(debugControlsEnabled({ DEV: true, MODE: "development" }, "recwell.wisc.edu")).toBe(true);
  });

  it("reads only the current closure and debug-clock contracts when enabled", () => {
    const getItem = vi.fn((key: string) => key === "reclive:closureOverride" ? "true" : "2026-08-31T08:00");
    expect(loadDebugOverrides(true, { search: "?debugNow=2026-08-31T09:00", storage: { getItem } }))
      .toEqual({ closureOverrideEnabled: true, debugNowValue: "2026-08-31T09:00" });
  });
});
```

2. Run `npm run test:run -- src/app/debugOverrides.test.ts`. Expected RED result: Vitest cannot resolve `./debugOverrides`.

3. Add the exact helper contract to `src/app/debugOverrides.ts`:

```ts
export interface DebugOverrideInput {
  closureOverrideEnabled: boolean;
  debugNowValue: string | null;
}

export interface DebugOverrideSources {
  search: string;
  storage: Pick<Storage, "getItem">;
}

const LOCAL_HOSTS = new Set(["localhost", "127.0.0.1"]);
const CLOSURE_STORAGE_KEY = "reclive:closureOverride";
const DEBUG_NOW_STORAGE_KEY = "reclive:debugNow";

export function debugControlsEnabled(
  env: Pick<ImportMetaEnv, "DEV" | "MODE">,
  hostname: string,
): boolean {
  return env.DEV || (env.MODE === "local-debug" && LOCAL_HOSTS.has(hostname));
}

export function loadDebugOverrides(enabled: boolean, sources?: DebugOverrideSources): DebugOverrideInput {
  if (!enabled) return { closureOverrideEnabled: false, debugNowValue: null };
  try {
    const search = sources?.search ?? window.location.search;
    const storage = sources?.storage ?? window.localStorage;
    const params = new URLSearchParams(search);
    const closureQuery = params.get("overrideClosure") ?? params.get("debugClosure");
    const storedClosure = storage.getItem(CLOSURE_STORAGE_KEY);
    const debugNowQuery = params.get("debugNow");
    return {
      closureOverrideEnabled: closureQuery === null
        ? storedClosure === "true"
        : closureQuery !== "0" && closureQuery.toLowerCase() !== "false",
      debugNowValue: debugNowQuery ?? storage.getItem(DEBUG_NOW_STORAGE_KEY),
    };
  } catch {
    return { closureOverrideEnabled: false, debugNowValue: null };
  }
}
```

4. In `src/app/App.tsx`, calculate `const debugEnabled = debugControlsEnabled(import.meta.env, window.location.hostname);` and initialize `const [initialDebugOverrides] = useState(() => loadDebugOverrides(debugEnabled));` before the debug states. Initialize closure from `initialDebugOverrides.closureOverrideEnabled`, initialize the clock only by passing `initialDebugOverrides.debugNowValue` through the existing `parseDebugNowMs`, and make both storage writers return immediately when `debugEnabled` is false. Replace **both** current `DEV || localhost` global-debug effect gates—including `recliveDebugDashboardState`—with `debugEnabled`; when false, delete every `window.reclive*` debug function and never read the persisted keys or query parameters. Prediction override remains initially false and can change only through globals installed behind this same gate. Remove the old `getInitialClosureOverride` and `getInitialDebugNowMs` window-reading functions after their callers migrate.

5. Run `npm run test:run -- src/app/debugOverrides.test.ts`, `npm run lint`, and `npm run build`. Expected GREEN: all three tests pass, lint succeeds, and the normal production build keeps every debug input disabled even on localhost.

6. Commit:

```bash
git add src/app/debugOverrides.ts src/app/debugOverrides.test.ts src/app/App.tsx
git commit -m "feat: gate local dashboard debug overrides"
```

## Task 2: Replace the hand-written worker with a typed, testable inject-manifest worker

**Files:**
- Modify: `package.json`, `package-lock.json`, `vite.config.ts`, `src/vite-env.d.ts`
- Create: `src/pwa/cachePolicy.ts`, `src/pwa/cachePolicy.test.ts`, `src/pwa/pushPayload.ts`, `src/pwa/pushPayload.test.ts`, `src/pwa/notificationTarget.ts`, `src/pwa/notificationTarget.test.ts`, `src/pwa/sw.ts`, `src/pwa/sw.test.ts`, `src/pwa/workerTypes.d.ts`, `tests/e2e/pwa-build.spec.ts`
- Delete: `public/sw.js`

1. Add the required build/test dependencies with an unversioned install command so npm writes the resolved versions to `package-lock.json`:

```bash
npm install -D vite-plugin-pwa workbox-core workbox-expiration workbox-precaching workbox-routing workbox-strategies
```

2. Write failing pure-helper tests in `src/pwa/cachePolicy.test.ts`, `src/pwa/pushPayload.test.ts`, and `src/pwa/notificationTarget.test.ts`. The cache-policy test proves behaviour directly instead of inferring it from minified worker text:

```ts
import { describe, expect, it } from "vitest";
import { runtimeAssetCacheKind } from "./cachePolicy";

describe("runtimeAssetCacheKind", () => {
  const origin = "https://dashboard.example";

  it("classifies only same-origin non-API maps, images, and fonts", () => {
    expect(runtimeAssetCacheKind({ url: `${origin}/floor-maps/nick.png`, destination: "image" }, origin)).toBe("image");
    expect(runtimeAssetCacheKind({ url: `${origin}/assets/app.woff2`, destination: "font" }, origin)).toBe("font");
    expect(runtimeAssetCacheKind({ url: `${origin}/api/floor-map.png`, destination: "image" }, origin)).toBeNull();
    expect(runtimeAssetCacheKind({ url: "https://cdn.example/nick.png", destination: "image" }, origin)).toBeNull();
    expect(runtimeAssetCacheKind({ url: `${origin}/api/live-counts`, destination: "" }, origin)).toBeNull();
  });
});
```

```ts
import { describe, expect, it } from "vitest";
import { safePushPayload } from "./pushPayload";

describe("safePushPayload", () => {
  it("uses bounded fallback content for missing or malformed data", () => {
    expect(safePushPayload(null)).toEqual({ title: "RecLive alert", body: "Your occupancy alert is ready.", url: "/" });
    expect(safePushPayload({ json: () => { throw new Error("bad json"); } })).toEqual({ title: "RecLive alert", body: "Your occupancy alert is ready.", url: "/" });
  });

  it("normalizes types, lengths, and an unsafe target", () => {
    const payload = safePushPayload({ json: () => ({ title: "x".repeat(81), body: "y".repeat(241), url: "https://other.example/a" }) });
    expect(payload).toEqual({ title: "x".repeat(80), body: "y".repeat(240), url: "/" });
  });
});
```

```ts
import { describe, expect, it, vi } from "vitest";
import { openOrFocusSameOrigin } from "./notificationTarget";

describe("openOrFocusSameOrigin", () => {
  it("focuses an existing same-origin window and never opens a cross-origin target", async () => {
    const focus = vi.fn(async () => undefined);
    const navigate = vi.fn(async () => undefined);
    const openWindow = vi.fn(async () => undefined);
    const result = await openOrFocusSameOrigin("https://evil.example/x", {
      origin: "https://dashboard.example",
      matchAll: vi.fn(async () => [{ url: "https://dashboard.example/old", navigate, focus }]),
      openWindow,
    });
    expect(result).toBe("focused");
    expect(navigate).toHaveBeenCalledWith("https://dashboard.example/");
    expect(openWindow).not.toHaveBeenCalled();
  });

  it("opens the normalized path only when no same-origin client exists", async () => {
    const openWindow = vi.fn(async () => undefined);
    await openOrFocusSameOrigin("/nick?floor=1", {
      origin: "https://dashboard.example",
      matchAll: vi.fn(async () => [{ url: "https://other.example/", navigate: vi.fn(), focus: vi.fn() }]),
      openWindow,
    });
    expect(openWindow).toHaveBeenCalledWith("https://dashboard.example/nick?floor=1");
  });
});
```

3. Run `npm run test:run -- src/pwa/cachePolicy.test.ts src/pwa/pushPayload.test.ts src/pwa/notificationTarget.test.ts`. Expected RED result: none of the three helper modules exists.

4. Implement `src/pwa/cachePolicy.ts`, `src/pwa/pushPayload.ts`, and `src/pwa/notificationTarget.ts` with these complete interfaces and bounds:

```ts
export interface RuntimeRequestLike { url: string; destination: string; }

export type RuntimeAssetCacheKind = "image" | "font";

export function runtimeAssetCacheKind(request: RuntimeRequestLike, origin: string): RuntimeAssetCacheKind | null {
  try {
    const url = new URL(request.url);
    if (url.origin !== origin || url.pathname.startsWith("/api/")) return null;
    if (request.destination === "image") return "image";
    if (request.destination === "font") return "font";
    return null;
  } catch {
    return null;
  }
}
```

```ts
export interface PushMessageDataLike { json(): unknown; }
export interface SafePushPayload { title: string; body: string; url: string; }
const fallback: SafePushPayload = { title: "RecLive alert", body: "Your occupancy alert is ready.", url: "/" };
const stringAtMost = (value: unknown, maximum: number, fallbackValue: string) =>
  typeof value === "string" && value.length > 0 ? value.slice(0, maximum) : fallbackValue;

export function safePushPayload(data: PushMessageDataLike | null): SafePushPayload {
  try {
    const value: unknown = data?.json();
    if (typeof value !== "object" || value === null) return fallback;
    const record = value as Record<string, unknown>;
    const rawUrl = stringAtMost(record.url, 2048, fallback.url);
    const url = rawUrl.startsWith("/") && !rawUrl.startsWith("//") ? rawUrl : fallback.url;
    return { title: stringAtMost(record.title, 80, fallback.title), body: stringAtMost(record.body, 240, fallback.body), url };
  } catch { return fallback; }
}
```

```ts
export interface NotificationWindowClient { url: string; navigate(url: string): Promise<unknown>; focus(): Promise<unknown>; }
export interface NotificationClientScope {
  origin: string;
  matchAll(options: { type: "window"; includeUncontrolled: boolean }): Promise<readonly NotificationWindowClient[]>;
  openWindow(url: string): Promise<unknown>;
}

export async function openOrFocusSameOrigin(rawTarget: unknown, scope: NotificationClientScope): Promise<"focused" | "opened"> {
  const fallback = new URL("/", `${scope.origin}/`);
  let safe = fallback;
  if (typeof rawTarget === "string" && rawTarget.startsWith("/") && !rawTarget.startsWith("//")) {
    try {
      const candidate = new URL(rawTarget, fallback);
      if (candidate.origin === scope.origin) safe = candidate;
    } catch { /* retain the same-origin root fallback */ }
  }
  const clients = await scope.matchAll({ type: "window", includeUncontrolled: true });
  const existing = clients.find((client) => {
    try { return new URL(client.url).origin === scope.origin; } catch { return false; }
  });
  if (existing) {
    try { await existing.navigate(safe.href); } catch { /* focus the existing client even when navigation is rejected */ }
    await existing.focus();
    return "focused";
  }
  await scope.openWindow(safe.href);
  return "opened";
}
```

5. Configure `VitePWA` in `vite.config.ts` with `strategies: "injectManifest"`, `srcDir: "src/pwa"`, `filename: "sw.ts"`, `registerType: "prompt"`, `injectManifest: { globPatterns: ["**/*.{js,css,html,svg,png,webp,woff2}"] }`, and a manifest whose `start_url` is `/`, `display` is `standalone`, and icons point to existing public icon assets. Add both references at the top of `src/vite-env.d.ts`:

```ts
/// <reference types="vite/client" />
/// <reference types="vite-plugin-pwa/client" />
```

6. Add `src/pwa/workerTypes.d.ts` and `src/pwa/sw.ts`:

```ts
/// <reference lib="webworker" />
export {};
declare const self: ServiceWorkerGlobalScope & typeof globalThis & { __WB_MANIFEST: Array<unknown> };
```

```ts
/// <reference lib="webworker" />
import { clientsClaim } from "workbox-core";
import { ExpirationPlugin } from "workbox-expiration";
import { precacheAndRoute, cleanupOutdatedCaches } from "workbox-precaching";
import { createHandlerBoundToURL, registerRoute, NavigationRoute } from "workbox-routing";
import { CacheFirst } from "workbox-strategies";
import { runtimeAssetCacheKind } from "./cachePolicy";
import { openOrFocusSameOrigin } from "./notificationTarget";
import { safePushPayload } from "./pushPayload";

declare const self: ServiceWorkerGlobalScope & typeof globalThis & { __WB_MANIFEST: Array<unknown> };
precacheAndRoute(self.__WB_MANIFEST);
cleanupOutdatedCaches();
clientsClaim();
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") void self.skipWaiting();
});
const appShellHandler = createHandlerBoundToURL("/index.html");
registerRoute(new NavigationRoute(appShellHandler, { denylist: [/^\/api\//] }));
registerRoute(({ request }) => runtimeAssetCacheKind(request, self.location.origin) === "image", new CacheFirst({ cacheName: "reclive-images", plugins: [new ExpirationPlugin({ maxEntries: 80, maxAgeSeconds: 60 * 60 * 24 * 30 })] }));
registerRoute(({ request }) => runtimeAssetCacheKind(request, self.location.origin) === "font", new CacheFirst({ cacheName: "reclive-fonts", plugins: [new ExpirationPlugin({ maxEntries: 20, maxAgeSeconds: 60 * 60 * 24 * 365 })] }));
self.addEventListener("push", (event) => {
  const payload = safePushPayload(event.data);
  event.waitUntil(self.registration.showNotification(payload.title, { body: payload.body, data: { url: payload.url }, tag: "reclive-occupancy" }));
});
self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  event.waitUntil(openOrFocusSameOrigin(event.notification.data?.url, {
    origin: self.location.origin,
    matchAll: (options) => self.clients.matchAll(options),
    openWindow: (url) => self.clients.openWindow(url),
  }));
});
```

7. Add `src/pwa/sw.test.ts` to prove that the worker actually registers the tested cache predicate, not merely that a disconnected helper works:

```ts
import { beforeAll, describe, expect, it, vi } from "vitest";

const workbox = vi.hoisted(() => ({
  registerRoute: vi.fn(),
  precacheAndRoute: vi.fn(),
  cleanupOutdatedCaches: vi.fn(),
  clientsClaim: vi.fn(),
}));

vi.mock("workbox-core", () => ({ clientsClaim: workbox.clientsClaim }));
vi.mock("workbox-expiration", () => ({ ExpirationPlugin: class ExpirationPlugin {} }));
vi.mock("workbox-precaching", () => ({
  precacheAndRoute: workbox.precacheAndRoute,
  cleanupOutdatedCaches: workbox.cleanupOutdatedCaches,
}));
vi.mock("workbox-routing", () => ({
  createHandlerBoundToURL: vi.fn(() => vi.fn()),
  NavigationRoute: class NavigationRoute { constructor(public handler: unknown, public options: unknown) {} },
  registerRoute: workbox.registerRoute,
}));
vi.mock("workbox-strategies", () => ({ CacheFirst: class CacheFirst {} }));

type Matcher = (input: { request: { url: string; destination: string } }) => boolean;

beforeAll(async () => {
  Object.defineProperty(self, "__WB_MANIFEST", { configurable: true, value: [] });
  await import("./sw");
});

describe("service-worker runtime route", () => {
  it("registers same-origin non-API map/image and font policies", () => {
    const matchers = workbox.registerRoute.mock.calls
      .map(([candidate]) => candidate)
      .filter((candidate): candidate is Matcher => typeof candidate === "function");
    expect(matchers).toHaveLength(2);
    const matches = (request: {url: string; destination: string}) => matchers.some((matcher) => matcher({request}));
    const origin = self.location.origin;
    expect(matches({ url: `${origin}/floor-maps/nick.png`, destination: "image" })).toBe(true);
    expect(matches({ url: `${origin}/assets/app.woff2`, destination: "font" })).toBe(true);
    expect(matches({ url: `${origin}/api/floor-map.png`, destination: "image" })).toBe(false);
    expect(matches({ url: `${origin}/api/app.woff2`, destination: "font" })).toBe(false);
    expect(matches({ url: "https://cdn.example/nick.png", destination: "image" })).toBe(false);
  });
});
```

8. Delete `public/sw.js` and remove the manual registration from `src/main.tsx`; Task 3 owns registering the virtual PWA module. Write `tests/e2e/pwa-build.spec.ts` to build before Playwright starts and assert emitted artifacts without source-name assumptions:

```ts
import { existsSync, readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";

test("production output contains a generated worker with precache and cache policy", () => {
  const workerPath = "dist/sw.js";
  expect(existsSync(workerPath)).toBe(true);
  expect(existsSync("public/sw.js")).toBe(false);
  const worker = readFileSync(workerPath, "utf8");
  expect(worker).not.toContain("__WB_MANIFEST");
  expect(worker).toContain("reclive-images");
  expect(worker).toContain("reclive-fonts");
  expect(worker).toContain("SKIP_WAITING");
});
```

The pure cache-policy test proves its boundary for static floor maps/images and fonts, `sw.test.ts` proves that the worker registers both exact matchers, and the artifact test proves a generated manifest-injected worker is emitted without depending on a minifier's regex spelling. The checked-in floor-map PNGs remain precached by the manifest and are also safely classifiable if requested outside the manifest; APIs and off-origin assets match neither runtime route.

9. Run `npm run test:run -- src/pwa/cachePolicy.test.ts src/pwa/pushPayload.test.ts src/pwa/notificationTarget.test.ts src/pwa/sw.test.ts`, `npm run build`, and `npx playwright test tests/e2e/pwa-build.spec.ts`. Expected GREEN: helper tests pass, the worker registers the tested predicate, API and off-origin inputs are rejected, the build emits a worker whose Workbox manifest has been injected, and artifact assertions pass even under minification.

10. Commit:

```bash
git add package.json package-lock.json vite.config.ts src/vite-env.d.ts src/pwa src/main.tsx tests/e2e/pwa-build.spec.ts
git rm public/sw.js
git commit -m "feat: add generated RecLive service worker"
```

## Task 3: Provide an accessible, typed PWA update prompt

**Files:**
- Create: `src/facilities/PwaUpdatePrompt.tsx`, `src/facilities/PwaUpdatePrompt.test.tsx`
- Modify: `src/main.tsx`
- Modify: `src/lib/api/pushNotifications.ts`
- Modify: `src/lib/api/pushNotifications.test.ts`

1. Write `src/facilities/PwaUpdatePrompt.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, it, vi } from "vitest";
import { PwaUpdatePrompt } from "./PwaUpdatePrompt";

it("announces and applies an available update", async () => {
  const user = userEvent.setup();
  const updateServiceWorker = vi.fn(async () => undefined);
  const onDismiss = vi.fn();
  render(<PwaUpdatePrompt needRefresh offlineReady={false} updateServiceWorker={updateServiceWorker} onDismiss={onDismiss} />);
  expect(screen.getByRole("status")).toHaveTextContent("A new version of RecLive is ready");
  await user.click(screen.getByRole("button", { name: "Update now" }));
  expect(updateServiceWorker).toHaveBeenCalledWith(true);
  expect(onDismiss).toHaveBeenCalledOnce();
});

it("does not render when neither worker state is active", () => {
  const { container } = render(<PwaUpdatePrompt needRefresh={false} offlineReady={false} updateServiceWorker={async () => undefined} onDismiss={() => undefined} />);
  expect(container).toBeEmptyDOMElement();
});
```

Append a push-client regression test that stubs `navigator.serviceWorker.ready` with an active registration, calls `getExistingPushSubscription()`, and asserts a `navigator.serviceWorker.register` spy was not called. This test is RED while `ensureActiveServiceWorker` still performs its fallback registration.

2. Run `npm run test:run -- src/facilities/PwaUpdatePrompt.test.tsx`. Expected RED result: module not found.

3. Implement the exact component interface:

```tsx
export interface PwaUpdatePromptProps {
  needRefresh: boolean;
  offlineReady: boolean;
  updateServiceWorker(reloadPage?: boolean): Promise<void>;
  onDismiss(): void;
}

export function PwaUpdatePrompt({ needRefresh, offlineReady, updateServiceWorker, onDismiss }: PwaUpdatePromptProps) {
  if (!needRefresh && !offlineReady) return null;
  const message = needRefresh ? "A new version of RecLive is ready." : "RecLive is ready to use offline.";
  return <div role="status" aria-live="polite">
    <p>{message}</p>
    {needRefresh && <button type="button" onClick={async () => { await updateServiceWorker(true); onDismiss(); }}>Update now</button>}
    <button type="button" onClick={onDismiss}>Dismiss</button>
  </div>;
}
```

4. In `src/main.tsx`, import `{ useRegisterSW }` from `virtual:pwa-register/react` and `{ PwaUpdatePrompt }` from `./facilities/PwaUpdatePrompt` (the `vite-plugin-pwa/client` reference from Task 2 supplies the virtual-module type). Call the hook inside the existing exported `Root` component, not at module scope, and pass its exact state to the component:

```tsx
const { needRefresh: [needRefresh, setNeedRefresh], offlineReady: [offlineReady, setOfflineReady], updateServiceWorker } = useRegisterSW({ immediate: true });
const dismissPwaPrompt = () => { setNeedRefresh(false); setOfflineReady(false); };
```

Render `<PwaUpdatePrompt needRefresh={needRefresh} offlineReady={offlineReady} updateServiceWorker={updateServiceWorker} onDismiss={dismissPwaPrompt} />` immediately after `<AppRoutes themeMode={themeMode} onThemeModeChange={setThemeMode}/>` inside the existing `BrowserRouter` in `Root`. Delete the manual `navigator.serviceWorker.register("/sw.js")` block at the end of `src/main.tsx`. In `src/lib/api/pushNotifications.ts`, change `ensureActiveServiceWorker` to await `navigator.serviceWorker.ready` and return that registration; remove its fallback `navigator.serviceWorker.register("/sw.js")` call. `useRegisterSW` is then the only registration path, while alert code only consumes the ready registration.

5. Run `npm run test:run -- src/facilities/PwaUpdatePrompt.test.tsx src/lib/api/pushNotifications.test.ts`, `npm run lint`, and `npm run build`. Expected GREEN: update and single-registration tests pass, lint/build succeed, and no unresolved `virtual:pwa-register/react` module remains.

6. Commit:

```bash
git add src/facilities/PwaUpdatePrompt.tsx src/facilities/PwaUpdatePrompt.test.tsx src/main.tsx src/lib/api/pushNotifications.ts src/lib/api/pushNotifications.test.ts
git commit -m "feat: prompt users to apply PWA updates"
```

## Task 4: Make heat-map zones semantic controls with a labelled non-modal dialog

**Files:**
- Modify: `src/facilities/FloorHeatMapCard.tsx`, `src/app/theme.tsx`
- Create: `src/facilities/FloorHeatMapCard.test.tsx`

1. Write the tests using concrete current RecWell data, not shared undeclared fixtures:

```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import FloorHeatMapCard, { zoneAccessibleLabel } from "./FloorHeatMapCard";

const powerHouse = { facilityId: 1186, locationId: 5761, locationName: "Power House", floor: 0, isClosed: false, currentCapacity: 30, maxCapacity: 50, lastUpdated: "2026-08-31T12:00:00Z", fetchedAt: "2026-08-31T12:00:00Z" };
afterEach(() => vi.useRealTimers());

it("describes partial coverage without presenting it as a complete count", () => {
  expect(zoneAccessibleLabel({ id: "power-house", label: "Power House", status: "partial", percent: 60, coverage: 60, count: 30 })).toBe("Power House: 60% occupied from observed capacity; 60% coverage.");
});

it("opens a labelled non-modal dialog from the SVG keyboard control and restores focus", async () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-08-31T12:00:00Z"));
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(<FloorHeatMapCard facilityId={1186} locations={[powerHouse]} occupancyThresholds={null} locationOccupancyThresholds={{}} />);
  await user.click(screen.getByRole("button", { name: /show map/i }));
  const zone = screen.getByRole("button", { name: "Power House: 60% occupied." });
  zone.focus();
  await user.keyboard("{Enter}");
  expect(screen.getByRole("dialog", { name: "Power House details" })).toBeVisible();
  await user.keyboard("{Escape}");
  await vi.runAllTimersAsync();
  expect(zone).toHaveFocus();
});

it("closes from its labelled control and from click-away", async () => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date("2026-08-31T12:00:00Z"));
  const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
  render(<FloorHeatMapCard facilityId={1186} locations={[powerHouse]} occupancyThresholds={null} locationOccupancyThresholds={{}} />);
  await user.click(screen.getByRole("button", { name: /show map/i }));
  const zone = screen.getByRole("button", { name: "Power House: 60% occupied." });
  await user.click(zone);
  await user.click(screen.getByRole("button", { name: "Close Power House details" }));
  await vi.runAllTimersAsync();
  expect(zone).toHaveFocus();
  await user.click(zone);
  await user.click(document.body);
  await vi.runAllTimersAsync();
  expect(screen.queryByRole("dialog", { name: "Power House details" })).not.toBeInTheDocument();
  expect(zone).toHaveFocus();
});
```

2. Run `npm run test:run -- src/facilities/FloorHeatMapCard.test.tsx`. Expected RED result: `zoneAccessibleLabel` is not exported and SVG zones are not accessible buttons.

3. In `src/facilities/FloorHeatMapCard.tsx`, export the precise presentation contract and label formatter:

```ts
import type { OccupancySummary } from "../shared/occupancy/computeOccupancySummary";

export interface HeatmapZonePresentation {
  id: string;
  label: string;
  status: OccupancySummary["status"];
  percent: number | null;
  coverage: number | null;
  count: number | null;
}

export function zoneAccessibleLabel(zone: HeatmapZonePresentation): string {
  if (zone.status === "closed") return `${zone.label}: closed.`;
  if (zone.status === "unknown" || zone.status === "insufficient" || zone.percent === null) return `${zone.label}: live occupancy unavailable.`;
  if (zone.status === "partial") return `${zone.label}: ${zone.percent}% occupied from observed capacity; ${zone.coverage ?? 0}% coverage.`;
  return `${zone.label}: ${zone.percent}% occupied.`;
}
```

4. Replace each interactive transparent SVG polygon with a real keyboard-accessible SVG control. Store the trigger element with the selected presentation, activate on click, Enter, or Space, and give the user agent a dialog relationship:

```tsx
<polygon
  role="button"
  tabIndex={0}
  aria-label={zoneAccessibleLabel(presentation)}
  aria-haspopup="dialog"
  aria-expanded={selectedZone?.presentation.id === presentation.id}
  onClick={(event) => setSelectedZone({ presentation, trigger: event.currentTarget })}
  onKeyDown={(event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    setSelectedZone({ presentation, trigger: event.currentTarget });
  }}
/>
```

Replace the current modal-backed `Popover` with a Material UI `Popper` anchored to `selectedZone.trigger`. Inside it, render a `Paper` with `role="dialog"`, `aria-modal="false"`, `aria-labelledby="heatmap-zone-dialog-title"`, a visible `<Typography id="heatmap-zone-dialog-title">{selectedZone.presentation.label} details</Typography>`, and a button labelled `Close ${selectedZone.presentation.label} details`. Wrap the paper in `ClickAwayListener`. While a zone is selected, a `useEffect` must install a document `keydown` listener that calls the same close function only for Escape and removes that listener in cleanup. The single close function captures the trigger, clears state, and restores trigger focus in `requestAnimationFrame`; Close, click-away, and Escape all call it. Because `Popper` does not hide page siblings or trap focus, the result is genuinely non-modal. The details must use the same presentation status and never invent coverage or capacity values.

5. In `src/app/theme.tsx`, add a `MuiButtonBase` root minimum block size of `44px`, a `MuiIconButton` minimum inline/block size of `44px`, and a `MuiCssBaseline` style override for `polygon[role="button"]:focus-visible` with a high-contrast outline/stroke. Do not shrink a control below those values in feature-level styles. Add a `prefers-reduced-motion: reduce` override that disables non-essential transitions and animations. Keep all normal color semantics as a secondary visual cue, not the only state signal. The mapped zones keep their real polygon hit regions; the browser accessibility test must prove each exposed zone has at least a 24-by-24 CSS-pixel target at the supported mobile viewport, while normal buttons retain the 44-pixel target.

6. Run `npm run test:run -- src/facilities/FloorHeatMapCard.test.tsx`, `npm run lint`, and `npm run build`. Expected GREEN: the direct fixture exercises label wording, keyboard dialog operation, Escape close, focus restoration, lint, and production build.

7. Commit:

```bash
git add src/facilities/FloorHeatMapCard.tsx src/facilities/FloorHeatMapCard.test.tsx src/app/theme.tsx
git commit -m "feat: make occupancy heat map keyboard accessible"
```

## Task 5: Announce live refresh and alert submission outcomes, then prove the page with axe

**Files:**
- Create: `src/facilities/LiveStatusAnnouncer.tsx`, `src/facilities/LiveStatusAnnouncer.test.tsx`, `tests/e2e/support/apiMocks.ts`, `tests/e2e/accessibility.spec.ts`
- Modify: `src/app/App.tsx`, `src/app/components/AlertsPanel.tsx`, `src/facilities/CrowdAlertSubscriptionCard.tsx`, `index.html`, `tests/e2e/route-smoke.spec.ts`

1. Write the concrete component test in `src/facilities/LiveStatusAnnouncer.test.tsx`:

```tsx
import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { LiveStatusAnnouncer } from "./LiveStatusAnnouncer";

it("announces refresh completion after an in-progress state", () => {
  const view = render(<LiveStatusAnnouncer status="refreshing" />);
  expect(screen.getByRole("status")).toHaveTextContent("Refreshing live occupancy");
  view.rerender(<LiveStatusAnnouncer status="updated" />);
  expect(screen.getByRole("status")).toHaveTextContent("Live occupancy updated");
});

it("uses assertive announcement for a refresh failure", () => {
  render(<LiveStatusAnnouncer status="refresh-error" />);
  expect(screen.getByRole("alert")).toHaveTextContent("Live occupancy could not be refreshed");
});
```

2. Run `npm run test:run -- src/facilities/LiveStatusAnnouncer.test.tsx`. Expected RED result: module not found.

3. Create the component with this complete API:

```tsx
import type { CSSProperties } from "react";

export type LiveStatus = "idle" | "refreshing" | "updated" | "refresh-error" | "alert-created";
const messages: Record<Exclude<LiveStatus, "idle">, string> = {
  refreshing: "Refreshing live occupancy.",
  updated: "Live occupancy updated.",
  "refresh-error": "Live occupancy could not be refreshed. Showing the last available data.",
  "alert-created": "Occupancy alert created.",
};

const visuallyHidden: CSSProperties = {
  position: "absolute",
  width: 1,
  height: 1,
  padding: 0,
  margin: -1,
  overflow: "hidden",
  clip: "rect(0 0 0 0)",
  whiteSpace: "nowrap",
  border: 0,
};

export function LiveStatusAnnouncer({ status }: { status: LiveStatus }) {
  if (status === "idle") return null;
  const assertive = status === "refresh-error";
  return <div role={assertive ? "alert" : "status"} aria-live={assertive ? "assertive" : "polite"} style={visuallyHidden}>{messages[status]}</div>;
}
```

4. In `src/app/App.tsx`, hold `const [liveStatus, setLiveStatus] = useState<LiveStatus>("idle");`. In the manual-refresh handler, set `"refreshing"` immediately before `triggerRefresh`. Use a `useRef` of the previous `isLoading` value and an effect that changes `"refreshing"` to `"refresh-error"` when the transition from loading to idle has an error, otherwise to `"updated"`; do not announce an initial query as a refresh. Render `<LiveStatusAnnouncer status={liveStatus} />` once inside the existing `<main>`.

5. In `src/facilities/CrowdAlertSubscriptionCard.tsx`, accept the existing Phase 3 prop `summary: OccupancySummary` in every test/render path; do not restore obsolete `total`, `max`, or `percent` props. After a successful subscription response, set local announcer status to `"alert-created"` and render the same `LiveStatusAnnouncer`; show server errors in visible text with `role="alert"` and preserve form focus. Add the import rather than duplicating live-region markup. In `src/app/components/AlertsPanel.tsx`, define one stable `alerts-panel-title`, assign it to the visible `Alerts` heading, pass `aria-labelledby="alerts-panel-title"` to the desktop `Dialog`, and give the mobile drawer paper `role="dialog"`, `aria-modal="true"`, and the same `aria-labelledby`. The existing `Close alerts` control stays inside both labelled dialogs.

6. Update `index.html` with the exact required zoom-safe declaration `<meta name="viewport" content="width=device-width, initial-scale=1.0" />`, keeping the existing title and icon links and removing any maximum-scale or user-scalable restriction.

7. Create `tests/e2e/support/apiMocks.ts` as the one final-schema browser fixture. It must intercept only `/api/*`, use no provider calls, and return Phase 2/4/6/7-safe shapes for both facilities:

```ts
import type { Page } from "@playwright/test";

const observedAt = "2026-08-31T12:00:00Z";
const rows = [
  { LocationId: 5761, IsClosed: false, LastCount: 30, LastUpdatedDateAndTime: observedAt, FetchedAt: observedAt },
  { LocationId: 8717, IsClosed: false, LastCount: 24, LastUpdatedDateAndTime: observedAt, FetchedAt: observedAt },
];

const facilityName = (id: number) => id === 1656 ? "Bakke Recreation & Wellbeing Center" : "Nicholas Recreation Center";

export async function installDashboardApiMocks(page: Page): Promise<void> {
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const id = Number(url.pathname.match(/facilities\/(1186|1656)/)?.[1] ?? 1186);
    const common = { facilityId: id, facilityName: facilityName(id) };

    if (url.pathname === "/api/live-counts") {
      await route.fulfill({ json: { ingestion: { lastSuccessfulFetchAt: observedAt, ageSeconds: 0, status: "healthy" }, rows } });
      return;
    }
    if (/\/api\/forecast\/facilities\/(1186|1656)\/actual-hours$/.test(url.pathname)) {
      await route.fulfill({ json: { facilityId: id, date: "2026-08-31", categories: [], totalHours: [] } });
      return;
    }
    if (/\/api\/forecast\/facilities\/(1186|1656)$/.test(url.pathname)) {
      await route.fulfill({ json: { ...common, weeklyForecast: [] } });
      return;
    }
    if (/\/api\/facility-hours\/facilities\/(1186|1656)$/.test(url.pathname)) {
      await route.fulfill({ json: {
        generatedAt: observedAt,
        sourceSite: "https://recwell.example.test",
        ...common,
        slug: id === 1656 ? "bakke" : "nick",
        url: `https://recwell.example.test/${id === 1656 ? "bakke" : "nick"}/`,
        resolvedUrl: `https://recwell.example.test/${id === 1656 ? "bakke" : "nick"}/`,
        status: "ok",
        source: "direct_html",
        sections: [{ title: "Building Hours", rows: [{ label: "Monday", hours: "6:00 am - 10:00 pm" }], note: null }],
        sourceFetchedAt: observedAt,
        lastSuccessfulAt: observedAt,
        stale: false,
        error: null,
        errorCategory: null,
        updatedAt: observedAt,
      } });
      return;
    }
    if (url.pathname === "/api/push/availability") {
      await route.fulfill({ json: { apiAvailable: true, dbAvailable: true, alertsAvailable: true, reason: null } });
      return;
    }
    if (url.pathname === "/api/push/public-key") {
      await route.fulfill({ json: { publicKey: "B".repeat(87) } });
      return;
    }
    if (url.pathname === "/api/push/rules/list") {
      await route.fulfill({ json: { status: "ok", rules: [] } });
      return;
    }
    await route.fulfill({ status: 404, json: { detail: "fixture route not found" } });
  });
}
```

Update `tests/e2e/route-smoke.spec.ts` to delete its Phase 1 inline `liveRows`, `emptyForecast`, and `emptySchedule`, import `installDashboardApiMocks`, and call it in `test.beforeEach`. This makes the old route smoke consume the same post-hardening response contracts as this accessibility test.

8. Write `tests/e2e/accessibility.spec.ts` using the `@axe-core/playwright` dependency installed in Phase 1. The test fixes the browser clock before navigation, opens the actual Alerts dialog and heat-map dialog, and scans each surface:

```ts
import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { installDashboardApiMocks } from "./support/apiMocks";

const seriousOrCritical = (violations: ReadonlyArray<{ impact: string | null }>) =>
  violations.filter((violation) => ["serious", "critical"].includes(violation.impact ?? ""));

test("dashboard, alert form, and heat-map dialog have no serious accessibility violations", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.clock.setFixedTime(new Date("2026-08-31T12:00:00Z"));
  await installDashboardApiMocks(page);
  await page.goto("/nick");
  await expect(page.locator("main")).toBeVisible();

  const alertsButtonBox = await page.getByRole("button", { name: "Alerts" }).boundingBox();
  expect(alertsButtonBox).not.toBeNull();
  expect(alertsButtonBox!.width).toBeGreaterThanOrEqual(44);
  expect(alertsButtonBox!.height).toBeGreaterThanOrEqual(44);

  const dashboardResults = await new AxeBuilder({ page }).include("main").analyze();
  expect(seriousOrCritical(dashboardResults.violations)).toEqual([]);

  await page.getByRole("button", { name: "Alerts" }).click();
  const alertsDialog = page.getByRole("dialog", { name: "Alerts" });
  await expect(alertsDialog).toBeVisible();
  const alertsResults = await new AxeBuilder({ page }).include('[role="dialog"]').analyze();
  expect(seriousOrCritical(alertsResults.violations)).toEqual([]);
  await alertsDialog.getByRole("button", { name: "Close alerts" }).click();

  await page.getByRole("button", { name: /show map/i }).click();
  const zoneTargets = page.locator('polygon[role="button"]');
  expect(await zoneTargets.count()).toBeGreaterThan(0);
  for (let index = 0; index < await zoneTargets.count(); index += 1) {
    const targetBox = await zoneTargets.nth(index).boundingBox();
    expect(targetBox).not.toBeNull();
    expect(targetBox!.width).toBeGreaterThanOrEqual(24);
    expect(targetBox!.height).toBeGreaterThanOrEqual(24);
  }
  const zone = page.getByRole("button", { name: /Power House:/i });
  await zone.focus();
  await page.keyboard.press("Enter");
  const heatmapDialog = page.getByRole("dialog", { name: "Power House details" });
  await expect(heatmapDialog).toBeVisible();
  const heatmapResults = await new AxeBuilder({ page }).include('[role="dialog"]').analyze();
  expect(seriousOrCritical(heatmapResults.violations)).toEqual([]);
});
```

9. Run `npm run test:run -- src/facilities/LiveStatusAnnouncer.test.tsx`, `npm run lint`, `npm run build`, and `npx playwright test tests/e2e/route-smoke.spec.ts tests/e2e/accessibility.spec.ts tests/e2e/pwa-build.spec.ts`. Expected GREEN: live regions have concrete transition coverage, lint/build pass, final-schema route fixtures work for both facilities, both dialogs have accessible names, and axe finds no serious or critical issues in the dashboard, alert form, or opened heat-map dialog.

10. Commit:

```bash
git add src/facilities/LiveStatusAnnouncer.tsx src/facilities/LiveStatusAnnouncer.test.tsx src/facilities/CrowdAlertSubscriptionCard.tsx src/app/components/AlertsPanel.tsx src/app/App.tsx index.html tests/e2e/support/apiMocks.ts tests/e2e/route-smoke.spec.ts tests/e2e/accessibility.spec.ts
git commit -m "feat: announce live dashboard status accessibly"
```

## Final verification

1. Run `npm run lint`, `npm run test:run`, `npm run build`, and `npx playwright test`.
2. Build normally and serve `dist` on localhost: confirm debug controls are absent. Build with `npm run build -- --mode local-debug`, serve on localhost, and confirm they are present; serve that same build from a non-local hostname and confirm they are absent.
3. In a supported browser, install the PWA, load it once, then test an offline reload from the app shell. Confirm stale/offline UI remains truthful and live API data was not cached.
4. Send a test notification whose data URL is cross-origin and confirm notification click focuses or opens only the dashboard origin.
5. Inspect production output for no `public/sw.js`, an injected manifest worker, `/api/` navigation denial, and the bounded same-origin map/image and font cache policies.

## Self-review checklist

- [ ] The only production-build debug path is Vite mode `local-debug` on an exact local hostname; a normal production build on localhost does not enable it and no extra public `VITE_` variable exists.
- [ ] PWA virtual-module types, worker `self` declaration, push payload bounds, and same-origin notification target handling compile under TypeScript.
- [ ] The service worker does not cache live API responses or send cross-origin notification navigations; only same-origin non-API maps/images and fonts enter bounded runtime caches.
- [ ] Every heat-map zone has a programmatic name, keyboard operation, labelled non-modal dialog, focus restoration, and a tested minimum 24-by-24 CSS-pixel hit area; standard buttons and icon buttons keep 44-pixel touch targets.
- [ ] Refresh, error, and alert-created transitions use the concrete `LiveStatusAnnouncer`; no undeclared dashboard test helper or fixture remains.
- [ ] Production artifact assertions rely on emitted asset references and behaviour-required literals, so minification cannot invalidate them.
