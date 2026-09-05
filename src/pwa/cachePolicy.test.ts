import {describe, expect, it} from "vitest";
import {runtimeAssetCacheKind} from "./cachePolicy";

describe("runtimeAssetCacheKind", () => {
    const origin = "https://dashboard.example";

    it("classifies only same-origin non-API maps, images, and fonts", () => {
        expect(runtimeAssetCacheKind({url: `${origin}/floor-maps/nick.png`, destination: "image"}, origin)).toBe("image");
        expect(runtimeAssetCacheKind({url: `${origin}/assets/app.woff2`, destination: "font"}, origin)).toBe("font");
        expect(runtimeAssetCacheKind({url: `${origin}/api/floor-map.png`, destination: "image"}, origin)).toBeNull();
        expect(runtimeAssetCacheKind({url: "https://cdn.example/nick.png", destination: "image"}, origin)).toBeNull();
        expect(runtimeAssetCacheKind({url: `${origin}/api/live-counts`, destination: ""}, origin)).toBeNull();
    });

    it("rejects invalid URLs and non-static request destinations", () => {
        expect(runtimeAssetCacheKind({url: "not a url", destination: "image"}, origin)).toBeNull();
        expect(runtimeAssetCacheKind({url: `${origin}/assets/app.js`, destination: "script"}, origin)).toBeNull();
        expect(runtimeAssetCacheKind({url: `${origin}/forecast`, destination: ""}, origin)).toBeNull();
    });
});
