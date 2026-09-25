export interface RuntimeRequestLike {
    url: string;
    destination: string;
}

export type RuntimeAssetCacheKind = "image" | "font";

export function runtimeAssetCacheKind(
    request: RuntimeRequestLike,
    origin: string,
): RuntimeAssetCacheKind | null {
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
