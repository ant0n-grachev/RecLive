export interface PushMessageDataLike {
    json(): unknown;
}

export interface SafePushPayload {
    title: string;
    body: string;
    url: string;
}

const fallback: SafePushPayload = {
    title: "RecLive alert",
    body: "Your occupancy alert is ready.",
    url: "/",
};

const stringAtMost = (value: unknown, maximum: number, fallbackValue: string) => (
    typeof value === "string" && value.length > 0
        ? value.slice(0, maximum)
        : fallbackValue
);

export function safePushPayload(data: PushMessageDataLike | null): SafePushPayload {
    try {
        const value: unknown = data?.json();
        if (typeof value !== "object" || value === null) return fallback;

        const record = value as Record<string, unknown>;
        const rawUrl = stringAtMost(record.url, 2048, fallback.url);
        const url = rawUrl.startsWith("/") && !rawUrl.startsWith("//") && !rawUrl.includes("\\")
            ? rawUrl
            : fallback.url;

        return {
            title: stringAtMost(record.title, 80, fallback.title),
            body: stringAtMost(record.body, 240, fallback.body),
            url,
        };
    } catch {
        return fallback;
    }
}
