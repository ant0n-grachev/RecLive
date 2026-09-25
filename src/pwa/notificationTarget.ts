export interface NotificationWindowClient {
    url: string;
    navigate(url: string): Promise<unknown>;
    focus(): Promise<unknown>;
}

export interface NotificationClientScope {
    origin: string;
    matchAll(options: {
        type: "window";
        includeUncontrolled: boolean;
    }): Promise<readonly NotificationWindowClient[]>;
    openWindow(url: string): Promise<unknown>;
}

export async function openOrFocusSameOrigin(
    rawTarget: unknown,
    scope: NotificationClientScope,
): Promise<"focused" | "opened"> {
    const fallback = new URL("/", `${scope.origin}/`);
    let safe = fallback;

    if (
        typeof rawTarget === "string"
        && rawTarget.startsWith("/")
        && !rawTarget.startsWith("//")
        && !rawTarget.includes("\\")
    ) {
        try {
            const candidate = new URL(rawTarget, fallback);
            if (candidate.origin === fallback.origin) safe = candidate;
        } catch {
            // Retain the same-origin root fallback.
        }
    }

    const clients = await scope.matchAll({type: "window", includeUncontrolled: true});
    const existing = clients.find((client) => {
        try {
            return new URL(client.url).origin === fallback.origin;
        } catch {
            return false;
        }
    });

    if (existing) {
        try {
            await existing.navigate(safe.href);
        } catch {
            // Focus the existing client even when navigation is rejected.
        }
        await existing.focus();
        return "focused";
    }

    await scope.openWindow(safe.href);
    return "opened";
}
