export interface PublicEnv {
    apiBaseUrl: string;
    siteUrl: string;
    isDev: boolean;
}

const unsafePlaceholder = (value: string): boolean => (
    value === "change_me" || value.includes("YOUR_ACCOUNT_API_KEY")
);

const normalizedPublicUrl = (
    name: "VITE_API_BASE_URL" | "VITE_SITE_URL",
    value: unknown,
    required: boolean,
): string => {
    const text = typeof value === "string" ? value.trim() : "";
    if (!text) {
        if (required) {
            throw new Error(`${name} is not configured safely`);
        }
        return "";
    }
    if (unsafePlaceholder(text)) {
        throw new Error(`${name} is not configured safely`);
    }

    try {
        const parsed = new URL(text);
        if (
            !["http:", "https:"].includes(parsed.protocol)
            || parsed.username
            || parsed.password
            || parsed.search
            || parsed.hash
        ) {
            throw new Error("unsafe public URL");
        }
        return parsed.toString().replace(/\/+$/, "");
    } catch {
        throw new Error(`${name} is not configured safely`);
    }
};

export const parsePublicEnv = (
    values: Record<string, unknown>,
    isProduction: boolean,
): PublicEnv => {
    const apiBaseUrl = normalizedPublicUrl(
        "VITE_API_BASE_URL",
        values.VITE_API_BASE_URL,
        isProduction,
    );
    const siteUrl = normalizedPublicUrl(
        "VITE_SITE_URL",
        values.VITE_SITE_URL,
        isProduction,
    );
    return {apiBaseUrl, siteUrl, isDev: !isProduction};
};

const parsed = parsePublicEnv(
    import.meta.env as Record<string, unknown>,
    import.meta.env.PROD,
);

export const env = {
    ...parsed,
    isDev: import.meta.env.DEV,
};
