const trimEnvValue = (value: unknown): string => (
    typeof value === "string" ? value.trim() : ""
);

const readViteEnv = (name: string): string => trimEnvValue(import.meta.env[name as keyof ImportMetaEnv]);

const VITE_DEFAULTS = {
    VITE_LIVE_COUNTS_URL: "/api/live-counts",
    VITE_FORECAST_API_BASE_URL: "",
    VITE_PUSH_API_BASE_URL: "",
    VITE_SITE_URL: "",
} as const;

const viteEnvWithDefault = (name: keyof typeof VITE_DEFAULTS): string => {
    const value = readViteEnv(name);
    return value || VITE_DEFAULTS[name];
};

const stripTrailingSlashes = (value: string): string => value.replace(/\/+$/, "");

const siteUrl = viteEnvWithDefault("VITE_SITE_URL");

export const env = {
    isDev: import.meta.env.DEV,
    liveCountsUrl: viteEnvWithDefault("VITE_LIVE_COUNTS_URL"),
    forecastApiBaseUrl: stripTrailingSlashes(viteEnvWithDefault("VITE_FORECAST_API_BASE_URL")),
    pushApiBaseUrl: stripTrailingSlashes(viteEnvWithDefault("VITE_PUSH_API_BASE_URL")),
    siteUrl: siteUrl ? stripTrailingSlashes(siteUrl) : "",
};
