/// <reference types="vite/client" />

interface ImportMetaEnv {
    readonly VITE_LIVE_COUNTS_URL?: string;
    readonly VITE_FORECAST_API_BASE_URL?: string;
    readonly VITE_PUSH_API_BASE_URL?: string;
    readonly VITE_SITE_URL?: string;
}

interface ImportMeta {
    readonly env: ImportMetaEnv;
}

interface Window {
    recliveShowPredictions?: () => string;
    recliveRestoreWarnings?: () => string;
    reclivePredictionOverrideStatus?: () => boolean;
}
