import type {LiveDataSource} from "../lib/types/facility";

export type LiveOutageState = "none" | "cache" | "no_cache";

export interface WarningResolverInput {
    hasAnyError: boolean;
    isOffline: boolean;
    liveOutageState: LiveOutageState;
    liveDataSource: LiveDataSource | null;
    forecastError: string | null;
    isScheduledClosedNow: boolean;
    isScheduledOpenButDataNotLive: boolean;
    isDataLikelyStale: boolean;
}

export type WarningKind =
    | "none"
    | "offline_cache"
    | "total_outage_cache"
    | "prediction_unavailable"
    | "facility_fallback"
    | "scheduled_closed"
    | "scheduled_open_not_live"
    | "stale";

export interface WarningResolution {
    kind: WarningKind;
    text: string | null;
    hidePredictions: boolean;
}

const WARNING_TEXT: Record<Exclude<WarningKind, "none">, string> = {
    offline_cache:
        "You're offline right now. Showing your last saved snapshot; live occupancy is paused and predictions are hidden until the connection is back.",
    total_outage_cache:
        "Live and prediction services are temporarily unavailable right now. Showing the last saved snapshot while systems recover, so occupancy is not live and predictions are hidden.",
    prediction_unavailable:
        "Prediction services are temporarily unavailable right now. Live occupancy is still shown, but forecasts are hidden for now.",
    facility_fallback:
        "Live updates are currently running on our backup feed, so occupancy may be slightly delayed and predictions are hidden for now.",
    scheduled_closed: "",
    scheduled_open_not_live:
        "The gym is open according to the official schedule, but live occupancy has not updated since opening. Current counts may be delayed.",
    stale:
        "The gym may be closed right now. Occupancy info is not live, and hourly room forecasts are hidden.",
};

export const resolveDashboardWarning = ({
    hasAnyError,
    isOffline,
    liveOutageState,
    liveDataSource,
    forecastError,
    isScheduledClosedNow,
    isScheduledOpenButDataNotLive,
    isDataLikelyStale,
}: WarningResolverInput): WarningResolution => {
    let kind: WarningKind = "none";

    if (isOffline && liveOutageState === "cache") {
        kind = "offline_cache";
    } else if (!isOffline && liveOutageState === "cache") {
        kind = "total_outage_cache";
    } else if (liveOutageState === "none" && liveDataSource === "fallback_api") {
        kind = "facility_fallback";
    } else if (isScheduledClosedNow) {
        kind = "scheduled_closed";
    } else if (isScheduledOpenButDataNotLive) {
        kind = "scheduled_open_not_live";
    } else if (forecastError) {
        kind = "prediction_unavailable";
    } else if (isDataLikelyStale) {
        kind = "stale";
    }

    if (kind === "none") {
        return {
            kind,
            text: null,
            hidePredictions: hasAnyError,
        };
    }

    // Scheduled closures are expected behavior. Show status in the dedicated schedule card,
    // but avoid rendering a warning banner.
    if (kind === "scheduled_closed") {
        return {
            kind,
            text: null,
            hidePredictions: true,
        };
    }

    if (kind === "scheduled_open_not_live") {
        return {
            kind,
            text: WARNING_TEXT[kind],
            hidePredictions: false,
        };
    }

    return {
        kind,
        text: WARNING_TEXT[kind],
        hidePredictions: true,
    };
};
