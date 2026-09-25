import type {LiveDataSource} from "../lib/types/facility";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";

export type LiveOutageState = "none" | "cache" | "no_cache";

export interface WarningResolverInput {
    hasAnyError: boolean;
    isOffline: boolean;
    liveOutageState: LiveOutageState;
    liveDataSource: LiveDataSource | null;
    forecastError: string | null;
    isScheduledClosedNow: boolean;
    isScheduledOpenButDataNotLive: boolean;
    occupancyStatus: OccupancySummary["status"];
}

export type WarningKind =
    | "none"
    | "offline_cache"
    | "total_outage_cache"
    | "prediction_unavailable"
    | "facility_fallback"
    | "scheduled_closed"
    | "scheduled_open_not_live"
    | "partial_live"
    | "occupancy_unavailable";

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
    partial_live:
        "Live occupancy is based on partial observed-capacity coverage. Forecasts remain available.",
    occupancy_unavailable:
        "Live occupancy unavailable. Forecasts remain available.",
};

const UNAVAILABLE_OCCUPANCY_WARNING_TEXT: Partial<Record<WarningKind, string>> = {
    offline_cache:
        "You're offline right now. Showing your last saved snapshot. Live occupancy unavailable, and predictions are hidden until the connection is back.",
    total_outage_cache:
        "Live and prediction services are temporarily unavailable right now. Showing the last saved snapshot while systems recover. Live occupancy unavailable, and predictions are hidden.",
    prediction_unavailable:
        "Prediction services are temporarily unavailable right now. Live occupancy unavailable, and forecasts are hidden for now.",
    facility_fallback:
        "Live updates are currently running on our backup feed. Live occupancy unavailable, and predictions are hidden for now.",
    scheduled_open_not_live:
        "The gym is open according to the official schedule. Live occupancy unavailable. Forecasts remain available.",
};

const CLOSED_OCCUPANCY_WARNING_TEXT: Partial<Record<WarningKind, string>> = {
    prediction_unavailable:
        "Prediction services are temporarily unavailable right now. The gym is currently CLOSED, and forecasts are hidden for now.",
    facility_fallback:
        "Live updates are currently running on our backup feed. The gym is currently CLOSED, and predictions are hidden for now.",
    scheduled_open_not_live:
        "The official schedule says the gym is open, but the current occupancy status is CLOSED.",
};

export const resolveDashboardWarning = ({
    hasAnyError,
    isOffline,
    liveOutageState,
    liveDataSource,
    forecastError,
    isScheduledClosedNow,
    isScheduledOpenButDataNotLive,
    occupancyStatus,
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
    } else if (occupancyStatus === "partial") {
        kind = "partial_live";
    } else if (occupancyStatus === "insufficient" || occupancyStatus === "unknown") {
        kind = "occupancy_unavailable";
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

    const occupancyUnavailable = occupancyStatus === "insufficient" || occupancyStatus === "unknown";
    const text = occupancyStatus === "closed"
        ? (CLOSED_OCCUPANCY_WARNING_TEXT[kind] ?? WARNING_TEXT[kind])
        : occupancyUnavailable
          ? (UNAVAILABLE_OCCUPANCY_WARNING_TEXT[kind] ?? WARNING_TEXT[kind])
          : WARNING_TEXT[kind];

    if (
        kind === "scheduled_open_not_live"
        || kind === "partial_live"
        || kind === "occupancy_unavailable"
    ) {
        return {
            kind,
            text,
            hidePredictions: false,
        };
    }

    return {
        kind,
        text,
        hidePredictions: true,
    };
};
