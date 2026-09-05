import type {TouchEvent} from "react";
import type {FacilityId, FacilityPayload, LiveDataSource} from "../../lib/types/facility";
import type {ForecastDay, ForecastHour} from "../../lib/types/forecast";
import type {FacilityScheduleResponse} from "../../lib/api/schemas";
import type {ForecastHourBounds} from "../../app/hooks/useForecastData";
import type {LiveOutageState, WarningResolution} from "../../app/warningStatus";
import type {OccupancySummary} from "../../shared/occupancy/computeOccupancySummary";
import type {OccupancyThresholds} from "../../shared/utils/styles";
import type {FacilityOpenStatus, FacilityOpenWindow} from "../../shared/utils/facilityScheduleStatus";
import type {FacilityDashboardConfig, SectionConfig} from "../../facilities/constants";
import type {LiveStatus} from "../../facilities/LiveStatusAnnouncer";
// Temporary type-only seam until Task 5 creates the canonical alertTypes module.
import type {AlertSectionOption} from "../../facilities/CrowdAlertSubscriptionCard";

export interface ForecastDaySelection {
    key: string | null;
    offset: number;
}

export interface DashboardSelectorInput {
    facility: FacilityId;
    nowTs: number;
    data: FacilityPayload | null;
    isLoading: boolean;
    error: string | null;
    liveDataSource: LiveDataSource | null;
    liveOutageState: LiveOutageState;
    hasPendingLiveRetry: boolean;
    cacheTimestampMs: number | null;
    isOffline: boolean;
    forecastDays: ForecastDay[];
    forecastOccupancyThresholds: OccupancyThresholds | null;
    forecastSectionOccupancyThresholds: Partial<Record<string, OccupancyThresholds>>;
    forecastLocationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>>;
    forecastHourBounds: ForecastHourBounds;
    forecastError: string | null;
    isForecastLoading: boolean;
    hasPendingForecastRetry: boolean;
    activeSchedule: FacilityScheduleResponse | null;
    isFacilityHoursLoading: boolean;
    facilityHoursError: string | null;
    hasPendingScheduleRetry: boolean;
    forecastDaySelection: ForecastDaySelection;
    predictionOverrideEnabled: boolean;
    closureOverrideEnabled: boolean;
}

export interface DashboardViewModel {
    activeData: FacilityPayload | null;
    facilitySummary: OccupancySummary;
    sectionSummaries: ReadonlyMap<string, OccupancySummary>;
    otherSummary: OccupancySummary | null;
    alertSections: AlertSectionOption[];
    dashboardConfig: FacilityDashboardConfig;
    knownIds: number[];
    sectionConfigs: SectionConfig[];
    hasOtherSectionLocations: boolean;
    visibleForecastDays: ForecastDay[];
    todayForecastDay: ForecastDay | null;
    selectedForecastDay: ForecastDay | null;
    forecastDisplayKey: string;
    resolvedForecastDayOffset: number;
    forecastHourBounds: ForecastHourBounds;
    todayDateKey: string | null;
    tomorrowDateKey: string | null;
    nextOpenDateKey: string | null;
    nextOpenLabel: string | null;
    scheduleStatus: FacilityOpenStatus;
    showClosedFacilityMode: boolean;
    isExpectedOpenTomorrow: boolean;
    canShowClosedTomorrowForecast: boolean;
    canShowActiveDailyForecast: boolean;
    canShowDailyForecastCard: boolean;
    canShowHourlyRoomForecasts: boolean;
    warning: WarningResolution;
    warningText: string | null;
    occupancyThresholds: OccupancyThresholds | null;
    sectionOccupancyThresholds: Partial<Record<string, OccupancyThresholds>>;
    sectionForecastMap: Record<string, ForecastHour[]>;
    sectionForecastOpenWindows: FacilityOpenWindow[];
    enforceSectionForecastWorkingHours: boolean;
}

export interface UseDashboardStateArgs {
    initialFacility?: FacilityId;
    onFacilityRouteChange?: (facility: FacilityId) => void;
    isPhoneViewport: boolean;
}

export interface DashboardState extends DashboardSelectorInput {
    view: DashboardViewModel;
    liveRefreshKey: number;
    forecastRefreshKey: number;
    scheduleRefreshKey: number;
    liveStatus: LiveStatus;
    lastManualRefresh: number;
    debugEnabled: boolean;
    debugNowMs: number | null;
    isCrowdAlertOpen: boolean;
    isInstallGuideOpen: boolean;
    isStandalonePwa: boolean;
    isTouchCapable: boolean;
    enablePullToRefresh: boolean;
    pullDistance: number;
    isPulling: boolean;
    isReadyToRefresh: boolean;
    showPullIndicator: boolean;
    handleFacilitySelect: (facility: FacilityId) => void;
    manualRefresh: () => void;
    setForecastDaySelection: (selection: ForecastDaySelection) => void;
    setIsCrowdAlertOpen: (open: boolean) => void;
    setIsInstallGuideOpen: (open: boolean) => void;
    setPredictionOverride: (enabled: boolean) => void;
    setClosureOverride: (enabled: boolean) => void;
    setDebugNowOverride: (value: string | null) => number | null;
    resetPullGesture: () => void;
    handleTouchStart: (event: TouchEvent<HTMLDivElement>) => void;
    handleTouchMove: (event: TouchEvent<HTMLDivElement>) => void;
    handleTouchEnd: () => void;
}
