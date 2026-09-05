import {Suspense, lazy, useMemo} from "react";
import {Alert, Box, CircularProgress, Container, Stack, Typography, useMediaQuery} from "@mui/material";
import {AnimatePresence, motion, useReducedMotion} from "framer-motion";
import {useTheme, type PaletteMode} from "@mui/material/styles";
import FacilitySelector from "../facilities/FacilitySelector";
import OccupancyHero from "../facilities/OccupancyHero";
import SectionCommandCenter from "../facilities/SectionCommandCenter";
import SectionSummaryOther from "../facilities/SectionSummaryOther";
import ForecastWindowsCard from "../facilities/ForecastWindowsCard";
import ScheduleStatusCard from "../facilities/ScheduleStatusCard";
import FacilityHoursBlock from "../facilities/FacilityHoursBlock";
import type {FacilityId} from "../lib/types/facility";
import {useFacilitySeo} from "./seo";
import AlertsPanel from "./components/AlertsPanel";
import InstallGuideDialog from "./components/InstallGuideDialog";
import AppFooter from "./components/AppFooter";
import ThemeModeToggle from "../shared/components/ThemeModeToggle";
import {LiveStatusAnnouncer} from "../facilities/LiveStatusAnnouncer";
import {normalizeSectionTitle} from "../features/dashboard/dashboardSelectors";
import {useDashboardState} from "../features/dashboard/useDashboardState";

const FloorHeatMapCard = lazy(() => import("../facilities/FloorHeatMapCard"));
const CONTENT_EASE = [0.22, 1, 0.36, 1] as const;
const CONTENT_EXIT_EASE = [0.4, 0, 1, 1] as const;

export interface AppProps {
    initialFacility?: FacilityId;
    onFacilityRouteChange?: (facility: FacilityId) => void;
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}

export default function App({
    initialFacility,
    onFacilityRouteChange,
    themeMode,
    onThemeModeChange,
}: AppProps) {
    const reduceMotion = useReducedMotion();
    const theme = useTheme();
    const useDesktopAlertsModal = useMediaQuery(theme.breakpoints.up("md"));
    const isPhoneViewport = useMediaQuery(theme.breakpoints.down("sm"));
    const {
        facility,
        nowTs,
        data,
        isLoading,
        error,
        forecastLocationOccupancyThresholds,
        forecastError,
        isForecastLoading,
        activeSchedule,
        isFacilityHoursLoading,
        facilityHoursError,
        liveStatus,
        isCrowdAlertOpen,
        isInstallGuideOpen,
        isStandalonePwa,
        isTouchCapable,
        enablePullToRefresh,
        pullDistance,
        isPulling,
        isReadyToRefresh,
        showPullIndicator,
        handleFacilitySelect,
        setForecastDaySelection,
        setIsCrowdAlertOpen,
        setIsInstallGuideOpen,
        resetPullGesture,
        handleTouchStart,
        handleTouchMove,
        handleTouchEnd,
        view,
    } = useDashboardState({initialFacility, onFacilityRouteChange, isPhoneViewport});
    const {
        activeData,
        facilitySummary,
        alertSections,
        dashboardConfig,
        knownIds,
        sectionConfigs,
        hasOtherSectionLocations,
        visibleForecastDays,
        selectedForecastDay,
        forecastDisplayKey,
        resolvedForecastDayOffset,
        nextOpenLabel,
        scheduleStatus,
        showClosedFacilityMode,
        canShowDailyForecastCard,
        warningText,
        occupancyThresholds,
        sectionOccupancyThresholds,
        sectionForecastMap,
    } = view;
    useFacilitySeo(facility);
    const facilityContentVariants = useMemo(
        () => (reduceMotion
            ? {
                hidden: {opacity: 1, y: 0},
                show: {opacity: 1, y: 0},
                exit: {opacity: 1, y: 0},
            }
            : {
                hidden: {opacity: 0, y: 14},
                show: {
                    opacity: 1,
                    y: 0,
                    transition: {
                        duration: 0.32,
                        ease: CONTENT_EASE,
                        staggerChildren: 0.075,
                        delayChildren: 0.03,
                    },
                },
                exit: {
                    opacity: 0,
                    y: -8,
                    transition: {duration: 0.18, ease: CONTENT_EXIT_EASE},
                },
            }),
        [reduceMotion]
    );
    const facilityItemVariants = useMemo(
        () => (reduceMotion
            ? {
                hidden: {opacity: 1, y: 0},
                show: {opacity: 1, y: 0},
            }
            : {
                hidden: {opacity: 0, y: 10},
                show: {
                    opacity: 1,
                    y: 0,
                    transition: {duration: 0.28, ease: CONTENT_EASE},
                },
            }),
        [reduceMotion]
    );

    const pullIndicatorHeight = showPullIndicator
        ? 28
        : (isPulling ? Math.min(28, Math.max(12, Math.round(pullDistance * 0.35))) : 0);
    const pullIndicatorText = showPullIndicator
        ? "Refreshing..."
        : (isReadyToRefresh ? "Release to refresh" : "Pull down to refresh");
    const sectionBlockGap = {xs: 2, sm: 2.5} as const;
    return (
        <Box
            component="main"
            onTouchStart={enablePullToRefresh ? handleTouchStart : undefined}
            onTouchMove={enablePullToRefresh ? handleTouchMove : undefined}
            onTouchEnd={enablePullToRefresh ? handleTouchEnd : undefined}
            onTouchCancel={enablePullToRefresh ? resetPullGesture : undefined}
            sx={{
                pt: {xs: 2, sm: 3},
                pb: {xs: 2.5, sm: 3.5},
                bgcolor: "background.default",
                minHeight: "100vh",
                overscrollBehaviorY: enablePullToRefresh ? "contain" : undefined,
            }}
        >
            <LiveStatusAnnouncer status={liveStatus}/>
            <Container
                maxWidth="sm"
                sx={{
                    display: "flex",
                    flexDirection: "column",
                    gap: {xs: 2, sm: 3},
                    pt: {xs: 1.5, sm: 2},
                    px: {xs: 2, sm: 0},
                }}
            >
                <Box
                    sx={{
                        height: pullIndicatorHeight,
                        opacity: pullIndicatorHeight > 0 ? 1 : 0,
                        overflow: "hidden",
                        transition: "height 180ms ease, opacity 180ms ease",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                    }}
                >
                    <Stack direction="row" spacing={0.8} alignItems="center">
                        {showPullIndicator && (
                            <CircularProgress size={14} thickness={6}/>
                        )}
                        <Typography variant="caption" color="text.secondary" sx={{fontWeight: 700}}>
                            {pullIndicatorText}
                        </Typography>
                    </Stack>
                </Box>

                <FacilitySelector facility={facility} onSelect={handleFacilitySelect}/>

                {isLoading && !data && (
                    <Box
                        sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            gap: 2,
                            py: 5,
                        }}
                    >
                        <CircularProgress size={28} thickness={5}/>
                        <Typography color="text.secondary" fontWeight={600}>
                            Loading...
                        </Typography>
                    </Box>
                )}

                {error && !activeData && (
                    <Alert severity="warning" sx={{borderRadius: 2}}>
                        {error}
                    </Alert>
                )}

                <AnimatePresence mode="wait" initial={false}>
                    {activeData && (
                        <Box
                            key={facility}
                            component={motion.div}
                            variants={facilityContentVariants}
                            initial="hidden"
                            animate="show"
                            exit="exit"
                            sx={{display: "flex", flexDirection: "column", gap: sectionBlockGap}}
                        >
                            {!showClosedFacilityMode && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <OccupancyHero
                                        summary={facilitySummary}
                                        nowTs={nowTs}
                                        facilityId={facility}
                                        headerAction={
                                            !showClosedFacilityMode
                                                ? (
                                                    <ThemeModeToggle
                                                        themeMode={themeMode}
                                                        onThemeModeChange={onThemeModeChange}
                                                    />
                                                )
                                                : null
                                        }
                                        occupancyThresholds={occupancyThresholds}
                                        onOpenAlerts={() => setIsCrowdAlertOpen(true)}
                                    />
                                </Box>
                            )}

                            <Box component={motion.div} variants={facilityItemVariants}>
                                <ScheduleStatusCard
                                    status={scheduleStatus}
                                    nextOpenLabel={nextOpenLabel}
                                    footerAction={
                                        showClosedFacilityMode
                                            ? (
                                                <ThemeModeToggle
                                                    themeMode={themeMode}
                                                    onThemeModeChange={onThemeModeChange}
                                                />
                                            )
                                            : null
                                    }
                                />
                            </Box>

                            {!showClosedFacilityMode && warningText && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <Alert severity="warning" variant="outlined" sx={{borderRadius: 2}}>
                                        {warningText}
                                    </Alert>
                                </Box>
                            )}

                            {canShowDailyForecastCard && visibleForecastDays.length > 0 && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <ForecastWindowsCard
                                        day={selectedForecastDay}
                                        comparisonDays={visibleForecastDays}
                                        schedule={activeSchedule}
                                        occupancyThresholds={occupancyThresholds}
                                        dayOffset={resolvedForecastDayOffset}
                                        titleDayOffset={
                                            showClosedFacilityMode
                                                ? resolvedForecastDayOffset + 1
                                                : resolvedForecastDayOffset
                                        }
                                        totalDays={visibleForecastDays.length}
                                        canPrev={resolvedForecastDayOffset > 0}
                                        canNext={resolvedForecastDayOffset < Math.min(3, visibleForecastDays.length - 1)}
                                        onPrev={() => setForecastDaySelection({
                                            key: forecastDisplayKey,
                                            offset: Math.max(0, resolvedForecastDayOffset - 1),
                                        })}
                                        onNext={() =>
                                            setForecastDaySelection({
                                                key: forecastDisplayKey,
                                                offset: Math.min(
                                                    Math.min(3, visibleForecastDays.length - 1),
                                                    resolvedForecastDayOffset + 1
                                                ),
                                            })
                                        }
                                        isLoading={isForecastLoading}
                                        error={forecastError}
                                    />
                                </Box>
                            )}

                            {!showClosedFacilityMode && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <Suspense
                                        fallback={(
                                            <Box sx={{display: "grid", placeItems: "center", py: 4}}>
                                                <CircularProgress size={24} thickness={5}/>
                                            </Box>
                                        )}
                                    >
                                        <FloorHeatMapCard
                                            facilityId={facility}
                                            locations={activeData.locations}
                                            nowTs={nowTs}
                                            occupancyThresholds={occupancyThresholds}
                                            locationOccupancyThresholds={forecastLocationOccupancyThresholds}
                                        />
                                    </Suspense>
                                </Box>
                            )}

                            {!showClosedFacilityMode && (
                                <Stack
                                    component={motion.div}
                                    variants={facilityItemVariants}
                                    spacing={sectionBlockGap}
                                    sx={{mt: sectionBlockGap, mb: sectionBlockGap}}
                                >
                                    {sectionConfigs.map((section) => (
                                        <Box key={section.title} component={motion.div} variants={facilityItemVariants}>
                                            <SectionCommandCenter
                                                title={section.title}
                                                ids={[...section.ids]}
                                                locations={activeData.locations}
                                                nowTs={nowTs}
                                                forecast={sectionForecastMap[normalizeSectionTitle(section.title)]}
                                                occupancyThresholds={
                                                    sectionOccupancyThresholds[normalizeSectionTitle(section.title)]
                                                    ?? occupancyThresholds
                                                }
                                                locationOccupancyThresholds={forecastLocationOccupancyThresholds}
                                            />
                                        </Box>
                                    ))}
                                </Stack>
                            )}

                            {!showClosedFacilityMode && hasOtherSectionLocations && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <SectionSummaryOther
                                        title={dashboardConfig.otherTitle}
                                        exclude={knownIds}
                                        locations={activeData.locations}
                                        nowTs={nowTs}
                                        occupancyThresholds={occupancyThresholds}
                                        locationOccupancyThresholds={forecastLocationOccupancyThresholds}
                                    />
                                </Box>
                            )}

                            <Box component={motion.div} variants={facilityItemVariants}>
                                <FacilityHoursBlock
                                    facilityName={activeData.facilityName}
                                    schedule={activeSchedule}
                                    isLoading={isFacilityHoursLoading}
                                    error={facilityHoursError}
                                />
                            </Box>
                        </Box>
                    )}
                </AnimatePresence>

                <AlertsPanel
                    open={isCrowdAlertOpen}
                    onClose={() => setIsCrowdAlertOpen(false)}
                    facility={facility}
                    sections={alertSections}
                    useDesktopModal={useDesktopAlertsModal}
                    isStandalonePwa={isStandalonePwa}
                    isTouchCapable={isTouchCapable}
                />

                <InstallGuideDialog
                    open={!isStandalonePwa && isInstallGuideOpen}
                    onClose={() => setIsInstallGuideOpen(false)}
                />

                <AppFooter
                    isStandalonePwa={isStandalonePwa}
                    onOpenInstallGuide={() => setIsInstallGuideOpen(true)}
                />
            </Container>
        </Box>
    );
}
