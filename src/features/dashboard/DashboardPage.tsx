import {Suspense, lazy, useMemo} from "react";
import {Box, Button, CircularProgress, Container, Stack, Typography, useMediaQuery} from "@mui/material";
import {useTheme, type PaletteMode} from "@mui/material/styles";
import {AnimatePresence, motion, useReducedMotion} from "framer-motion";
import AlertsPanel from "../../app/components/AlertsPanel";
import AppFooter from "../../app/components/AppFooter";
import InstallGuideDialog from "../../app/components/InstallGuideDialog";
import FacilityHoursBlock from "../../facilities/FacilityHoursBlock";
import FacilitySelector from "../../facilities/FacilitySelector";
import ForecastWindowsCard from "../../facilities/ForecastWindowsCard";
import {LiveStatusAnnouncer} from "../../facilities/LiveStatusAnnouncer";
import OccupancyHero from "../../facilities/OccupancyHero";
import ScheduleStatusCard from "../../facilities/ScheduleStatusCard";
import SectionCommandCenter from "../../facilities/SectionCommandCenter";
import SectionSummaryOther from "../../facilities/SectionSummaryOther";
import ThemeModeToggle from "../../shared/components/ThemeModeToggle";
import {normalizeSectionTitle} from "./dashboardSelectors";
import type {DashboardState, DashboardViewModel} from "./dashboardTypes";

const FloorHeatMapCard = lazy(() => import("../../facilities/FloorHeatMapCard"));
const CONTENT_EASE = [0.22, 1, 0.36, 1] as const;
const CONTENT_EXIT_EASE = [0.4, 0, 1, 1] as const;

export interface DashboardPageProps {
    state: DashboardState;
    view: DashboardViewModel;
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}

export function DashboardPage({state, view, themeMode, onThemeModeChange}: DashboardPageProps) {
    const reduceMotion = useReducedMotion();
    const theme = useTheme();
    const useDesktopAlertsModal = useMediaQuery(theme.breakpoints.up("md"));
    const {
        facility, nowTs, isLoading, error, forecastLocationOccupancyThresholds,
        forecastError, isForecastLoading, activeSchedule, isFacilityHoursLoading,
        facilityHoursError, liveStatus, isCrowdAlertOpen, isInstallGuideOpen,
        isStandalonePwa, isTouchCapable, enablePullToRefresh, pullDistance,
        isPulling, isReadyToRefresh, showPullIndicator, handleFacilitySelect,
        setForecastDaySelection, setIsCrowdAlertOpen, setIsInstallGuideOpen,
        resetPullGesture, handleTouchStart, handleTouchMove, handleTouchEnd, manualRefresh,
    } = state;
    const {
        activeData, facilitySummary, alertSections, dashboardConfig, knownIds,
        sectionConfigs, hasOtherSectionLocations, visibleForecastDays,
        selectedForecastDay, forecastDisplayKey, resolvedForecastDayOffset,
        nextOpenLabel, scheduleStatus, showClosedFacilityMode,
        canShowDailyForecastCard, occupancyThresholds,
        sectionOccupancyThresholds, sectionForecastMap,
    } = view;
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
    const hasUsableOccupancy = facilitySummary.status === "live" || facilitySummary.status === "closed";
    if (!hasUsableOccupancy) {
        const waitingForFirstReading = !activeData && (isLoading || !error);
        return (
            <Box component="main" sx={{minHeight: "100vh", bgcolor: "background.default", py: {xs: 3.5, sm: 5}}}>
                <Container maxWidth="sm">
                    <FacilitySelector facility={facility} onSelect={handleFacilitySelect}/>
                    <Stack spacing={2} alignItems="center" justifyContent="center" sx={{minHeight: "60vh", textAlign: "center"}}>
                        {waitingForFirstReading ? (
                            <>
                                <CircularProgress size={28} aria-label="Loading occupancy"/>
                                <Typography role="status" color="text.secondary">Loading...</Typography>
                            </>
                        ) : (
                            <>
                                <Typography component="h1" variant="h5" sx={{fontWeight: 700}}>
                                    RecLive is unavailable.
                                </Typography>
                                <Button variant="outlined" onClick={manualRefresh} disabled={isLoading}
                                    sx={{minHeight: 44, borderRadius: 999, textTransform: "none",
                                        color: "text.primary", borderColor: "text.secondary"}}>
                                    Try again
                                </Button>
                            </>
                        )}
                    </Stack>
                </Container>
            </Box>
        );
    }
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
                        {showPullIndicator && <CircularProgress size={14} thickness={6}/>}
                        <Typography variant="caption" color="text.secondary" sx={{fontWeight: 700}}>
                            {pullIndicatorText}
                        </Typography>
                    </Stack>
                </Box>

                <FacilitySelector facility={facility} onSelect={handleFacilitySelect}/>

                <AnimatePresence mode="wait" initial={false}>
                    {activeData && (
                        <Box
                            key={facility}
                            component={motion.div}
                            variants={facilityContentVariants}
                            initial="hidden"
                            animate="show"
                            exit="exit"
                            sx={{display: "flex", flexDirection: "column", gap: sectionBlockGap,
                                "& > :empty, & > * > :empty": {display: "none"}}}
                        >
                            {!showClosedFacilityMode && (
                                <Box component={motion.div} variants={facilityItemVariants}>
                                    <OccupancyHero
                                        summary={facilitySummary}
                                        nowTs={nowTs}
                                        facilityId={facility}
                                        headerAction={(
                                            <ThemeModeToggle
                                                themeMode={themeMode}
                                                onThemeModeChange={onThemeModeChange}
                                            />
                                        )}
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
                                    useFlexGap
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
