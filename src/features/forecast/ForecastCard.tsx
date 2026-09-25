import {type TouchEvent, useEffect, useMemo, useRef, useState} from "react";
import {Box, Button, CircularProgress, Collapse, Stack, Typography} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import {AnimatePresence, motion, useReducedMotion} from "framer-motion";
import ModernCard from "../../shared/components/ModernCard";
import type {ForecastDay} from "../../lib/types/forecast";
import type {FacilityHoursFacilityPayload} from "../../lib/types/facilitySchedule";
import {INNER_SURFACE_SX, type OccupancyThresholds} from "../../shared/utils/styles";
import {getFacilityOpenWindowsForDate} from "../../shared/utils/facilityScheduleStatus";
import {
    BAND_LEVEL_ORDER,
    BAND_STYLES,
    buildCrowdBandsFromDisplaySlots,
    EMPTY_CROWD_BANDS,
    isHistogramLevelVisible,
    sortBands,
    type CrowdBandLevel,
} from "./forecastBands";
import {
    buildForecastDisplaySlots,
    clipBandsToWorkingHours,
    filterWindowsToWorkingHours,
    getChicagoDateKeyFromTimestamp,
    getDateKeyDayIndex,
    parseShortDate,
} from "./forecastTime";
import {buildHistogramModel} from "./forecastHistogram";
import ForecastChart from "./ForecastChart";
import ForecastDayControls from "./ForecastDayControls";
import ForecastWindowsList from "./ForecastWindowsList";

interface Props {
    day: ForecastDay | null;
    comparisonDays?: ForecastDay[];
    schedule?: FacilityHoursFacilityPayload | null;
    occupancyThresholds?: OccupancyThresholds | null;
    dayOffset: number;
    titleDayOffset?: number;
    totalDays: number;
    canPrev: boolean;
    canNext: boolean;
    onPrev: () => void;
    onNext: () => void;
    isLoading: boolean;
    error: string | null;
}

const SWIPE_THRESHOLD_PX = 52;
const SWIPE_VERTICAL_TOLERANCE_PX = 24;
const SWIPE_INTENT_THRESHOLD_PX = 12;
const SWIPE_INTENT_BIAS_PX = 6;

const dayContentVariants = {
    enter: (direction: 1 | -1) => ({
        x: direction > 0 ? 36 : -36,
        opacity: 0,
    }),
    center: {
        x: 0,
        opacity: 1,
    },
    exit: (direction: 1 | -1) => ({
        x: direction > 0 ? -28 : 28,
        opacity: 0,
    }),
};

export default function ForecastWindowsCard({
    day,
    comparisonDays = [],
    schedule = null,
    occupancyThresholds = null,
    dayOffset,
    titleDayOffset,
    totalDays,
    canPrev,
    canNext,
    onPrev,
    onNext,
    isLoading,
    error,
}: Props) {
    const theme = useTheme();
    const isDark = theme.palette.mode === "dark";
    const neutralButtonBg = isDark ? alpha(theme.palette.common.white, 0.06) : "#ffffff";
    const neutralButtonBorder = alpha(theme.palette.text.primary, isDark ? 0.32 : 0.22);
    const neutralButtonHoverBg = isDark ? alpha(theme.palette.common.white, 0.1) : "rgba(15, 23, 42, 0.03)";
    const reduceMotion = useReducedMotion();
    const [selectedLevels, setSelectedLevels] = useState<CrowdBandLevel[]>([]);
    const [nowTs, setNowTs] = useState<number>(() => Date.now());
    const [slideDirection, setSlideDirection] = useState<1 | -1>(1);
    const [trendExpanded, setTrendExpanded] = useState(false);
    const [selectedHistogramSelection, setSelectedHistogramSelection] = useState<{
        date: string | null;
        startTs: number | null;
    }>({date: null, startTs: null});
    const swipeAreaRef = useRef<HTMLDivElement | null>(null);
    const touchStartRef = useRef<{x: number; y: number} | null>(null);
    const swipeIntentRef = useRef<"pending" | "horizontal" | "vertical">("pending");
    const horizontalLockRef = useRef(false);

    useEffect(() => {
        const timerId = window.setInterval(() => {
            setNowTs(Date.now());
        }, 30000);
        return () => window.clearInterval(timerId);
    }, []);

    const toggleLevel = (level: CrowdBandLevel) => {
        setSelectedLevels((prev) => (
            prev.includes(level)
                ? prev.filter((item) => item !== level)
                : [...prev, level]
        ));
    };

    const openWindows = useMemo(
        () => getFacilityOpenWindowsForDate(schedule, day?.date ?? null),
        [schedule, day?.date]
    );

    const hasWorkingHours = Boolean(schedule && day?.date);
    const fallbackCrowdBands = day?.crowdBands ?? EMPTY_CROWD_BANDS;

    const displaySlots = useMemo(
        () => buildForecastDisplaySlots(
            day,
            openWindows,
            hasWorkingHours,
            occupancyThresholds,
            fallbackCrowdBands,
            nowTs
        ),
        [day, fallbackCrowdBands, hasWorkingHours, nowTs, occupancyThresholds, openWindows]
    );

    const slotDerivedBands = useMemo(
        () => buildCrowdBandsFromDisplaySlots(displaySlots),
        [displaySlots]
    );
    const histogramScaleMaxCount = useMemo(() => {
        const daysForScale = comparisonDays.length > 0 ? comparisonDays : (day ? [day] : []);
        return daysForScale.reduce((globalMax, comparisonDay) => {
            const comparisonOpenWindows = getFacilityOpenWindowsForDate(schedule, comparisonDay?.date ?? null);
            const comparisonHasWorkingHours = Boolean(schedule && comparisonDay?.date);
            const comparisonFallbackCrowdBands = comparisonDay?.crowdBands ?? EMPTY_CROWD_BANDS;
            const comparisonDisplaySlots = buildForecastDisplaySlots(
                comparisonDay,
                comparisonOpenWindows,
                comparisonHasWorkingHours,
                occupancyThresholds,
                comparisonFallbackCrowdBands,
                nowTs
            );
            const comparisonHistogram = buildHistogramModel(comparisonDisplaySlots);
            return Math.max(globalMax, comparisonHistogram?.maxCount ?? 0);
        }, 0);
    }, [comparisonDays, day, schedule, occupancyThresholds, nowTs]);

    const workingHoursBands = useMemo(() => {
        if (displaySlots.length > 0) {
            return slotDerivedBands;
        }
        return clipBandsToWorkingHours(fallbackCrowdBands, openWindows, hasWorkingHours, day?.date ?? null);
    }, [day?.date, displaySlots.length, fallbackCrowdBands, hasWorkingHours, openWindows, slotDerivedBands]);

    const displayBands = useMemo(() => {
        const allBands = workingHoursBands;
        const showDefault =
            selectedLevels.length === 0 || selectedLevels.length === BAND_LEVEL_ORDER.length;

        if (showDefault) {
            return sortBands(allBands);
        }

        const selected = new Set(selectedLevels);
        return sortBands(allBands).filter((band) => selected.has(band.level));
    }, [workingHoursBands, selectedLevels]);
    const showAllHistogramLevels =
        workingHoursBands.length === 0
        || selectedLevels.length === 0
        || selectedLevels.length === BAND_LEVEL_ORDER.length;
    const selectedLevelSet = useMemo(
        () => new Set(selectedLevels),
        [selectedLevels]
    );

    const histogram = useMemo(
        () => buildHistogramModel(displaySlots, histogramScaleMaxCount),
        [displaySlots, histogramScaleMaxCount]
    );

    const filteredBestWindows = useMemo(
        () => displaySlots.length > 0 ? [] : filterWindowsToWorkingHours(day?.bestWindows ?? [], openWindows, hasWorkingHours, day?.date ?? null),
        [day?.bestWindows, day?.date, displaySlots.length, openWindows, hasWorkingHours]
    );

    const filteredAvoidWindows = useMemo(
        () => displaySlots.length > 0 ? [] : filterWindowsToWorkingHours(day?.avoidWindows ?? [], openWindows, hasWorkingHours, day?.date ?? null),
        [day?.avoidWindows, day?.date, displaySlots.length, openWindows, hasWorkingHours]
    );
    const selectedHistogramStartTs = (
        selectedHistogramSelection.date === (day?.date ?? null)
            ? selectedHistogramSelection.startTs
            : null
    );
    const selectedHistogramBar = useMemo(
        () => (
            histogram?.bars.find((bar) => {
                if (bar.startTs !== selectedHistogramStartTs) {
                    return false;
                }
                return isHistogramLevelVisible(bar.level, selectedLevelSet, showAllHistogramLevels);
            }) ?? null
        ),
        [histogram, selectedHistogramStartTs, selectedLevelSet, showAllHistogramLevels]
    );
    const currentTimeMarker = useMemo(() => {
        if (!histogram || !day?.date) return null;

        const todayDateKey = getChicagoDateKeyFromTimestamp(nowTs);
        if (!todayDateKey || day.date !== todayDateKey) return null;

        const activeBar = histogram.bars.find((bar) => nowTs >= bar.startTs && nowTs < bar.endTs);
        if (!activeBar) return null;
        if (!isHistogramLevelVisible(activeBar.level, selectedLevelSet, showAllHistogramLevels)) return null;

        const elapsed = nowTs - activeBar.startTs;
        const slotDuration = Math.max(1, activeBar.endTs - activeBar.startTs);
        return {
            x: activeBar.x + ((elapsed / slotDuration) * activeBar.width),
            label: "Now",
        };
    }, [day, histogram, nowTs, selectedLevelSet, showAllHistogramLevels]);

    const shortDate = parseShortDate(day?.date);
    const shortWeekday = day?.dayName ? day.dayName.slice(0, 3) : null;
    const slideVariants = useMemo(
        () => (reduceMotion
            ? {
                enter: {x: 0, opacity: 1},
                center: {x: 0, opacity: 1},
                exit: {x: 0, opacity: 1},
            }
            : dayContentVariants),
        [reduceMotion]
    );
    const relativeDayOffset = (() => {
        const todayDateKey = getChicagoDateKeyFromTimestamp(nowTs);
        const todayIndex = getDateKeyDayIndex(todayDateKey);
        const dayIndex = getDateKeyDayIndex(day?.date);
        if (todayIndex === null || dayIndex === null) return titleDayOffset ?? dayOffset;
        return dayIndex - todayIndex;
    })();
    const title = (() => {
        if (relativeDayOffset === 0) return "Forecast Today";
        if (relativeDayOffset === 1) return "Forecast Tomorrow";
        if (shortWeekday && shortDate) return `Forecast ${shortWeekday} ${shortDate}`;
        if (shortWeekday) return `Forecast ${shortWeekday}`;
        if (shortDate) return `Forecast ${shortDate}`;
        return relativeDayOffset === 2 ? "Forecast In 2 Days" : "Forecast In 3 Days";
    })();

    const handlePrevDay = () => {
        if (!canPrev || isLoading) return;
        setSlideDirection(-1);
        onPrev();
    };

    const handleNextDay = () => {
        if (!canNext || isLoading) return;
        setSlideDirection(1);
        onNext();
    };

    const handleTouchStart = (event: TouchEvent<HTMLDivElement>) => {
        if (isLoading || event.touches.length !== 1) return;
        const touch = event.touches[0];
        touchStartRef.current = {x: touch.clientX, y: touch.clientY};
        swipeIntentRef.current = "pending";
        horizontalLockRef.current = false;
        const node = swipeAreaRef.current;
        if (node) node.style.touchAction = "";
    };

    const handleTouchMove = (event: TouchEvent<HTMLDivElement>) => {
        const start = touchStartRef.current;
        if (!start || event.touches.length !== 1) return;

        const touch = event.touches[0];
        const deltaX = touch.clientX - start.x;
        const deltaY = touch.clientY - start.y;
        const absX = Math.abs(deltaX);
        const absY = Math.abs(deltaY);

        if (swipeIntentRef.current === "pending") {
            const movedEnough = absX >= SWIPE_INTENT_THRESHOLD_PX || absY >= SWIPE_INTENT_THRESHOLD_PX;
            if (!movedEnough) return;

            if (absX > absY + SWIPE_INTENT_BIAS_PX) {
                swipeIntentRef.current = "horizontal";
                if (!horizontalLockRef.current) {
                    horizontalLockRef.current = true;
                    const node = swipeAreaRef.current;
                    if (node) node.style.touchAction = "none";
                }
            } else if (absY > absX + SWIPE_INTENT_BIAS_PX) {
                swipeIntentRef.current = "vertical";
                if (horizontalLockRef.current) {
                    horizontalLockRef.current = false;
                    const node = swipeAreaRef.current;
                    if (node) node.style.touchAction = "";
                }
            } else {
                return;
            }
        }

        if (swipeIntentRef.current === "horizontal") {
            event.preventDefault();
        }
    };

    const resetTouchGesture = () => {
        touchStartRef.current = null;
        swipeIntentRef.current = "pending";
        horizontalLockRef.current = false;
        const node = swipeAreaRef.current;
        if (node) node.style.touchAction = "";
    };

    const handleTouchEnd = (event: TouchEvent<HTMLDivElement>) => {
        const start = touchStartRef.current;
        const swipeIntent = swipeIntentRef.current;
        resetTouchGesture();
        if (!start || event.changedTouches.length !== 1) return;
        if (swipeIntent !== "horizontal") return;

        const touch = event.changedTouches[0];
        const deltaX = touch.clientX - start.x;
        const deltaY = touch.clientY - start.y;
        const absX = Math.abs(deltaX);
        const absY = Math.abs(deltaY);

        if (absX < SWIPE_THRESHOLD_PX || absX < absY + SWIPE_VERTICAL_TOLERANCE_PX) {
            return;
        }

        if (deltaX < 0) {
            handleNextDay();
            return;
        }

        handlePrevDay();
    };

    const toggleHistogramSlotSelection = (startTs: number) => {
        const currentDate = day?.date ?? null;
        setSelectedHistogramSelection((prev) => {
            const previousStartTsForCurrentDay = prev.date === currentDate ? prev.startTs : null;
            return {
                date: currentDate,
                startTs: previousStartTsForCurrentDay === startTs ? null : startTs,
            };
        });
    };

    const hasForecastContent = workingHoursBands.length > 0
        || filteredBestWindows.length > 0
        || filteredAvoidWindows.length > 0
        || histogram !== null;

    if (!isLoading && (Boolean(error) || !day || !hasForecastContent)) {
        return null;
    }

    return (
        <ModernCard>
            <Box sx={{display: "flex", alignItems: "center", justifyContent: "space-between"}}>
                <Typography variant="subtitle2" color="text.secondary">
                    {title}
                </Typography>
                <ForecastDayControls
                    variant="buttons"
                    canPrev={canPrev}
                    canNext={canNext}
                    isLoading={isLoading}
                    onPrev={handlePrevDay}
                    onNext={handleNextDay}
                    totalDays={totalDays}
                    dayOffset={dayOffset}
                />
            </Box>

            {isLoading && (
                <Box sx={{display: "flex", alignItems: "center", gap: 1, mt: 1}}>
                    <CircularProgress size={16} thickness={5}/>
                    <Typography variant="body2" color="text.secondary">
                        Loading forecast...
                    </Typography>
                </Box>
            )}

            <Box
                ref={swipeAreaRef}
                data-disable-pull-refresh="true"
                onTouchStartCapture={handleTouchStart}
                onTouchMoveCapture={handleTouchMove}
                onTouchEndCapture={handleTouchEnd}
                onTouchCancelCapture={resetTouchGesture}
                sx={{
                    mt: 1,
                    touchAction: {xs: "pan-y", sm: "auto"},
                    overflowX: "hidden",
                    overscrollBehaviorX: "contain",
                    position: "relative",
                }}
            >
                <AnimatePresence mode="wait" custom={slideDirection} initial={false}>
                    {!error && day && (
                        <Box
                            key={day.date || `offset-${dayOffset}`}
                            component={motion.div}
                            custom={slideDirection}
                            variants={slideVariants}
                            initial="enter"
                            animate="center"
                            exit="exit"
                            transition={reduceMotion ? {duration: 0} : {duration: 0.24, ease: [0.22, 1, 0.36, 1]}}
                        >
                            {workingHoursBands.length > 0 && (
                                <Stack direction="row" spacing={0.75} sx={{mb: 1, flexWrap: "wrap", rowGap: 0.75}}>
                                    {BAND_LEVEL_ORDER.map((level) => {
                                        const active = selectedLevels.includes(level);
                                        const style = BAND_STYLES[level];
                                        return (
                                            <Button
                                                key={level}
                                                size="small"
                                                variant="outlined"
                                                onClick={() => toggleLevel(level)}
                                                sx={{
                                                    textTransform: "uppercase",
                                                    fontWeight: 700,
                                                    borderRadius: 999,
                                                    px: 1.15,
                                                    minWidth: 88,
                                                    minHeight: 44,
                                                    bgcolor: active ? style.bg : neutralButtonBg,
                                                    borderColor: active
                                                        ? alpha(style.color, 0.6)
                                                        : neutralButtonBorder,
                                                    color: active ? style.color : "text.secondary",
                                                    fontSize: "0.68rem",
                                                    "@media (hover: hover) and (pointer: fine)": {
                                                        "&:hover": {
                                                            borderColor: active
                                                                ? alpha(style.color, 0.7)
                                                                : alpha(theme.palette.text.primary, isDark ? 0.46 : 0.36),
                                                            bgcolor: active
                                                                ? alpha(style.color, isDark ? 0.2 : 0.16)
                                                                : neutralButtonHoverBg,
                                                        },
                                                    },
                                                }}
                                            >
                                                {style.label.replace(" CROWD", "")}
                                            </Button>
                                        );
                                    })}
                                </Stack>
                            )}
                            <ForecastWindowsList
                                workingHoursBands={workingHoursBands}
                                displayBands={displayBands}
                                filteredBestWindows={filteredBestWindows}
                                filteredAvoidWindows={filteredAvoidWindows}
                                nowTs={nowTs}
                                isDark={isDark}
                            />
                            {histogram && (
                                <Box sx={{mt: 1}}>
                                    <Button
                                        size="small"
                                        variant="outlined"
                                        onClick={() => setTrendExpanded((prev) => !prev)}
                                        endIcon={
                                            <ExpandMoreIcon
                                                sx={{
                                                    transform: trendExpanded ? "rotate(180deg)" : "rotate(0deg)",
                                                    transition: "transform 180ms ease",
                                                }}
                                            />
                                        }
                                        sx={{
                                            borderRadius: 999,
                                            textTransform: "none",
                                            fontWeight: 700,
                                            px: 1.2,
                                            color: "text.secondary",
                                            borderColor: "divider",
                                        }}
                                    >
                                        {trendExpanded ? "Hide crowd chart" : "Show crowd chart"}
                                    </Button>
                                    <Collapse in={trendExpanded} timeout={180} unmountOnExit>
                                        <Box
                                            sx={{
                                                ...INNER_SURFACE_SX,
                                                mt: 1,
                                                p: 1.2,
                                                borderColor: "divider",
                                                boxShadow: "none",
                                            }}
                                        >
                                            <Typography variant="caption" color="text.secondary" sx={{fontWeight: 700}}>
                                                People by hour
                                            </Typography>
                                            <Box sx={{mt: 0.75, overflowX: "hidden", pb: 0.6}}>
                                                <ForecastChart
                                                    histogram={histogram}
                                                    selectedLevelSet={selectedLevelSet}
                                                    showAllHistogramLevels={showAllHistogramLevels}
                                                    selectedStartTs={selectedHistogramStartTs}
                                                    onToggleBar={toggleHistogramSlotSelection}
                                                    currentTimeMarker={currentTimeMarker}
                                                />
                                                <Typography variant="body2" color="text.secondary" sx={{display: "block", mt: 0.4, fontWeight: 700}}>
                                                    Max people: {Math.round(histogram.maxCount)}
                                                </Typography>
                                                {selectedHistogramBar && (
                                                    <Typography variant="body2" color="text.primary" sx={{display: "block", mt: 0.25, fontWeight: 700}}>
                                                        {selectedHistogramBar.rangeLabel}: {Math.round(selectedHistogramBar.count)}
                                                    </Typography>
                                                )}
                                            </Box>
                                        </Box>
                                    </Collapse>
                                </Box>
                            )}
                        </Box>
                    )}
                </AnimatePresence>
            </Box>

            <ForecastDayControls
                variant="pagination"
                canPrev={canPrev}
                canNext={canNext}
                isLoading={isLoading}
                onPrev={handlePrevDay}
                onNext={handleNextDay}
                totalDays={totalDays}
                dayOffset={dayOffset}
            />
        </ModernCard>
    );
}
