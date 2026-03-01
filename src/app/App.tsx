import {useCallback, useEffect, useMemo, useState} from "react";
import {
    Alert,
    Box,
    CircularProgress,
    Container,
    IconButton,
    Link,
    Stack,
    Tooltip,
    Typography,
} from "@mui/material";
import RefreshIcon from "@mui/icons-material/Refresh";
import GitHubIcon from "@mui/icons-material/GitHub";
import FacilitySelector from "../facilities/FacilitySelector";
import OccupancyCard from "../facilities/OccupancyCard";
import SectionSummary from "../facilities/SectionSummary";
import SectionSummaryOther from "../facilities/SectionSummaryOther";
import ForecastWindowsCard from "../facilities/ForecastWindowsCard";
import {fetchFacility} from "../lib/api/facilityParser";
import {fetchForecastDays} from "../lib/api/forecastParser";
import {
    FACILITY_DASHBOARD_CONFIG,
    FACILITY_KNOWN_IDS,
    isSectionRow,
    type SectionConfig,
    type SectionLayout,
} from "../facilities/constants";
import {getFacilityCache, setFacilityCache} from "../lib/storage/facilityCache";
import type {FacilityId, FacilityPayload} from "../lib/types/facility";
import type {ForecastDay, ForecastHour} from "../lib/types/forecast";
import {
    formatChicagoDateTime,
    getChicagoDayAge,
    getChicagoTimestampMs,
} from "../shared/utils/chicagoTime";

const FACILITY_STORAGE_KEY = "reclive:selectedFacility";
const AUTO_REFRESH_INTERVAL_MS = 15 * 60 * 1000;
const AUTO_REFRESH_CHECK_INTERVAL_MS = 60 * 1000;
const MANUAL_REFRESH_COOLDOWN_MS = 3000;
const FORECAST_VISIBLE_SECTIONS = new Set(["fitness floors", "basketball courts"]);

const getStoredFacility = (): FacilityId => {
    if (typeof window === "undefined") return 1186;
    const stored = Number(window.localStorage.getItem(FACILITY_STORAGE_KEY));
    return stored === 1656 ? 1656 : 1186;
};

const getLatestTimestamp = (payload: FacilityPayload | null | undefined): string | null => {
    if (!payload) return null;

    let latest: string | null = null;
    for (const location of payload.locations) {
        if (location.lastUpdated && (!latest || location.lastUpdated > latest)) {
            latest = location.lastUpdated;
        }
    }

    return latest;
};

const normalizeSectionTitle = (title: string): string =>
    title.replace(/^[^a-zA-Z0-9]+/, "").replace(/\s+/g, " ").trim().toLowerCase();

const getDaysAgoFromIso = (lastUpdated: string | null | undefined): number | null => {
    return getChicagoDayAge(lastUpdated);
};

const buildSectionForecastMap = (
    day: ForecastDay | null,
    hideForecast: boolean
): Record<string, ForecastHour[]> => {
    if (hideForecast) return {};
    if (!day?.categories) return {};

    const nowTs = Date.now();
    const sectionMap: Record<string, ForecastHour[]> = {};

    for (const category of day.categories) {
        const key = normalizeSectionTitle(category.title);
        if (!FORECAST_VISIBLE_SECTIONS.has(key)) {
            continue;
        }

        const sorted = [...(category.hours ?? [])].sort(
            (a, b) => {
                const leftTs = getChicagoTimestampMs(a.hourStart);
                const rightTs = getChicagoTimestampMs(b.hourStart);
                if (leftTs === null && rightTs === null) return 0;
                if (leftTs === null) return 1;
                if (rightTs === null) return -1;
                return leftTs - rightTs;
            }
        );
        sectionMap[key] = sorted
            .filter((hour) => {
                const hourTs = getChicagoTimestampMs(hour.hourStart);
                return hourTs !== null && hourTs > nowTs;
            })
            .slice(0, 3);
    }

    return sectionMap;
};

const renderSection = (
    section: SectionConfig,
    locations: FacilityPayload["locations"],
    forecastMap: Record<string, ForecastHour[]>
) => (
    <SectionSummary
        key={section.title}
        title={section.title}
        ids={[...section.ids]}
        locations={locations}
        forecast={forecastMap[normalizeSectionTitle(section.title)]}
    />
);

const renderSectionLayout = (
    layout: SectionLayout,
    locations: FacilityPayload["locations"],
    index: number,
    forecastMap: Record<string, ForecastHour[]>
) => {
    if (!isSectionRow(layout)) {
        return renderSection(layout, locations, forecastMap);
    }

    return (
        <Stack key={`row-${index}`} direction={{xs: "column", sm: "row"}} spacing={2} alignItems="stretch">
            {layout.map((section) => (
                <Box key={section.title} sx={{flex: 1, minWidth: 0}}>
                    <SectionSummary
                        title={section.title}
                        ids={[...section.ids]}
                        locations={locations}
                        forecast={forecastMap[normalizeSectionTitle(section.title)]}
                    />
                </Box>
            ))}
        </Stack>
    );
};

export default function App() {
    const [facility, setFacility] = useState<FacilityId>(getStoredFacility);
    const [refreshKey, setRefreshKey] = useState(0);
    const [data, setData] = useState<FacilityPayload | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [freshness, setFreshness] = useState<"live" | "cached" | null>(null);
    const [lastAutoRefresh, setLastAutoRefresh] = useState(() => Date.now());
    const [lastManualRefresh, setLastManualRefresh] = useState(0);
    const [forecastDays, setForecastDays] = useState<ForecastDay[]>([]);
    const [forecastDayOffset, setForecastDayOffset] = useState(0);
    const [forecastError, setForecastError] = useState<string | null>(null);
    const [isForecastLoading, setIsForecastLoading] = useState(false);

    useEffect(() => {
        if (typeof window === "undefined") return;
        try {
            window.localStorage.setItem(FACILITY_STORAGE_KEY, String(facility));
        } catch {
            // Ignore storage write failures (private mode/quota exceeded).
        }
    }, [facility]);

    useEffect(() => {
        const controller = new AbortController();
        let isCancelled = false;

        const cached = getFacilityCache(facility);
        if (cached) {
            setData(cached.payload);
            setFreshness(null);
        } else {
            setData(null);
            setFreshness(null);
        }

        const load = async () => {
            setIsLoading(true);
            setError(null);
            setFreshness(null);

            try {
                const payload = await fetchFacility(facility, controller.signal);
                if (isCancelled || controller.signal.aborted) return;

                setData(payload);
                setError(null);
                setFreshness("live");
                setFacilityCache(facility, payload);
            } catch (loadError) {
                if (isCancelled || controller.signal.aborted) return;

                console.error("Failed to fetch facility data", loadError);
                const fallback = getFacilityCache(facility);
                if (fallback) {
                    setData(fallback.payload);
                    setFreshness("cached");
                    setError(null);
                } else {
                    setData(null);
                    setFreshness(null);
                    setError("Data unavailable right now. Please try again shortly.");
                }
            } finally {
                if (!isCancelled && !controller.signal.aborted) {
                    setIsLoading(false);
                }
            }
        };

        void load();

        return () => {
            isCancelled = true;
            controller.abort();
        };
    }, [facility, refreshKey]);

    useEffect(() => {
        const controller = new AbortController();
        let isCancelled = false;

        const loadForecast = async () => {
            setIsForecastLoading(true);
            setForecastError(null);

            try {
                const days = await fetchForecastDays(facility, controller.signal);
                if (isCancelled || controller.signal.aborted) return;
                setForecastDays(days);
                setForecastDayOffset(0);
            } catch (loadError) {
                if (isCancelled || controller.signal.aborted) return;
                console.error("Failed to fetch forecast data", loadError);
                setForecastDays([]);
                setForecastDayOffset(0);
                setForecastError("Forecast unavailable right now.");
            } finally {
                if (!isCancelled && !controller.signal.aborted) {
                    setIsForecastLoading(false);
                }
            }
        };

        void loadForecast();

        return () => {
            isCancelled = true;
            controller.abort();
        };
    }, [facility, refreshKey]);

    const triggerRefresh = useCallback((action: () => void) => {
        setIsLoading(true);
        setError(null);
        setFreshness(null);
        setLastAutoRefresh(Date.now());
        action();
    }, []);

    useEffect(() => {
        if (typeof window === "undefined") return;

        const interval = window.setInterval(() => {
            if (!isLoading && Date.now() - lastAutoRefresh >= AUTO_REFRESH_INTERVAL_MS) {
                triggerRefresh(() => setRefreshKey((key) => key + 1));
            }
        }, AUTO_REFRESH_CHECK_INTERVAL_MS);

        return () => {
            window.clearInterval(interval);
        };
    }, [isLoading, lastAutoRefresh, triggerRefresh]);

    const handleFacilitySelect = (next: FacilityId) => {
        if (next === facility) return;
        setLastManualRefresh(0);
        setForecastDayOffset(0);
        triggerRefresh(() => setFacility(next));
    };

    const total = data
        ? data.locations.reduce((sum, l) => sum + (l.currentCapacity ?? 0), 0)
        : 0;

    const max = data
        ? data.locations.reduce((sum, l) => sum + (l.maxCapacity ?? 0), 0)
        : 0;

    const lastUpdated = getLatestTimestamp(data);
    const daysAgo = getDaysAgoFromIso(lastUpdated);
    const showStaleNotice = !isLoading && typeof daysAgo === "number" && daysAgo >= 1;

    const manualRefresh = () => {
        if (isLoading) return;
        const now = Date.now();
        if (now - lastManualRefresh < MANUAL_REFRESH_COOLDOWN_MS) return;
        setLastManualRefresh(now);
        triggerRefresh(() => setRefreshKey((key) => key + 1));
    };

    const dashboardConfig = FACILITY_DASHBOARD_CONFIG[facility];
    const knownIds = FACILITY_KNOWN_IDS[facility];
    const visibleForecastDays = useMemo(
        () => forecastDays.slice(0, 4),
        [forecastDays]
    );
    const todayForecastDay = visibleForecastDays[0] ?? null;
    const selectedForecastDay =
        visibleForecastDays[forecastDayOffset] ?? todayForecastDay;

    useEffect(() => {
        if (forecastDayOffset > Math.max(0, visibleForecastDays.length - 1)) {
            setForecastDayOffset(0);
        }
    }, [forecastDayOffset, visibleForecastDays.length]);

    const sectionForecastMap = useMemo(
        () => buildSectionForecastMap(todayForecastDay, showStaleNotice),
        [todayForecastDay, showStaleNotice]
    );

    const formattedCachedTime = (() => {
        return formatChicagoDateTime(lastUpdated);
    })();

    return (
        <Box sx={{py: {xs: 2, sm: 3}, bgcolor: "background.default", minHeight: "100vh"}}>
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
                <Box sx={{display: "flex", alignItems: "center", gap: 1}}>
                    <Box sx={{flexGrow: 1}}>
                        <FacilitySelector facility={facility} onSelect={handleFacilitySelect}/>
                    </Box>
                    <Tooltip title="Refresh">
                        <span>
                            <IconButton
                                onClick={manualRefresh}
                                disabled={isLoading}
                                aria-label="Refresh occupancy data"
                            >
                                {isLoading ? (
                                    <CircularProgress size={18} thickness={5}/>
                                ) : (
                                    <RefreshIcon fontSize="small"/>
                                )}
                            </IconButton>
                        </span>
                    </Tooltip>
                </Box>

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

                {error && (
                    <Alert severity="warning" sx={{borderRadius: 2}}>
                        {error}
                    </Alert>
                )}

                {data && (
                    <>
                        <OccupancyCard
                            title="Live Occupancy"
                            total={total}
                            max={max}
                            lastUpdated={lastUpdated}
                            facilityId={facility}
                            isLoading={isLoading}
                        />

                        {!showStaleNotice && (
                            <ForecastWindowsCard
                                day={selectedForecastDay}
                                dayOffset={forecastDayOffset}
                                canPrev={forecastDayOffset > 0}
                                canNext={forecastDayOffset < Math.min(3, visibleForecastDays.length - 1)}
                                onPrev={() => setForecastDayOffset((prev) => Math.max(0, prev - 1))}
                                onNext={() =>
                                    setForecastDayOffset((prev) =>
                                        Math.min(Math.min(3, visibleForecastDays.length - 1), prev + 1)
                                    )
                                }
                                isLoading={isForecastLoading}
                                error={forecastError}
                            />
                        )}

                        {freshness === "cached" && (
                            <Alert severity="info" variant="outlined" sx={{borderRadius: 2}}>
                                Showing cached data
                                {formattedCachedTime ? ` from ${formattedCachedTime}` : ""}. We'll
                                refresh again when the service is back.
                            </Alert>
                        )}

                        {dashboardConfig.sections.map((layout, index) =>
                            renderSectionLayout(layout, data.locations, index, sectionForecastMap)
                        )}

                        <SectionSummaryOther
                            title={dashboardConfig.otherTitle}
                            exclude={knownIds}
                            locations={data.locations}
                        />
                    </>
                )}

                <Box component="footer" sx={{py: 2}}>
                    <Stack spacing={1} alignItems="center" justifyContent="center">
                        <Stack direction="row" spacing={0.5} alignItems="center" justifyContent="center">
                            <Typography color="text.secondary" sx={{fontSize: "0.85rem"}}>
                                Built by{" "}
                                <Link href="https://anton.grachev.us" target="_blank" rel="noopener noreferrer" underline="hover">
                                    Anton
                                </Link>
                            </Typography>
                            <Tooltip title="Source code">
                                <IconButton
                                    size="small"
                                    component="a"
                                    href="https://github.com/ant0n-grachev/RecLive"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    aria-label="View source code on GitHub"
                                >
                                    <GitHubIcon fontSize="small"/>
                                </IconButton>
                            </Tooltip>
                        </Stack>

                        <Box
                            component="img"
                            src="https://api.netlify.com/api/v1/badges/d0d6bc80-46d8-402f-bcc3-7047df4d4b52/deploy-status"
                            alt="Netlify Status"
                            sx={{display: "block"}}
                        />
                    </Stack>
                </Box>
            </Container>
        </Box>
    );
}
