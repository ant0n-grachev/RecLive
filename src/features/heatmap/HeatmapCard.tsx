import {useCallback, useEffect, useMemo, useState} from "react";
import {
    Box,
    Button,
    ClickAwayListener,
    Collapse,
    Paper,
    Popper,
    Stack,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import type {FacilityId, Location} from "../../lib/types/facility";
import ModernCard from "../../shared/components/ModernCard";
import type {OccupancyThresholds} from "../../shared/utils/styles";
import {debugControlsEnabled} from "../../app/debugOverrides";
import {HeatmapSvg} from "./HeatmapSvg";
import {
    buildFloorRenderData,
    EMPTY_RENDER_DATA,
    getZonePresentation,
    HEATMAP_ZONE_DIALOG_ID,
    HEATMAP_ZONE_DIALOG_TITLE_ID,
} from "./heatmapModel";

declare global {
    interface Window {
        heatmapdebug?: () => void;
    }
}

interface Props {
    facilityId: FacilityId;
    locations: Location[];
    nowTs: number;
    occupancyThresholds?: OccupancyThresholds | null;
    locationOccupancyThresholds?: Partial<Record<number, OccupancyThresholds>>;
}

const DEFAULT_DEBUG_COORDS = false;
const floorLabel = (floor: number): string => (floor === 0 ? "Lower" : `Floor ${floor}`);

export default function HeatmapCard({
    facilityId,
    locations,
    nowTs,
    occupancyThresholds = null,
    locationOccupancyThresholds = {},
}: Props) {
    const theme = useTheme();
    const isDark = theme.palette.mode === "dark";
    const floors = useMemo(() => (
        [...new Set(locations.map((loc) => loc.floor))]
            .sort((a, b) => a - b)
    ), [locations]);

    const [showDebugCoords, setShowDebugCoords] = useState(DEFAULT_DEBUG_COORDS);
    const [expanded, setExpanded] = useState(false);
    const [selectedFloor, setSelectedFloor] = useState<number>(floors[0] ?? 0);
    const [selectedZoneInfo, setSelectedZoneInfo] = useState<{
        key: string;
        trigger: SVGPolygonElement;
    } | null>(null);
    const effectiveSelectedFloor = floors.includes(selectedFloor) ? selectedFloor : (floors[0] ?? 0);
    const neutralControlBg = isDark ? alpha(theme.palette.common.white, 0.06) : theme.palette.background.paper;
    const neutralControlBorder = alpha(theme.palette.text.primary, isDark ? 0.32 : 0.14);
    const neutralControlHoverBg = isDark
        ? alpha(theme.palette.common.white, 0.1)
        : alpha(theme.palette.text.primary, 0.03);
    const activeControlBg = isDark
        ? alpha(theme.palette.common.white, 0.15)
        : alpha(theme.palette.text.primary, 0.08);
    const activeControlBorder = alpha(theme.palette.text.primary, isDark ? 0.56 : 0.45);
    const activeControlHoverBg = isDark
        ? alpha(theme.palette.common.white, 0.19)
        : alpha(theme.palette.text.primary, 0.12);
    const activeControlShadow = isDark
        ? "0 1px 2px rgba(2, 6, 23, 0.42), 0 8px 18px rgba(2, 6, 23, 0.28)"
        : "0 1px 2px rgba(0, 0, 0, 0.08), 0 8px 18px rgba(0, 0, 0, 0.05)";
    const debugEnabled = typeof window !== "undefined"
        && debugControlsEnabled(import.meta.env, window.location.hostname);

    useEffect(() => {
        if (typeof window === "undefined") return;
        if (!debugEnabled) {
            delete window.heatmapdebug;
            return;
        }

        window.heatmapdebug = () => {
            setShowDebugCoords((prev) => {
                const next = !prev;
                console.info(`[reclive] heatmap debug ${next ? "enabled" : "disabled"}`);
                return next;
            });
        };
        console.info(
            "[reclive] debug commands: window.heatmapdebug(), window.recliveShowPredictions(), window.recliveRestoreWarnings(), window.reclivePredictionOverrideStatus(), window.recliveOverrideClosure(), window.recliveRestoreClosure(), window.recliveClosureOverrideStatus(), window.recliveSetDebugNow('2026-05-18T23:00:00'), window.recliveClearDebugNow()"
        );

        return () => {
            delete window.heatmapdebug;
        };
    }, [debugEnabled]);

    const singleFloorData = useMemo(
        () => {
            if (!expanded) {
                return EMPTY_RENDER_DATA;
            }
            return buildFloorRenderData(
                facilityId,
                effectiveSelectedFloor,
                locations,
                nowTs,
                occupancyThresholds,
                locationOccupancyThresholds
            );
        },
        [expanded, facilityId, effectiveSelectedFloor, locations, nowTs, occupancyThresholds, locationOccupancyThresholds]
    );
    const selectedZoneSummary = selectedZoneInfo
        ? singleFloorData.zoneSummaries.find((zoneSummary) => zoneSummary.key === selectedZoneInfo.key) ?? null
        : null;
    const selectedZonePresentation = selectedZoneSummary
        ? getZonePresentation(selectedZoneSummary, occupancyThresholds)
        : null;
    const closeSelectedZone = useCallback(() => {
        const trigger = selectedZoneInfo?.trigger;
        setSelectedZoneInfo(null);
        if (trigger) {
            requestAnimationFrame(() => trigger.focus());
        }
    }, [selectedZoneInfo]);

    useEffect(() => {
        if (!selectedZoneInfo) return;

        const handleKeyDown = (event: KeyboardEvent) => {
            if (event.key === "Escape") {
                closeSelectedZone();
            }
        };
        document.addEventListener("keydown", handleKeyDown);
        return () => document.removeEventListener("keydown", handleKeyDown);
    }, [closeSelectedZone, selectedZoneInfo]);

    return (
        <ModernCard disableMinHeight>
            <Stack direction={{xs: "column", sm: "row"}} alignItems={{xs: "flex-start", sm: "center"}} justifyContent="space-between" gap={1}>
                <Box>
                    <Typography
                        variant="subtitle2"
                        sx={{fontWeight: 800, letterSpacing: 0.45, textTransform: "uppercase"}}
                        color="text.secondary"
                    >
                        Floor Heat Map
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                        View one floor at a time with occupancy overlays.
                    </Typography>
                </Box>
                <Button
                    size="small"
                    variant={expanded ? "contained" : "outlined"}
                    onClick={() => {
                        setSelectedZoneInfo(null);
                        setExpanded((prev) => !prev);
                    }}
                    sx={{
                        border: "1px solid",
                        borderColor: expanded ? activeControlBorder : neutralControlBorder,
                        borderRadius: 999,
                        textTransform: "none",
                        minHeight: 44,
                        px: 1.5,
                        fontWeight: 700,
                        color: "text.primary",
                        bgcolor: expanded ? activeControlBg : neutralControlBg,
                        boxShadow: expanded ? activeControlShadow : "none",
                        "&:hover": {
                            borderColor: expanded ? activeControlBorder : neutralControlBorder,
                            bgcolor: expanded ? activeControlHoverBg : neutralControlHoverBg,
                            boxShadow: expanded ? activeControlShadow : "none",
                        },
                    }}
                >
                    {expanded ? "Hide map" : "Show map"}
                </Button>
            </Stack>

            <Collapse in={expanded} timeout={180} unmountOnExit>
                <Stack spacing={1.1}>
                    <Box
                        sx={{
                            overflowX: "auto",
                            overflowY: "hidden",
                            WebkitOverflowScrolling: "touch",
                            pb: 0.3,
                            "&::-webkit-scrollbar": {display: "none"},
                            scrollbarWidth: "none",
                        }}
                    >
                        <ToggleButtonGroup
                            size="small"
                            exclusive
                            value={effectiveSelectedFloor}
                            onChange={(_, value: number | null) => {
                                if (value !== null) {
                                    setSelectedZoneInfo(null);
                                    setSelectedFloor(value);
                                }
                            }}
                            sx={{
                                flexWrap: "nowrap",
                                width: "max-content",
                                gap: 0.7,
                                "& .MuiToggleButton-root": {
                                    borderRadius: 999,
                                    borderColor: neutralControlBorder,
                                    color: "text.secondary",
                                    bgcolor: neutralControlBg,
                                    textTransform: "none",
                                    minHeight: 44,
                                    px: 1.05,
                                    minWidth: 0,
                                    fontWeight: 700,
                                    fontSize: "0.68rem",
                                    whiteSpace: "nowrap",
                                    "&:hover": {
                                        bgcolor: neutralControlHoverBg,
                                    },
                                },
                                "& .MuiToggleButton-root.Mui-selected": {
                                    color: "text.primary",
                                    bgcolor: activeControlBg,
                                    borderColor: activeControlBorder,
                                    boxShadow: activeControlShadow,
                                },
                                "& .MuiToggleButton-root.Mui-selected:hover": {
                                    bgcolor: activeControlHoverBg,
                                },
                            }}
                        >
                            {floors.map((floor) => (
                                <ToggleButton key={floor} value={floor}>
                                    {floorLabel(floor)}
                                </ToggleButton>
                            ))}
                        </ToggleButtonGroup>
                    </Box>

                    <HeatmapSvg
                        facilityId={facilityId}
                        floor={effectiveSelectedFloor}
                        data={singleFloorData}
                        showDebugCoords={showDebugCoords}
                        selectedZoneKey={selectedZoneInfo?.key ?? null}
                        occupancyThresholds={occupancyThresholds}
                        onSelectZone={(key, trigger) => setSelectedZoneInfo({key, trigger})}
                    />
                </Stack>
            </Collapse>
            <Popper
                open={Boolean(selectedZoneInfo && selectedZoneSummary && selectedZonePresentation)}
                anchorEl={selectedZoneInfo?.trigger as unknown as HTMLElement | undefined}
                placement="bottom"
                sx={{zIndex: theme.zIndex.tooltip}}
            >
                <ClickAwayListener onClickAway={closeSelectedZone}>
                    <Paper
                        id={HEATMAP_ZONE_DIALOG_ID}
                        role="dialog"
                        aria-modal="false"
                        aria-labelledby={HEATMAP_ZONE_DIALOG_TITLE_ID}
                        sx={{
                            px: 1.15,
                            py: 0.9,
                            borderRadius: 2,
                            border: "1px solid",
                            borderColor: "rgba(15, 23, 42, 0.16)",
                            boxShadow: "0 10px 24px rgba(0,0,0,0.12)",
                            minWidth: 140,
                        }}
                    >
                        <Typography
                            id={HEATMAP_ZONE_DIALOG_TITLE_ID}
                            variant="body2"
                            sx={{fontWeight: 700, color: "text.primary"}}
                        >
                            {selectedZonePresentation?.label} details
                        </Typography>
                        <Typography
                            variant="caption"
                            role={selectedZonePresentation?.value === "—" ? "img" : undefined}
                            aria-label={selectedZonePresentation?.value === "—" ? selectedZonePresentation.ariaLabel : undefined}
                            sx={{
                                display: "block",
                                fontWeight: 800,
                                color: selectedZonePresentation?.valueColor ?? "text.secondary",
                                whiteSpace: "pre-line",
                            }}
                        >
                            {selectedZonePresentation?.value}
                        </Typography>
                        <Button
                            type="button"
                            size="small"
                            aria-label={`Close ${selectedZonePresentation?.label ?? "zone"} details`}
                            onClick={closeSelectedZone}
                            sx={{mt: 0.4, textTransform: "none"}}
                        >
                            Close
                        </Button>
                    </Paper>
                </ClickAwayListener>
            </Popper>
        </ModernCard>
    );
}
