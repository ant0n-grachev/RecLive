import {useEffect, useMemo, useState} from "react";
import {
    Box,
    Button,
    Collapse,
    Popover,
    Stack,
    ToggleButton,
    ToggleButtonGroup,
    Typography,
} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import ModernCard from "../shared/components/ModernCard";
import type {Location, FacilityId} from "../lib/types/facility";
import {
    combineOccupancyThresholds,
    getOccupancyColor,
    getOccupancyTone,
    OCCUPANCY_MAIN_HEX,
    type OccupancyThresholds,
} from "../shared/utils/styles";
import {env} from "../lib/config/env";

declare global {
    interface Window {
        heatmapdebug?: () => void;
    }
}

interface Props {
    facilityId: FacilityId;
    locations: Location[];
    occupancyThresholds?: OccupancyThresholds | null;
    locationOccupancyThresholds?: Partial<Record<number, OccupancyThresholds>>;
}

interface Point {
    x: number;
    y: number;
}

interface ZoneConfig {
    label: string;
    ids: readonly number[];
    corners: readonly [Point, Point, Point, ...Point[]];
}

interface FloorMapConfig {
    image: string;
    aspectRatio: string;
    zoom: number;
    zones: readonly ZoneConfig[];
}

const DEFAULT_DEBUG_COORDS = false;
const GRID_PRECISION_MULTIPLIER = 4;
const ENABLE_HEATMAP_DEBUG_HOOK = env.isDev;

const FLOOR_MAPS: Record<FacilityId, Partial<Record<number, FloorMapConfig>>> = {
    1186: {
        0: {
            image: "/floor-maps/nick_page_01.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Power House",
                    ids: [5761],
                    corners: [
                        {x: 40, y: 60.5},
                        {x: 70.5, y: 60.5},
                        {x: 70.5, y: 72.6},
                        {x: 40, y: 72.6},
                    ],
                },
                {
                    label: "Pool",
                    ids: [5764],
                    corners: [
                        {x: 50.5, y: 32.6},
                        {x: 70.5, y: 32.6},
                        {x: 70.5, y: 60.5},
                        {x: 50.5, y: 60.5},
                    ],
                },
            ],
        },
        1: {
            image: "/floor-maps/nick_page_02.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Level 1 Fitness",
                    ids: [5760],
                    corners: [
                        {x: 40, y: 32.8},
                        {x: 45, y: 32.7},
                        {x: 45, y: 57.5},
                        {x: 36.5, y: 57.5},
                    ],
                },
                {
                    label: "Courts 1 & 2",
                    ids: [7089],
                    corners: [
                        {x: 50.5, y: 72.5},
                        {x: 71, y: 72.5},
                        {x: 71, y: 59.5},
                        {x: 50.5, y: 59.5},
                    ],
                },
            ],
        },
        2: {
            image: "/floor-maps/nick_page_03.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Level 2 Fitness",
                    ids: [5762],
                    corners: [
                        {x: 39.5, y: 32},
                        {x: 45.5, y: 32},
                        {x: 45.5, y: 73},
                        {x: 38.5, y: 73},
                        {x: 36, y: 55.5},
                    ],
                },
            ],
        },
        3: {
            image: "/floor-maps/nick_page_04.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Level 3 Fitness",
                    ids: [5758],
                    corners: [
                        {x: 38.5, y: 32.5},
                        {x: 50.5, y: 32.5},
                        {x: 50.5, y: 73.5},
                        {x: 35.5, y: 73.5},
                        {x: 38.5, y: 73.5},
                        {x: 35.5, y: 56},
                    ],
                },
                {
                    label: "Courts 3-6",
                    ids: [7090],
                    corners: [
                        {x: 50.5, y: 31.5},
                        {x: 70.5, y: 31.5},
                        {x: 70.5, y: 59.7},
                        {x: 50.5, y: 59.7},
                    ],
                },
                {
                    label: "Courts 7 & 8",
                    ids: [5766],
                    corners: [
                        {x: 70.5, y: 59.7},
                        {x: 50.5, y: 59.7},
                        {x: 50.5, y: 72.5},
                        {x: 70.5, y: 72.5},
                    ],
                },
            ],
        },
        4: {
            image: "/floor-maps/nick_page_05.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Track",
                    ids: [5763],
                    corners: [
                        {x: 38.5, y: 31},
                        {x: 70.5, y: 31},
                        {x: 70.5, y: 60},
                        {x: 35, y: 60},
                    ],
                },
                {
                    label: "Racquetball",
                    ids: [5753, 5754],
                    corners: [
                        {x: 40, y: 47},
                        {x: 44, y: 47},
                        {x: 44, y: 56.5},
                        {x: 40, y: 56.5},
                    ],
                },
            ],
        },
    },
    1656: {
        1: {
            image: "/floor-maps/bakke_page_01.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                // {
                //     label: "The Point",
                //     ids: [8718],
                //     corners: [
                //         {x: 22.5, y: 33},
                //         {x: 45.5, y: 33},
                //         {x: 45.5, y: 53},
                //         {x: 22.5, y: 53},
                //     ],
                // },
                {
                    label: "Level 1 Fitness",
                    ids: [8717],
                    corners: [
                        {x: 28.7, y: 38.5},
                        {x: 36, y: 38.5},
                        {x: 44.5, y: 30},
                        {x: 44.5, y: 48.5},
                        {x: 48, y: 48.5},
                        {x: 48, y: 51},
                        {x: 28.7, y: 51},
                    ],
                },
                {
                    label: "Courts 1 & 2",
                    ids: [8720],
                    corners: [
                        {x: 28.7, y: 51},
                        {x: 40, y: 51},
                        {x: 40, y: 72.5},
                        {x: 28.7, y: 72.5},
                    ],
                },
                {
                    label: "Courts 5-8",
                    ids: [8698],
                    corners: [
                        {x: 60.4, y: 30},
                        {x: 73, y: 30},
                        {x: 73, y: 48.5},
                        {x: 60.4, y: 48.5},
                    ],
                },
                {
                    label: "Cove Pool",
                    ids: [8716],
                    corners: [
                        {x: 44.5, y: 30},
                        {x: 60.4, y: 30},
                        {x: 60.4, y: 48.5},
                        {x: 44.5, y: 48.5},
                    ],
                },
                {
                    label: "Ice Center",
                    ids: [10550],
                    corners: [
                        {x: 48, y: 48.5},
                        {x: 70.5, y: 48.5},
                        {x: 70.5, y: 71.5},
                        {x: 48, y: 71.5},
                    ],
                },
            ],
        },
        2: {
            image: "/floor-maps/bakke_page_02.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                {
                    label: "Level 2 Fitness",
                    ids: [8705],
                    corners: [
                        {x: 28, y: 38.5},
                        {x: 37, y: 38.5},
                        {x: 45, y: 30},
                        {x: 45, y: 49},
                        {x: 42, y: 51.5},
                        {x: 41.5, y: 58},
                        {x: 28, y: 58},
                    ],
                },
                // {
                //     label: "Esports Room",
                //     ids: [8712],
                //     corners: [
                //         {x: 26.5, y: 37.5},
                //         {x: 41.5, y: 37.5},
                //         {x: 41.5, y: 52.5},
                //         {x: 26.5, y: 52.5},
                //     ],
                // },
            ],
        },
        3: {
            image: "/floor-maps/bakke_page_03.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                {
                    label: "Level 3 Fitness",
                    ids: [8700],
                    corners: [
                        {x: 27, y: 45},
                        {x: 38, y: 37.5},
                        {x: 45, y: 28},
                        {x: 48.5, y: 28},
                        {x: 48.5, y: 47},
                        {x: 42.5, y: 50},
                        {x: 39, y: 69},
                        {x: 28.5, y: 69},
                    ],
                },
                {
                    label: "Courts 3 & 4",
                    ids: [8714],
                    corners: [
                        {x: 48.5, y: 28},
                        {x: 72.5, y: 28},
                        {x: 72.5, y: 47},
                        {x: 48.5, y: 47},
                    ],
                },
                // {
                //     label: "Mount Mendota",
                //     ids: [8701],
                //     corners: [
                //         {x: 30, y: 37.5},
                //         {x: 48, y: 37.5},
                //         {x: 48, y: 52.5},
                //         {x: 30, y: 52.5},
                //     ],
                // },
            ],
        },
        4: {
            image: "/floor-maps/bakke_page_04.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                {
                    label: "Track",
                    ids: [8694],
                    corners: [
                        {x: 48.5, y: 31.5},
                        {x: 72.7, y: 31.5},
                        {x: 72.7, y: 47.5},
                        {x: 48.5, y: 47.5},
                    ],
                },
                {
                    label: "Level 4 Fitness",
                    ids: [8699],
                    corners: [
                        {x: 28, y: 44},
                        {x: 38, y: 38},
                        {x: 45, y: 28.5},
                        {x: 55, y: 28.5},
                        {x: 55, y: 31.5},
                        {x: 48.5, y: 31.5},
                        {x: 45, y: 32.9},
                        {x: 45, y: 47.5},
                        {x: 39.3, y: 61.5},
                        {x: 28, y: 61.5},
                    ],
                },
                {
                    label: "Orbit",
                    ids: [8696],
                    corners: [
                        {x: 34, y: 48.8},
                        {x: 38.5, y: 48.8},
                        {x: 38.5, y: 57},
                        {x: 34, y: 57},
                    ],
                },
                {
                    label: "Skybox",
                    ids: [8695],
                    corners: [
                        {x: 45, y: 32.9},
                        {x: 48.5, y: 31.5},
                        {x: 48.5, y: 47.5},
                        {x: 45, y: 47.5},
                    ],
                },
            ],
        },
    },
};

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

const getZoneRatio = (zone: ZoneConfig, locations: Location[]): number => {
    const matched = locations.filter((loc) => zone.ids.includes(loc.locationId));
    const current = matched.reduce((sum, loc) => sum + Math.max(0, loc.currentCapacity ?? 0), 0);
    const max = matched.reduce((sum, loc) => sum + Math.max(0, loc.maxCapacity ?? 0), 0);
    return max > 0 ? current / max : 0;
};

const getZoneBounds = (zone: ZoneConfig): {minX: number; maxX: number; minY: number; maxY: number} => ({
    minX: Math.min(...zone.corners.map((point) => point.x)),
    maxX: Math.max(...zone.corners.map((point) => point.x)),
    minY: Math.min(...zone.corners.map((point) => point.y)),
    maxY: Math.max(...zone.corners.map((point) => point.y)),
});

const getZoneOccupancyThresholds = (
    zone: ZoneConfig,
    locations: Location[],
    locationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>>,
    fallback?: OccupancyThresholds | null
): OccupancyThresholds | null => (
    combineOccupancyThresholds(
        locations
            .filter((loc) => zone.ids.includes(loc.locationId))
            .map((loc) => ({
                thresholds: locationOccupancyThresholds[loc.locationId],
                weight: loc.maxCapacity ?? 0,
            }))
    ) ?? fallback ?? null
);

interface HeatCell {
    x: number;
    y: number;
    size: number;
    fill: string;
}

const getOverlayFill = (ratio: number, occupancyThresholds?: OccupancyThresholds | null): string | null => {
    const percent = clamp01(ratio) * 100;
    const tone = getOccupancyTone(percent, occupancyThresholds);
    if (tone === null) {
        return "rgba(100, 116, 139, 0.22)";
    }
    return alpha(OCCUPANCY_MAIN_HEX[tone], 0.75);
};

const isInsidePolygon = (zone: ZoneConfig, x: number, y: number): boolean => {
    const points = zone.corners;
    let inside = false;

    for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
        const xi = points[i].x;
        const yi = points[i].y;
        const xj = points[j].x;
        const yj = points[j].y;

        const intersects =
            (yi > y) !== (yj > y)
            && x < ((xj - xi) * (y - yi)) / ((yj - yi) || Number.EPSILON) + xi;

        if (intersects) {
            inside = !inside;
        }
    }

    return inside;
};

interface FloorRenderData {
    floorMap: FloorMapConfig | null;
    gridCols: number;
    gridRows: number;
    mapScale: number;
    heatCells: HeatCell[];
    closedZones: readonly ZoneConfig[];
    zoneSummaries: readonly {
        zone: ZoneConfig;
        percent: number;
        closed: boolean;
        thresholds: OccupancyThresholds | null;
    }[];
}

const EMPTY_RENDER_DATA: FloorRenderData = {
    floorMap: null,
    gridCols: 0,
    gridRows: 0,
    mapScale: 1,
    heatCells: [],
    closedZones: [],
    zoneSummaries: [],
};

const buildFloorRenderData = (
    facilityId: FacilityId,
    floor: number,
    locations: Location[],
    occupancyThresholds?: OccupancyThresholds | null,
    locationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>> = {}
): FloorRenderData => {
    const floorMap = FLOOR_MAPS[facilityId][floor];
    if (!floorMap) {
        return EMPTY_RENDER_DATA;
    }

    const floorLocations = locations.filter((loc) => loc.floor === floor);
    const zoneModels = floorMap.zones.map((zone) => ({
        bounds: getZoneBounds(zone),
        zone,
        ratio: getZoneRatio(zone, floorLocations),
        thresholds: getZoneOccupancyThresholds(zone, floorLocations, locationOccupancyThresholds, occupancyThresholds),
        closed: zone.ids.length > 0 && zone.ids.every((id) => (
            floorLocations.some((loc) => loc.locationId === id && loc.isClosed === true)
        )),
    }));

    const baseGridCols = floorMap.aspectRatio === "9 / 16" ? 25 : 48;
    const baseGridRows = floorMap.aspectRatio === "9 / 16" ? 44 : 28;
    const gridCols = baseGridCols * GRID_PRECISION_MULTIPLIER;
    const gridRows = baseGridRows * GRID_PRECISION_MULTIPLIER;
    const heatCells: HeatCell[] = [];

    for (let row = 0; row < gridRows; row += 1) {
        for (let col = 0; col < gridCols; col += 1) {
            const x = ((col + 0.5) / gridCols) * 100;
            const y = ((row + 0.5) / gridRows) * 100;
            let zoneValue = 0;
            let insideAnyOpenZone = false;
            let zoneThresholds: OccupancyThresholds | null = occupancyThresholds ?? null;

            for (const item of zoneModels) {
                if (x < item.bounds.minX || x > item.bounds.maxX || y < item.bounds.minY || y > item.bounds.maxY) {
                    continue;
                }
                if (!isInsidePolygon(item.zone, x, y)) {
                    continue;
                }

                if (item.closed) {
                    continue;
                }
                insideAnyOpenZone = true;
                if (item.ratio >= zoneValue) {
                    zoneValue = item.ratio;
                    zoneThresholds = item.thresholds ?? occupancyThresholds ?? null;
                }
            }

            if (insideAnyOpenZone) {
                const value = clamp01(zoneValue);
                const fill = getOverlayFill(value, zoneThresholds);
                if (fill) {
                    heatCells.push({x: col, y: row, size: 1, fill});
                }
            }
        }
    }

    return {
        floorMap,
        gridCols,
        gridRows,
        mapScale: floorMap.zoom,
        heatCells,
        closedZones: zoneModels.filter((item) => item.closed).map((item) => item.zone),
        zoneSummaries: zoneModels.map((item) => ({
            zone: item.zone,
            percent: item.ratio * 100,
            closed: item.closed,
            thresholds: item.thresholds,
        })),
    };
};

const floorLabel = (floor: number): string => (floor === 0 ? "Lower" : `Floor ${floor}`);

export default function FloorHeatMapCard({
    facilityId,
    locations,
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
        label: string;
        value: string;
        valueColor: string;
        top: number;
        left: number;
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

    useEffect(() => {
        if (!ENABLE_HEATMAP_DEBUG_HOOK || typeof window === "undefined") {
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
            "[reclive] debug commands: window.heatmapdebug(), window.recliveShowPredictions(), window.recliveRestoreWarnings(), window.reclivePredictionOverrideStatus()"
        );

        return () => {
            delete window.heatmapdebug;
        };
    }, []);

    const singleFloorData = useMemo(
        () => {
            if (!expanded) {
                return EMPTY_RENDER_DATA;
            }
            return buildFloorRenderData(
                facilityId,
                effectiveSelectedFloor,
                locations,
                occupancyThresholds,
                locationOccupancyThresholds
            );
        },
        [expanded, facilityId, effectiveSelectedFloor, locations, occupancyThresholds, locationOccupancyThresholds]
    );

    const renderFloorMap = (floor: number, data: FloorRenderData) => {
        if (!data.floorMap) {
            return (
                <Typography variant="body2" color="text.secondary">
                    No map configured for this floor yet.
                </Typography>
            );
        }

        const isNickMap = facilityId === 1186;
        const effectiveMapScale = isNickMap ? 2 : data.mapScale;
        const mapLayers = (
            <Box
                sx={{
                    position: "absolute",
                    inset: 0,
                    transform: `scale(${effectiveMapScale})`,
                    transformOrigin: "center",
                }}
            >
                <Box
                    component="img"
                    src={data.floorMap.image}
                    alt={`Floor map ${floor}`}
                    sx={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        filter: "saturate(0.95) contrast(1.02)",
                    }}
                />

                <Box
                    component="svg"
                    viewBox={`0 0 ${data.gridCols} ${data.gridRows}`}
                    preserveAspectRatio="none"
                    sx={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        opacity: 0.92,
                        mixBlendMode: "multiply",
                        pointerEvents: "auto",
                    }}
                >
                    <defs>
                        <pattern id={`heat-grid-${facilityId}-${floor}`} width="1" height="1" patternUnits="userSpaceOnUse">
                            <path d="M 1 0 L 0 0 0 1" fill="none" stroke={alpha(OCCUPANCY_MAIN_HEX.success, 0.28)} strokeWidth="0.05"/>
                        </pattern>
                        <pattern
                            id={`closed-stripes-${facilityId}-${floor}`}
                            width="3"
                            height="3"
                            patternUnits="userSpaceOnUse"
                            patternTransform="rotate(45)"
                        >
                            <rect width="3" height="3" fill={alpha(OCCUPANCY_MAIN_HEX.error, 0.22)}/>
                            <line x1="0" y1="0" x2="0" y2="3" stroke="rgba(15, 23, 42, 0.7)" strokeWidth="1.1"/>
                        </pattern>
                    </defs>
                    {data.heatCells.map((cell) => (
                        <rect
                            key={`${floor}-${cell.x}-${cell.y}-${cell.size}`}
                            x={cell.x}
                            y={cell.y}
                            width={cell.size}
                            height={cell.size}
                            fill={cell.fill}
                        />
                    ))}
                    <rect x="0" y="0" width={data.gridCols} height={data.gridRows} fill={`url(#heat-grid-${facilityId}-${floor})`}/>

                    {data.closedZones.map((zone, zoneIndex) => {
                        const polygonPoints = zone.corners
                            .map((point) => `${(point.x / 100) * data.gridCols},${(point.y / 100) * data.gridRows}`)
                            .join(" ");

                        return (
                            <polygon
                                key={`closed-${floor}-${zone.label}-${zoneIndex}`}
                                points={polygonPoints}
                                fill={`url(#closed-stripes-${facilityId}-${floor})`}
                                stroke="rgba(15, 23, 42, 0.82)"
                                strokeWidth={0.14}
                            />
                        );
                    })}

                    {showDebugCoords && data.floorMap.zones.map((zone, zoneIndex) => {
                        const points = zone.corners.map((point) => ({
                            x: (point.x / 100) * data.gridCols,
                            y: (point.y / 100) * data.gridRows,
                            rawX: point.x,
                            rawY: point.y,
                        }));
                        const polygonPoints = points.map((point) => `${point.x},${point.y}`).join(" ");
                        const roomLabel = `${zone.label} ${zoneIndex + 1}`;
                        const centerX = points.reduce((sum, point) => sum + point.x, 0) / points.length;
                        const centerY = points.reduce((sum, point) => sum + point.y, 0) / points.length;

                        return (
                            <g key={`debug-${floor}-${zone.label}-${zoneIndex}`}>
                                <polygon
                                    points={polygonPoints}
                                    fill="rgba(255, 255, 255, 0.06)"
                                    stroke="rgba(79, 70, 229, 0.95)"
                                    strokeWidth={0.15}
                                />
                                {points.map((point, pointIndex) => (
                                    <g key={`debug-point-${floor}-${zone.label}-${pointIndex}`}>
                                        <circle
                                            cx={point.x}
                                            cy={point.y}
                                            r={0.2}
                                            fill="rgba(79, 70, 229, 0.95)"
                                        />
                                        <text
                                            x={point.x + 0.2}
                                            y={point.y - 0.15}
                                            fontSize={0.65}
                                            fill="rgba(30, 27, 75, 0.98)"
                                            stroke="rgba(255,255,255,0.94)"
                                            strokeWidth={0.03}
                                            paintOrder="stroke"
                                        >
                                            {`${pointIndex + 1}: ${point.rawX},${point.rawY}`}
                                        </text>
                                    </g>
                                ))}
                                <text
                                    x={centerX}
                                    y={centerY}
                                    fontSize={0.8}
                                    fill="rgba(15, 23, 42, 0.98)"
                                    stroke="rgba(255,255,255,0.96)"
                                    strokeWidth={0.04}
                                    paintOrder="stroke"
                                    textAnchor="middle"
                                    dominantBaseline="middle"
                                >
                                    {roomLabel}
                                </text>
                            </g>
                        );
                    })}

                    {data.zoneSummaries.map((summary, zoneIndex) => {
                        const polygonPoints = summary.zone.corners
                            .map((point) => `${(point.x / 100) * data.gridCols},${(point.y / 100) * data.gridRows}`)
                            .join(" ");

                        return (
                            <polygon
                                key={`hit-area-${floor}-${summary.zone.label}-${zoneIndex}`}
                                points={polygonPoints}
                                fill="rgba(0, 0, 0, 0.001)"
                                stroke="transparent"
                                strokeWidth={0.2}
                                style={{cursor: "pointer"}}
                                onClick={(event) => {
                                    event.stopPropagation();
                                    const percentText = `${Math.round(summary.percent)}% full`;
                                    const percentColor = getOccupancyColor(
                                        summary.percent,
                                        summary.thresholds ?? occupancyThresholds
                                    );
                                    setSelectedZoneInfo({
                                        label: summary.zone.label,
                                        value: summary.closed ? "CLOSED" : percentText,
                                        valueColor: summary.closed ? "error.main" : percentColor,
                                        top: Math.round(event.clientY),
                                        left: Math.round(event.clientX),
                                    });
                                }}
                            />
                        );
                    })}
                </Box>
            </Box>
        );

        return (
            <Box
                sx={{
                    position: "relative",
                    borderRadius: 2,
                    overflow: "hidden",
                    border: "1px solid",
                    borderColor: "divider",
                    aspectRatio: isNickMap ? "16 / 9" : data.floorMap.aspectRatio,
                    bgcolor: "#e7e7e7",
                    width: isNickMap ? {xs: "92%", sm: "100%"} : "100%",
                    mx: isNickMap ? "auto" : 0,
                }}
            >
                {isNickMap ? (
                    <Box
                        sx={{
                            position: "absolute",
                            left: "50%",
                            top: "50%",
                            width: "56.25%",
                            height: "177.78%",
                            transform: "translate(-50%, -50%) rotate(90deg)",
                        }}
                    >
                        {mapLayers}
                    </Box>
                ) : (
                    mapLayers
                )}
            </Box>
        );
    };

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
                        minHeight: 32,
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
                                    minHeight: 34,
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

                    {renderFloorMap(effectiveSelectedFloor, singleFloorData)}
                </Stack>
            </Collapse>
            <Popover
                open={Boolean(selectedZoneInfo)}
                onClose={() => setSelectedZoneInfo(null)}
                anchorReference="anchorPosition"
                anchorPosition={selectedZoneInfo ? {top: selectedZoneInfo.top, left: selectedZoneInfo.left} : undefined}
                transformOrigin={{vertical: "top", horizontal: "center"}}
                PaperProps={{
                    sx: {
                        px: 1.15,
                        py: 0.9,
                        borderRadius: 2,
                        border: "1px solid",
                        borderColor: "rgba(15, 23, 42, 0.16)",
                        boxShadow: "0 10px 24px rgba(0,0,0,0.12)",
                        minWidth: 140,
                    },
                }}
            >
                <Typography variant="body2" sx={{fontWeight: 700, color: "text.primary"}}>
                    {selectedZoneInfo?.label}
                </Typography>
                <Typography
                    variant="caption"
                    sx={{fontWeight: 800, color: selectedZoneInfo?.valueColor ?? "text.secondary"}}
                >
                    {selectedZoneInfo?.value}
                </Typography>
            </Popover>
        </ModernCard>
    );
}
