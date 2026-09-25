import {Box} from "@mui/material";
import {alpha} from "@mui/material/styles";
import type {FacilityId} from "../../lib/types/facility";
import {OCCUPANCY_MAIN_HEX, type OccupancyThresholds} from "../../shared/utils/styles";
import {getInteractiveZoneCorners} from "./heatmapGeometry";
import {
    getZonePresentation,
    HEATMAP_BASE_FILL,
    HEATMAP_ZONE_DIALOG_ID,
    type FloorRenderData,
} from "./heatmapModel";

interface HeatmapSvgProps {
    facilityId: FacilityId;
    floor: number;
    data: FloorRenderData;
    showDebugCoords: boolean;
    selectedZoneKey: string | null;
    occupancyThresholds?: OccupancyThresholds | null;
    onSelectZone: (key: string, trigger: SVGPolygonElement) => void;
}

export function HeatmapSvg({
    facilityId,
    floor,
    data,
    showDebugCoords,
    selectedZoneKey,
    occupancyThresholds,
    onSelectZone,
}: HeatmapSvgProps) {
    if (!data.floorMap) {
        return null;
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
                <rect
                    x="0"
                    y="0"
                    width={data.gridCols}
                    height={data.gridRows}
                    fill={HEATMAP_BASE_FILL}
                />
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

                {data.zoneSummaries.map((zoneSummary, zoneIndex) => {
                    const polygonPoints = getInteractiveZoneCorners(zoneSummary.zone)
                        .map((point) => `${(point.x / 100) * data.gridCols},${(point.y / 100) * data.gridRows}`)
                        .join(" ");
                    const presentation = getZonePresentation(zoneSummary, occupancyThresholds);
                    if (!presentation) {
                        return null;
                    }
                    const isSelected = selectedZoneKey === zoneSummary.key;

                    return (
                        <polygon
                            key={`hit-area-${floor}-${zoneSummary.zone.label}-${zoneIndex}`}
                            points={polygonPoints}
                            fill="rgba(0, 0, 0, 0.001)"
                            stroke="transparent"
                            strokeWidth={0.2}
                            vectorEffect="non-scaling-stroke"
                            role="button"
                            tabIndex={0}
                            aria-label={presentation.ariaLabel}
                            aria-haspopup="dialog"
                            aria-expanded={isSelected}
                            aria-controls={isSelected ? HEATMAP_ZONE_DIALOG_ID : undefined}
                            style={{cursor: "pointer"}}
                            onClick={(event) => {
                                event.stopPropagation();
                                onSelectZone(zoneSummary.key, event.currentTarget);
                            }}
                            onKeyDown={(event) => {
                                if (event.key !== "Enter" && event.key !== " ") return;
                                event.preventDefault();
                                event.stopPropagation();
                                onSelectZone(zoneSummary.key, event.currentTarget);
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
}
