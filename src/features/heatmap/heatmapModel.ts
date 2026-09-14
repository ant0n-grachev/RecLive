import {alpha} from "@mui/material/styles";
import type {FacilityId, Location} from "../../lib/types/facility";
import {
    computeOccupancySummary,
    type OccupancySummary,
} from "../../shared/occupancy/computeOccupancySummary";
import {
    combineOccupancyThresholds,
    getOccupancyColor,
    getOccupancyTone,
    OCCUPANCY_MAIN_HEX,
    type OccupancyThresholds,
} from "../../shared/utils/styles";
import {FLOOR_MAPS, type FloorMapConfig, type ZoneConfig} from "./floorMaps";
import {getZoneBounds, isInsidePolygon} from "./heatmapGeometry";

const GRID_PRECISION_MULTIPLIER = 4;

const getZoneOccupancyThresholds = (
    zone: ZoneConfig,
    locations: readonly Location[],
    locationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>>,
    fallback?: OccupancyThresholds | null
): OccupancyThresholds | null => (
    combineOccupancyThresholds(
        locations
            .filter((loc) => zone.ids.includes(loc.locationId))
            .map((loc) => ({
                thresholds: locationOccupancyThresholds[loc.locationId],
                weight: typeof loc.maxCapacity === "number" && Number.isFinite(loc.maxCapacity)
                    ? Math.max(0, loc.maxCapacity)
                    : 0,
            }))
    ) ?? fallback ?? null
);

export interface HeatCell {
    x: number;
    y: number;
    size: number;
    fill: string;
}

export interface ZoneSummaryModel {
    key: string;
    zone: ZoneConfig;
    summary: OccupancySummary;
    thresholds: OccupancyThresholds | null;
}

export interface HeatmapZonePresentation {
    id: string;
    label: string;
    status: OccupancySummary["status"];
    percent: number | null;
    coverage: number | null;
    count: number | null;
}

export interface ZonePresentation extends HeatmapZonePresentation {
    ariaLabel: string;
    value: string;
    valueColor: string;
}

export const HEATMAP_BASE_FILL = "rgba(100, 116, 139, 0.18)";
export const HEATMAP_ZONE_DIALOG_ID = "heatmap-zone-dialog";
export const HEATMAP_ZONE_DIALOG_TITLE_ID = "heatmap-zone-dialog-title";

export function zoneAccessibleLabel(zone: HeatmapZonePresentation): string {
    if (zone.status === "closed") {
        return `${zone.label}: CLOSED`;
    }
    if (zone.status !== "live" || zone.percent === null) {
        return `${zone.label}: occupancy unavailable`;
    }

    return `${zone.label}: ${Math.round(zone.percent)}% full`;
}

const getOverlayFill = (percent: number, occupancyThresholds?: OccupancyThresholds | null): string => {
    const tone = getOccupancyTone(percent, occupancyThresholds);
    if (tone === null) {
        return "rgba(100, 116, 139, 0.22)";
    }
    return alpha(OCCUPANCY_MAIN_HEX[tone], 0.75);
};

export interface FloorRenderData {
    floorMap: FloorMapConfig | null;
    gridCols: number;
    gridRows: number;
    mapScale: number;
    heatCells: readonly HeatCell[];
    closedZones: readonly ZoneConfig[];
    zoneSummaries: readonly ZoneSummaryModel[];
}

export const EMPTY_RENDER_DATA: FloorRenderData = {
    floorMap: null,
    gridCols: 0,
    gridRows: 0,
    mapScale: 1,
    heatCells: [],
    closedZones: [],
    zoneSummaries: [],
};

export function buildFloorRenderData(
    facilityId: FacilityId,
    floor: number,
    locations: readonly Location[],
    nowTs: number,
    occupancyThresholds: OccupancyThresholds | null = null,
    locationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>> = {}
): FloorRenderData {
    const floorMap = FLOOR_MAPS[facilityId][floor];
    if (!floorMap) {
        return EMPTY_RENDER_DATA;
    }

    const floorLocations = locations.filter((loc) => loc.floor === floor);
    const zoneModels = floorMap.zones.map((zone, zoneIndex) => {
        const matched = floorLocations.filter((loc) => zone.ids.includes(loc.locationId));
        return {
            key: `${facilityId}:${floor}:${zoneIndex}`,
            bounds: getZoneBounds(zone),
            zone,
            summary: computeOccupancySummary(matched, {nowMs: nowTs}),
            thresholds: getZoneOccupancyThresholds(zone, floorLocations, locationOccupancyThresholds, occupancyThresholds),
        };
    });

    const baseGridCols = floorMap.aspectRatio === "9 / 16" ? 25 : 48;
    const baseGridRows = floorMap.aspectRatio === "9 / 16" ? 44 : 28;
    const gridCols = baseGridCols * GRID_PRECISION_MULTIPLIER;
    const gridRows = baseGridRows * GRID_PRECISION_MULTIPLIER;
    const heatCells: HeatCell[] = [];

    for (let row = 0; row < gridRows; row += 1) {
        for (let col = 0; col < gridCols; col += 1) {
            const x = ((col + 0.5) / gridCols) * 100;
            const y = ((row + 0.5) / gridRows) * 100;
            let zonePercent = Number.NEGATIVE_INFINITY;
            let insideAnyUsableZone = false;
            let zoneThresholds: OccupancyThresholds | null = occupancyThresholds;

            for (const item of zoneModels) {
                if (x < item.bounds.minX || x > item.bounds.maxX || y < item.bounds.minY || y > item.bounds.maxY) {
                    continue;
                }
                if (!isInsidePolygon(item.zone, x, y)) {
                    continue;
                }

                if (
                    item.summary.status !== "live"
                    || item.summary.percent === null
                ) {
                    continue;
                }
                insideAnyUsableZone = true;
                if (item.summary.percent >= zonePercent) {
                    zonePercent = item.summary.percent;
                    zoneThresholds = item.thresholds ?? occupancyThresholds;
                }
            }

            if (insideAnyUsableZone) {
                heatCells.push({
                    x: col,
                    y: row,
                    size: 1,
                    fill: getOverlayFill(zonePercent, zoneThresholds),
                });
            }
        }
    }

    return {
        floorMap,
        gridCols,
        gridRows,
        mapScale: floorMap.zoom,
        heatCells,
        closedZones: zoneModels.filter((item) => item.summary.status === "closed").map((item) => item.zone),
        zoneSummaries: zoneModels.map((item) => ({
            key: item.key,
            zone: item.zone,
            summary: item.summary,
            thresholds: item.thresholds,
        })),
    };
}

export const getZonePresentation = (
    zoneSummary: ZoneSummaryModel,
    fallbackThresholds?: OccupancyThresholds | null
): ZonePresentation => {
    const {summary, zone} = zoneSummary;
    const presentation = {
        id: zoneSummary.key,
        label: zone.label,
        status: summary.status,
        percent: summary.percent,
        coverage: summary.coverage,
        count: summary.count,
    } satisfies HeatmapZonePresentation;
    if (summary.status === "closed") {
        return {
            ...presentation,
            ariaLabel: zoneAccessibleLabel(presentation),
            value: "CLOSED",
            valueColor: "error.main",
        };
    }

    const isObserved = summary.status === "live" && summary.percent !== null;
    if (!isObserved || summary.percent === null) {
        return {
            ...presentation,
            ariaLabel: zoneAccessibleLabel(presentation),
            value: "—",
            valueColor: "text.secondary",
        };
    }

    const percentText = `${Math.round(summary.percent)}% full`;
    return {
        ...presentation,
        ariaLabel: zoneAccessibleLabel(presentation),
        value: percentText,
        valueColor: getOccupancyColor(
            summary.percent,
            zoneSummary.thresholds ?? fallbackThresholds
        ),
    };
};
