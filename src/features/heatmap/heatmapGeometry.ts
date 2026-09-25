import type {Point, ZoneConfig} from "./floorMaps";

export interface ZoneBounds {
    minX: number;
    maxX: number;
    minY: number;
    maxY: number;
}

export const getZoneBounds = (zone: ZoneConfig): ZoneBounds => ({
    minX: Math.min(...zone.corners.map((point) => point.x)),
    maxX: Math.max(...zone.corners.map((point) => point.x)),
    minY: Math.min(...zone.corners.map((point) => point.y)),
    maxY: Math.max(...zone.corners.map((point) => point.y)),
});

const MIN_INTERACTIVE_ZONE_SPAN_PERCENT = 7.25;

export const getInteractiveZoneCorners = (zone: ZoneConfig): readonly Point[] => {
    const bounds = getZoneBounds(zone);
    const centerX = (bounds.minX + bounds.maxX) / 2;
    const centerY = (bounds.minY + bounds.maxY) / 2;
    const width = bounds.maxX - bounds.minX;
    const height = bounds.maxY - bounds.minY;
    const scaleX = width > 0 ? Math.max(1, MIN_INTERACTIVE_ZONE_SPAN_PERCENT / width) : 1;
    const scaleY = height > 0 ? Math.max(1, MIN_INTERACTIVE_ZONE_SPAN_PERCENT / height) : 1;

    return zone.corners.map((point) => ({
        x: Math.min(100, Math.max(0, centerX + (point.x - centerX) * scaleX)),
        y: Math.min(100, Math.max(0, centerY + (point.y - centerY) * scaleY)),
    }));
};

export const isInsidePolygon = (zone: ZoneConfig, x: number, y: number): boolean => {
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
