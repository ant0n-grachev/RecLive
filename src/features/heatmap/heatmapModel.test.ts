import type {Location} from "../../lib/types/facility";
import {
    fixtureLocationsByFacility,
    fixtureNowTs,
    fixtureThresholds,
} from "../../test/fixtures/dashboard";
import {FLOOR_MAPS, type ZoneConfig} from "./floorMaps";
import {getInteractiveZoneCorners, isInsidePolygon} from "./heatmapGeometry";
import {buildFloorRenderData} from "./heatmapModel";

const withLocation = (
    locations: readonly Location[],
    locationId: number,
    overrides: Partial<Location>
): Location[] => locations.map((location) => (
    location.locationId === locationId ? {...location, ...overrides} : location
));

describe("buildFloorRenderData", () => {
    it("retains closed zones, neutral unknown zones, and colored observed zones", () => {
        let locations = fixtureLocationsByFacility[1656].filter((location) => location.locationId !== 8720);
        locations = withLocation(locations, 10550, {isClosed: true});
        locations = withLocation(locations, 8698, {currentCapacity: null, fetchedAt: null});

        const result = buildFloorRenderData(
            1656,
            1,
            locations,
            fixtureNowTs,
            fixtureThresholds,
            {}
        );

        expect(result.closedZones.map((zone) => zone.label)).toEqual(["Ice Center"]);
        expect(result.zoneSummaries.find((zone) => zone.zone.label === "Courts 1 & 2")?.summary.status)
            .toBe("unknown");
        expect(result.zoneSummaries.find((zone) => zone.zone.label === "Courts 5-8")?.summary.status)
            .toBe("insufficient");
        expect(result.zoneSummaries.find((zone) => zone.zone.label === "Level 1 Fitness")?.summary.status)
            .toBe("live");
        expect(result.heatCells.length).toBeGreaterThan(0);
        expect(result.zoneSummaries.map((zone) => zone.key)).toEqual([
            "1656:1:0",
            "1656:1:1",
            "1656:1:2",
            "1656:1:3",
            "1656:1:4",
        ]);
        expect(result.floorMap?.zones.some((zone) => zone.label === "The Point")).toBe(false);
    });

    it("keeps fractional coverage and colored cells for a partially observed zone", () => {
        let locations = withLocation(fixtureLocationsByFacility[1186], 5753, {
            currentCapacity: 3,
            maxCapacity: 6,
        });
        locations = withLocation(locations, 5754, {
            currentCapacity: 2,
            maxCapacity: 6,
            fetchedAt: "2026-08-31T10:00:00Z",
        });

        const result = buildFloorRenderData(
            1186,
            4,
            locations,
            fixtureNowTs,
            fixtureThresholds,
            {}
        );
        const racquetball = result.zoneSummaries.find((zone) => zone.zone.label === "Racquetball");

        expect(racquetball?.summary.status).toBe("partial");
        expect(racquetball?.summary.coverage).toBe(0.5);
        expect(racquetball?.summary.percent).toBe(50);
        expect(result.heatCells.length).toBeGreaterThan(0);
        expect(result.mapScale).toBe(2);
    });

    it("returns the immutable empty render model for an unmapped floor", () => {
        expect(buildFloorRenderData(
            1656,
            0,
            fixtureLocationsByFacility[1656],
            fixtureNowTs,
            fixtureThresholds,
            {}
        )).toEqual({
            floorMap: null,
            gridCols: 0,
            gridRows: 0,
            mapScale: 1,
            heatCells: [],
            closedZones: [],
            zoneSummaries: [],
        });
    });
});

describe("heat-map geometry", () => {
    const square: ZoneConfig = {
        label: "Square",
        ids: [],
        corners: [
            {x: 0, y: 0},
            {x: 10, y: 0},
            {x: 10, y: 10},
            {x: 0, y: 10},
        ],
    };

    it("preserves polygon containment behavior at the current edge boundaries", () => {
        expect(isInsidePolygon(square, 5, 5)).toBe(true);
        expect(isInsidePolygon(square, 0, 5)).toBe(true);
        expect(isInsidePolygon(square, 10, 5)).toBe(false);
        expect(isInsidePolygon(square, 5, 0)).toBe(true);
        expect(isInsidePolygon(square, 5, 10)).toBe(false);
        expect(isInsidePolygon(square, -0.01, 5)).toBe(false);
    });

    it("expands only the Racquetball hit geometry while preserving its visible corners", () => {
        const racquetball = FLOOR_MAPS[1186][4]!.zones.find((zone) => zone.label === "Racquetball")!;
        const hitCorners = getInteractiveZoneCorners(racquetball);

        expect(racquetball.corners.map((point) => point.x)).toEqual([40, 44, 44, 40]);
        expect(hitCorners.map((point) => point.x)).toEqual([38.375, 45.625, 45.625, 38.375]);
        expect(hitCorners.map((point) => point.y)).toEqual([47, 47, 56.5, 56.5]);
    });
});
