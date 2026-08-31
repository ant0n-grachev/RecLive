import {fireEvent, render, screen} from "@testing-library/react";
import type {Location} from "../lib/types/facility";
import FloorHeatMapCard from "./FloorHeatMapCard";

const NOW_TS = Date.parse("2026-08-31T12:05:00Z");
const FRESH_FETCHED_AT = "2026-08-31T12:00:00Z";

const location = (
    locationId: number,
    locationName: string,
    floor: number,
    overrides: Partial<Location> = {}
): Location => ({
    facilityId: 1186,
    locationId,
    locationName,
    floor,
    isClosed: false,
    currentCapacity: 20,
    maxCapacity: 100,
    lastUpdated: null,
    fetchedAt: FRESH_FETCHED_AT,
    ...overrides,
});

describe("FloorHeatMapCard", () => {
    it("updates an open zone popover when freshness changes", () => {
        const locations = [location(5761, "Power House", 0)];
        const {rerender} = render(
            <FloorHeatMapCard facilityId={1186} locations={locations} nowTs={NOW_TS} />
        );
        fireEvent.click(screen.getByRole("button", {name: "Show map"}));

        const zone = screen.getByRole("button", {name: "Power House: 20% full"});
        fireEvent.click(zone, {clientX: 30, clientY: 40});
        expect(screen.getByText("20% full")).toBeInTheDocument();

        rerender(
            <FloorHeatMapCard
                facilityId={1186}
                locations={locations}
                nowTs={Date.parse("2026-08-31T12:10:00.001Z")}
            />
        );

        expect(screen.queryByText("20% full")).not.toBeInTheDocument();
        expect(screen.getByText("Live occupancy unavailable")).toBeInTheDocument();
    });

    it("names zone controls by trust state and supports keyboard activation", () => {
        const trackUnavailable = location(5763, "Track", 4, {
            currentCapacity: null,
            fetchedAt: null,
        });
        const racquetballLocations = [
            location(5753, "Racquetball Court 1", 4, {
                currentCapacity: 3,
                maxCapacity: 6,
            }),
            location(5754, "Racquetball Court 2", 4, {
                currentCapacity: 2,
                maxCapacity: 6,
                fetchedAt: "2026-08-31T10:00:00Z",
            }),
        ];
        const props = {
            facilityId: 1186 as const,
            nowTs: NOW_TS,
        };
        const {rerender} = render(
            <FloorHeatMapCard
                {...props}
                locations={[trackUnavailable, ...racquetballLocations]}
            />
        );
        fireEvent.click(screen.getByRole("button", {name: "Show map"}));

        const partialZone = screen.getByRole("button", {
            name: "Racquetball: 50% full. Coverage: 50% of open capacity observed",
        });
        const unavailableZone = screen.getByRole("button", {
            name: "Track: Live occupancy unavailable",
        });
        expect(partialZone).toHaveAttribute("tabindex", "0");
        expect(unavailableZone).toHaveAttribute("tabindex", "0");
        partialZone.focus();
        expect(partialZone).toHaveFocus();

        fireEvent.keyDown(partialZone, {key: "Enter"});
        expect(screen.getByText("Racquetball")).toBeInTheDocument();
        expect(screen.getByText(/Coverage: 50%/)).toBeInTheDocument();

        fireEvent.keyDown(unavailableZone, {key: " "});
        expect(screen.getByText("Track")).toBeInTheDocument();
        expect(screen.getByText("Live occupancy unavailable")).toBeInTheDocument();
        fireEvent.keyDown(screen.getByRole("presentation"), {key: "Escape"});

        rerender(
            <FloorHeatMapCard
                {...props}
                locations={[
                    location(5763, "Track", 4, {isClosed: true}),
                    ...racquetballLocations,
                ]}
            />
        );
        expect(screen.getByRole("button", {name: "Track: CLOSED"})).toBeInTheDocument();
    });
});
