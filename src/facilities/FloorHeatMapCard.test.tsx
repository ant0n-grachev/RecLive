import {act, fireEvent, render, screen, waitFor} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type {Location} from "../lib/types/facility";
import FloorHeatMapCard, {zoneAccessibleLabel} from "./FloorHeatMapCard";

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

afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
});

beforeEach(() => {
    vi.spyOn(console, "info").mockImplementation(() => undefined);
});

describe("FloorHeatMapCard", () => {
    it("describes partial coverage without presenting it as complete coverage", () => {
        expect(zoneAccessibleLabel({
            id: "power-house",
            label: "Power House",
            status: "partial",
            percent: 60,
            coverage: 0.6,
            count: 30,
        })).toBe("Power House: 60% full. Coverage: 60% of open capacity observed");
    });

    it("removes a stale heatmap debug global in normal production", () => {
        vi.stubEnv("DEV", false);
        vi.stubEnv("MODE", "production");
        window.heatmapdebug = vi.fn();

        render(
            <FloorHeatMapCard
                facilityId={1186}
                locations={[location(5761, "Power House", 0)]}
                nowTs={NOW_TS}
            />
        );

        expect(window.heatmapdebug).toBeUndefined();
    });

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
        expect(partialZone).toHaveAttribute("vector-effect", "non-scaling-stroke");
        expect(unavailableZone).toHaveAttribute("tabindex", "0");
        partialZone.focus();
        expect(partialZone).toHaveFocus();

        fireEvent.keyDown(partialZone, {key: "Enter"});
        expect(screen.getByRole("dialog", {name: "Racquetball details"})).toBeInTheDocument();
        expect(screen.getByText(/Coverage: 50%/)).toBeInTheDocument();

        fireEvent.keyDown(unavailableZone, {key: " "});
        expect(screen.getByRole("dialog", {name: "Track details"})).toBeInTheDocument();
        expect(screen.getByText("Live occupancy unavailable")).toBeInTheDocument();
        fireEvent.keyDown(document, {key: "Escape"});

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

    it("expands only a narrow zone's invisible hit polygon", () => {
        render(
            <FloorHeatMapCard
                facilityId={1186}
                locations={[
                    location(5753, "Racquetball Court 1", 4, {isClosed: true}),
                    location(5754, "Racquetball Court 2", 4, {isClosed: true}),
                ]}
                nowTs={NOW_TS}
            />
        );
        fireEvent.click(screen.getByRole("button", {name: "Show map"}));

        const visiblePolygon = document.querySelector('polygon[fill^="url(#closed-stripes-"]');
        const interactivePolygon = screen.getByRole("button", {name: "Racquetball: CLOSED"});
        const xValues = (element: Element) => (element.getAttribute("points") ?? "")
            .split(" ")
            .map((pair) => Number(pair.split(",")[0]));

        expect(visiblePolygon).not.toBeNull();
        expect(Math.min(...xValues(visiblePolygon!))).toBe(40);
        expect(Math.max(...xValues(visiblePolygon!))).toBe(44);
        expect(Math.min(...xValues(interactivePolygon))).toBe(38.375);
        expect(Math.max(...xValues(interactivePolygon))).toBe(45.625);
    });

    it("opens a labelled non-modal dialog from a zone and restores focus after Escape", async () => {
        const user = userEvent.setup();
        render(
            <FloorHeatMapCard
                facilityId={1186}
                locations={[location(5761, "Power House", 0, {
                    currentCapacity: 30,
                    maxCapacity: 50,
                })]}
                nowTs={NOW_TS}
            />
        );
        await user.click(screen.getByRole("button", {name: "Show map"}));

        const zone = screen.getByRole("button", {name: "Power House: 60% full"});
        act(() => zone.focus());
        await user.keyboard("{Enter}");

        const dialog = screen.getByRole("dialog", {name: "Power House details"});
        expect(dialog).toBeVisible();
        expect(dialog).toHaveAttribute("aria-modal", "false");
        expect(zone).toHaveAttribute("aria-haspopup", "dialog");
        expect(zone).toHaveAttribute("aria-expanded", "true");
        expect(zone).toHaveAttribute("aria-controls", dialog.id);

        await user.keyboard("{Escape}");

        expect(screen.queryByRole("dialog", {name: "Power House details"})).not.toBeInTheDocument();
        await waitFor(() => expect(zone).toHaveFocus());
    });

    it("uses the same focus-restoring close path for its close control and click-away", async () => {
        const user = userEvent.setup();
        render(
            <FloorHeatMapCard
                facilityId={1186}
                locations={[location(5761, "Power House", 0, {
                    currentCapacity: 30,
                    maxCapacity: 50,
                })]}
                nowTs={NOW_TS}
            />
        );
        await user.click(screen.getByRole("button", {name: "Show map"}));

        const zone = screen.getByRole("button", {name: "Power House: 60% full"});
        await user.click(zone);
        await user.click(screen.getByRole("button", {name: "Close Power House details"}));
        await waitFor(() => expect(zone).toHaveFocus());

        await user.click(zone);
        await user.click(document.body);

        expect(screen.queryByRole("dialog", {name: "Power House details"})).not.toBeInTheDocument();
        await waitFor(() => expect(zone).toHaveFocus());
    });
});
