import {render, screen} from "@testing-library/react";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import OccupancyHero from "./OccupancyHero";

const NOW_TS = Date.parse("2026-08-31T12:00:00Z");

const summary = (
    status: OccupancySummary["status"],
    count: number | null,
    observedCapacity: number
): OccupancySummary => ({
    count,
    observedCapacity,
    expectedOpenCapacity: observedCapacity || 100,
    coverage: status === "live" ? 1 : status === "partial" ? 0.6 : 0,
    percent: count === null || observedCapacity <= 0 ? null : (count / observedCapacity) * 100,
    observedLocations: count === null ? 0 : 1,
    expectedLocations: 1,
    latestFetchedAt: count === null ? null : "2026-08-31T11:55:00Z",
    oldestFetchedAt: count === null ? null : "2026-08-31T11:55:00Z",
    status,
});

const hero = (occupancySummary: OccupancySummary) => (
    <OccupancyHero
        summary={occupancySummary}
        nowTs={NOW_TS}
        facilityId={1186}
    />
);

const expectTuple = (count: number, capacity: number, percent: number) => {
    expect(screen.getByRole("heading", {level: 3})).toHaveTextContent(
        new RegExp(`${count}\\s*\\/\\s*${capacity}`)
    );
    expect(screen.getByText(`${percent}% full`)).toBeInTheDocument();
    expect(screen.getByRole("progressbar", {name: "Current occupancy percentage"}))
        .toHaveAttribute("aria-valuenow", String(percent));
};

describe("OccupancyHero trusted tuple transitions", () => {
    it("shows the truthful tuple on the first observed render", () => {
        render(hero(summary("live", 30, 100)));

        expectTuple(30, 100, 30);
    });

    it("shows the truthful tuple immediately when occupancy recovers", () => {
        const {rerender} = render(hero(summary("insufficient", null, 0)));
        expect(screen.queryByText("Live Occupancy")).not.toBeInTheDocument();

        rerender(hero(summary("live", 30, 100)));

        expect(screen.queryByLabelText("Current count unavailable")).not.toBeInTheDocument();
        expectTuple(30, 100, 30);
    });

    it("updates count, capacity, percent, and bar atomically when capacity changes", () => {
        const {rerender} = render(hero(summary("live", 30, 100)));
        expectTuple(30, 100, 30);

        rerender(hero(summary("live", 30, 200)));

        expectTuple(30, 200, 15);
    });
});
