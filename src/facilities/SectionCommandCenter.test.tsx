import {render, screen} from "@testing-library/react";
import type {Location} from "../lib/types/facility";
import SectionCommandCenter from "./SectionCommandCenter";

const NOW_TS = Date.parse("2026-08-31T12:00:00Z");

const location = (
    currentCapacity: number | null,
    maxCapacity: number,
    fetchedAt: string | null = "2026-08-31T11:55:00Z"
): Location => ({
    facilityId: 1186,
    locationId: 5763,
    locationName: "Running Track",
    floor: 4,
    isClosed: false,
    currentCapacity,
    maxCapacity,
    lastUpdated: null,
    fetchedAt,
});

const section = (current: Location) => (
    <SectionCommandCenter
        title="Running Track"
        ids={[5763]}
        locations={[current]}
        nowTs={NOW_TS}
    />
);

const expectTuple = (count: number, capacity: number, percent: number) => {
    expect(screen.getByText(`${count} / ${capacity}`)).toBeInTheDocument();
    expect(screen.getByText(`${percent}% full`)).toBeInTheDocument();
};

describe("SectionCommandCenter trusted tuple transitions", () => {
    it("shows the truthful tuple on the first observed render", () => {
        render(section(location(30, 100)));

        expectTuple(30, 100, 30);
    });

    it("shows the truthful tuple immediately when occupancy recovers", () => {
        const {rerender} = render(section(location(null, 100, null)));
        expect(screen.getByLabelText("Current count unavailable")).toHaveTextContent("—");

        rerender(section(location(30, 100)));

        expect(screen.queryByLabelText("Current count unavailable")).not.toBeInTheDocument();
        expectTuple(30, 100, 30);
    });

    it("updates count, capacity, and percent atomically when capacity changes", () => {
        const {rerender} = render(section(location(30, 100)));
        expectTuple(30, 100, 30);

        rerender(section(location(30, 200)));

        expectTuple(30, 200, 15);
    });
});
