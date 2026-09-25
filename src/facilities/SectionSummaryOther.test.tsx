import {render, screen} from "@testing-library/react";
import type {Location} from "../lib/types/facility";
import SectionSummaryOther from "./SectionSummaryOther";

const NOW_TS = Date.parse("2026-08-31T12:00:00Z");

const makeLocation = (overrides: Partial<Location> = {}): Location => ({
    facilityId: 1186,
    locationId: 1,
    locationName: "Healthy Room",
    floor: 1,
    isClosed: false,
    currentCapacity: 30,
    maxCapacity: 60,
    fetchedAt: "2026-08-31T11:59:00Z",
    lastUpdated: null,
    ...overrides,
});

describe("SectionSummaryOther", () => {
    it("omits an unavailable row while preserving a healthy row", () => {
        render(
            <SectionSummaryOther
                title="Other Rooms"
                exclude={[]}
                nowTs={NOW_TS}
                locations={[
                    makeLocation(),
                    makeLocation({
                        locationId: 2,
                        locationName: "Unavailable Room",
                        currentCapacity: null,
                        fetchedAt: null,
                    }),
                ]}
            />,
        );

        expect(screen.getByText("Healthy Room")).toBeVisible();
        expect(screen.queryByText("Unavailable Room")).not.toBeInTheDocument();
        expect(screen.queryByLabelText("Current count unavailable")).not.toBeInTheDocument();
    });

    it("hides the whole card when every row is unavailable", () => {
        render(
            <SectionSummaryOther
                title="Other Rooms"
                exclude={[]}
                nowTs={NOW_TS}
                locations={[makeLocation({currentCapacity: null, fetchedAt: null})]}
            />,
        );

        expect(screen.queryByText("Other Rooms")).not.toBeInTheDocument();
    });

    it("preserves an explicit facility closure", () => {
        render(
            <SectionSummaryOther
                title="Other Rooms"
                exclude={[]}
                nowTs={NOW_TS}
                locations={[makeLocation({isClosed: true})]}
            />,
        );

        expect(screen.getAllByText("CLOSED")).toHaveLength(2);
    });
});
