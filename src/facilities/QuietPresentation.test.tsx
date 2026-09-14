import {fireEvent, screen} from "@testing-library/react";
import {renderWithApp} from "../test/render";
import {partialSummary, fixtureSchedule} from "../test/fixtures/dashboard";
import type {Location} from "../lib/types/facility";
import OccupancyHero from "./OccupancyHero";
import SectionCommandCenter from "./SectionCommandCenter";
import SectionSummaryOther from "./SectionSummaryOther";
import ScheduleStatusCard from "./ScheduleStatusCard";
import FacilityHoursBlock from "./FacilityHoursBlock";

const nowTs = Date.parse("2026-08-31T12:00:00Z");
const locations: Location[] = [
    {facilityId: 1186, locationId: 1, locationName: "Room A", floor: 1, isClosed: false,
        currentCapacity: 30, maxCapacity: 60, fetchedAt: "2026-08-31T11:59:00Z", lastUpdated: null},
    {facilityId: 1186, locationId: 2, locationName: "Room B", floor: 1, isClosed: false,
        currentCapacity: 20, maxCapacity: 40, fetchedAt: "2026-08-30T11:59:00Z", lastUpdated: null},
];

it.each(["partial", "insufficient", "unknown"] as const)(
    "keeps %s facility counts neutral without presenting incomplete numbers as live", (status) => {
        renderWithApp(<OccupancyHero summary={{...partialSummary, status}} nowTs={nowTs} facilityId={1186}/>);
        expect(screen.getByLabelText("Current count unavailable")).toHaveTextContent("—");
        expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
        expect(screen.queryByText(/% full|Coverage:|Live occupancy unavailable|Updated/i)).not.toBeInTheDocument();
    },
);

it.each(["primary", "other"])("keeps a partial %s section neutral but preserves fresh room readings", (section) => {
    renderWithApp(section === "primary"
        ? <SectionCommandCenter title="Rooms" ids={[1, 2]} locations={locations} nowTs={nowTs}/>
        : <SectionSummaryOther title="Rooms" exclude={[]} locations={locations} nowTs={nowTs}/>);
    if (section === "primary") fireEvent.click(screen.getByRole("button", {name: /Rooms/}));
    expect(screen.getAllByLabelText("Current count unavailable")).toHaveLength(2);
    expect(screen.getAllByRole("img", {name: "Current count unavailable"})).toHaveLength(section === "primary" ? 2 : 1);
    expect(screen.getByText(/30 \/ 60/)).toBeVisible();
    expect(screen.queryByText(/20 \/ 40/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Coverage:|Live occupancy unavailable/)).not.toBeInTheDocument();
});

it("keeps unknown opening hours neutral without inventing an open status", () => {
    renderWithApp(<ScheduleStatusCard status={{state: "unknown", matchedRule: null}}/>);
    expect(screen.getByRole("img", {name: "Opening hours unavailable"})).toHaveTextContent("—");
    expect(screen.queryByText(/UNKNOWN|temporarily unavailable|Open now/)).not.toBeInTheDocument();
});

it("keeps saved official hours accessible without a stale-data warning", () => {
    renderWithApp(<FacilityHoursBlock facilityName="Nick" isLoading={false} error={null}
        schedule={{...fixtureSchedule, status: "stale", stale: true, error: "Update failed", errorCategory: "upstream_http"}}/>);
    fireEvent.click(screen.getByRole("button", {name: /official hours/i}));
    expect(screen.getByRole("table")).toBeVisible();
    expect(screen.queryByText(/out of date|last verified|Update failed|upstream_http/)).not.toBeInTheDocument();
});
