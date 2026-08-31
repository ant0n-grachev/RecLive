import {fireEvent, render, screen} from "@testing-library/react";
import type {OccupancySummary} from "../shared/occupancy/computeOccupancySummary";
import {
    default as CrowdAlertSubscriptionCard,
    resolveInitialSectionKey,
    type AlertSectionOption,
} from "./CrowdAlertSubscriptionCard";

vi.mock("../lib/api/pushNotifications", () => ({
    ensurePushSubscription: () => Promise.reject(new Error("not used in these tests")),
    getExistingPushSubscription: () => Promise.resolve(null),
    getPushAvailability: () => Promise.resolve({
        apiAvailable: true,
        dbAvailable: true,
        alertsAvailable: true,
        reason: null,
    }),
    hasMatchingPushRule: () => Promise.resolve(false),
    isWebPushSupported: () => true,
    upsertPushRule: () => Promise.resolve(),
}));

const summary = (
    status: OccupancySummary["status"],
    percent: number | null,
    overrides: Partial<OccupancySummary> = {}
): OccupancySummary => ({
    count: percent === null ? null : 25,
    observedCapacity: percent === null ? 0 : 100,
    expectedOpenCapacity: 100,
    coverage: status === "live" ? 1 : status === "partial" ? 0.6 : 0,
    percent,
    observedLocations: percent === null ? 0 : 1,
    expectedLocations: 1,
    latestFetchedAt: percent === null ? null : "2026-08-31T12:00:00Z",
    oldestFetchedAt: percent === null ? null : "2026-08-31T12:00:00Z",
    status,
    ...overrides,
});

const option = (
    key: string,
    occupancySummary: OccupancySummary
): AlertSectionOption => ({key, label: key, summary: occupancySummary});

describe("resolveInitialSectionKey", () => {
    it("clears an insufficient saved selection and selects the first live option", () => {
        window.localStorage.setItem("reclive:crowd-alert-subscriptions", JSON.stringify({
            "1186": {sectionKey: "overall", threshold: 20},
        }));

        expect(resolveInitialSectionKey(1186, [
            option("overall", summary("insufficient", null)),
            option("weights", summary("live", 25)),
        ])).toBe("weights");
    });

    it("preserves a saved partial selection", () => {
        window.localStorage.setItem("reclive:crowd-alert-subscriptions", JSON.stringify({
            "1186": {sectionKey: "courts", threshold: 20},
        }));

        expect(resolveInitialSectionKey(1186, [
            option("overall", summary("live", 25)),
            option("courts", summary("partial", 40)),
        ])).toBe("courts");
    });

    it("returns no selection when no option has a usable percentage", () => {
        expect(resolveInitialSectionKey(1186, [
            option("overall", summary("unknown", null)),
            option("weights", summary("insufficient", null)),
            option("closed", summary("closed", null)),
        ])).toBe("");
    });
});

describe("CrowdAlertSubscriptionCard", () => {
    it("keeps an invalid selection cleared after its summary becomes usable again", () => {
        const liveOverall = option("overall", summary("live", 25));
        const unavailableOverall = option("overall", summary("insufficient", null));
        const liveWeights = option("weights", summary("live", 40));
        const props = {
            facility: 1186 as const,
            isOpen: false,
            onClose: vi.fn(),
        };
        const {rerender} = render(
            <CrowdAlertSubscriptionCard {...props} sections={[liveOverall, liveWeights]} />
        );

        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("overall");

        rerender(
            <CrowdAlertSubscriptionCard {...props} sections={[unavailableOverall, liveWeights]} />
        );
        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("weights");

        rerender(
            <CrowdAlertSubscriptionCard {...props} sections={[liveOverall, liveWeights]} />
        );
        expect(screen.getByRole("combobox", {name: "Gym area"})).toHaveTextContent("weights");
    });

    it("caps an over-capacity alert threshold at 100 percent", () => {
        render(
            <CrowdAlertSubscriptionCard
                facility={1186}
                isOpen={false}
                onClose={vi.fn()}
                sections={[option("overall", summary("live", 150, {
                    count: 150,
                    observedCapacity: 100,
                }))]}
            />
        );

        const thresholdInput = screen.getByLabelText("Alert threshold (%)");
        const submitButton = screen.getByRole("button", {name: "Set alert"});
        expect(thresholdInput).toHaveAttribute("max", "100");
        expect(submitButton).toBeEnabled();

        fireEvent.change(thresholdInput, {target: {value: "101"}});
        expect(screen.getByText("Enter a number between 1 and 100.")).toBeInTheDocument();
        expect(submitButton).toBeDisabled();

        fireEvent.change(thresholdInput, {target: {value: "100"}});
        expect(submitButton).toBeEnabled();
    });
});
