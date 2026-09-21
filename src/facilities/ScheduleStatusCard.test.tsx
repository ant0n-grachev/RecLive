import {screen} from "@testing-library/react";
import {renderWithApp} from "../test/render";
import ScheduleStatusCard from "./ScheduleStatusCard";

describe("ScheduleStatusCard", () => {
    it("hides the card when schedule status is unknown", () => {
        renderWithApp(<ScheduleStatusCard status={{state: "unknown", matchedRule: null}}/>);

        expect(screen.queryByText("Schedule Status")).not.toBeInTheDocument();
        expect(screen.queryByLabelText("Opening hours unavailable")).not.toBeInTheDocument();
    });

    it.each([
        ["open", "Open now according to the official schedule"],
        ["closed", "Closed now according to the official schedule"],
    ] as const)("preserves an explicit %s schedule status", (state, label) => {
        renderWithApp(<ScheduleStatusCard status={{state, matchedRule: null}}/>);

        expect(screen.getByText(label)).toBeVisible();
        expect(screen.getByText(state.toUpperCase())).toBeVisible();
    });
});
