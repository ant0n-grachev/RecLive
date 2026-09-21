import {fireEvent, screen} from "@testing-library/react";
import {describe, expect, it} from "vitest";
import {renderWithApp} from "../test/render";
import type {FacilityHoursFacilityPayload} from "../lib/types/facilitySchedule";
import FacilityHoursBlock from "./FacilityHoursBlock";

const staleSchedule: FacilityHoursFacilityPayload = {
    generatedAt: "2026-08-31T12:00:00Z",
    sourceSite: "https://recwell.example.test",
    facilityId: 1186,
    facilityName: "Nick",
    slug: "nick",
    url: "https://recwell.example.test/nick/",
    resolvedUrl: "https://recwell.example.test/nick/",
    status: "stale",
    source: "direct_html",
    sourceModifiedGmt: null,
    sections: [{
        title: "Building Hours",
        rows: [{label: "Mon", hours: "6:00 am - 10:00 pm"}],
        note: null,
    }],
    sourceFetchedAt: "2026-08-30T12:00:00Z",
    lastSuccessfulAt: "2026-08-30T12:00:00Z",
    stale: true,
    error: "Official hours could not be refreshed.",
    errorCategory: "anti_bot",
    updatedAt: "2026-08-31T12:00:00Z",
};

describe("FacilityHoursBlock", () => {
    it("does not show an empty hours accordion while its first request is pending", () => {
        renderWithApp(<FacilityHoursBlock facilityName="Nick" isLoading error={null} schedule={null}/>);
        expect(screen.queryByRole("button", {name: /official hours, closures & notices/i})).not.toBeInTheDocument();
    });

    it("keeps preserved hours without an automatic stale-data notice", () => {
        renderWithApp(
            <FacilityHoursBlock
                facilityName="Nick"
                isLoading={false}
                error={null}
                schedule={staleSchedule}
            />,
        );

        const accordionButton = screen.getByRole("button", {
            name: /official hours, closures & notices/i,
        });
        expect(accordionButton).toHaveAttribute("aria-expanded", "false");
        fireEvent.click(accordionButton);

        expect(accordionButton).toHaveAttribute("aria-expanded", "true");
        expect(screen.queryByText(
            "Official hours may be out of date. Showing the last verified schedule.",
        )).not.toBeInTheDocument();
        expect(screen.getByText("6:00 am - 10:00 pm")).toBeVisible();
        expect(screen.queryByText("anti_bot")).not.toBeInTheDocument();
    });

    it.each([null, {...staleSchedule, sections: []}])("hides the hours card when no official information is available", (schedule) => {
        renderWithApp(<FacilityHoursBlock facilityName="Nick" isLoading={false} error="Internal failure" schedule={schedule}/>);

        expect(screen.queryByRole("button", {name: /official hours, closures & notices/i})).not.toBeInTheDocument();
        expect(screen.queryByText(/schedule is unavailable|No schedule rows|Internal failure/i)).not.toBeInTheDocument();
    });

    it("omits incomplete rows while keeping valid hours and notices", () => {
        renderWithApp(
            <FacilityHoursBlock
                facilityName="Nick"
                isLoading={false}
                error={null}
                schedule={{
                    ...staleSchedule,
                    sections: [{
                        title: "Building Hours",
                        rows: [
                            {label: "Mon", hours: "6:00 am - 10:00 pm"},
                            {label: "Tue", hours: ""},
                        ],
                        note: "Closed on university holidays.",
                    }],
                }}
            />,
        );
        fireEvent.click(screen.getByRole("button", {name: /official hours, closures & notices/i}));

        expect(screen.getByText("6:00 am - 10:00 pm")).toBeVisible();
        expect(screen.queryByText("Tue")).not.toBeInTheDocument();
        expect(screen.getByText("Closed on university holidays.")).toBeVisible();
    });

    it("does not label a fresh official schedule as stale", () => {
        renderWithApp(
            <FacilityHoursBlock
                facilityName="Nick"
                isLoading={false}
                error={null}
                schedule={{
                    ...staleSchedule,
                    status: "ok",
                    stale: false,
                    error: null,
                    errorCategory: null,
                }}
            />,
        );

        fireEvent.click(screen.getByRole("button", {
            name: /official hours, closures & notices/i,
        }));

        expect(screen.queryByText(
            "Official hours may be out of date. Showing the last verified schedule.",
        )).not.toBeInTheDocument();
        expect(screen.getByText("6:00 am - 10:00 pm")).toBeVisible();
    });
});
