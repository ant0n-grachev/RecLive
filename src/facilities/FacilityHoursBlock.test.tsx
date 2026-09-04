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
    it("labels preserved schedule data as stale without hiding its rows", () => {
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
        expect(screen.getByText(
            "Official hours may be out of date. Showing the last verified schedule.",
        )).toBeVisible();
        expect(screen.getByText("6:00 am - 10:00 pm")).toBeVisible();
        expect(screen.queryByText("anti_bot")).not.toBeInTheDocument();
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
