import test from "node:test";
import assert from "node:assert/strict";
import {
    getFacilityOpenStatus,
    getFacilityOpenWindowsForDate,
} from "./facilityScheduleStatus.ts";
import type {FacilityHoursFacilityPayload} from "../../lib/types/facilitySchedule.ts";

const buildSchedule = (
    sections: FacilityHoursFacilityPayload["sections"]
): FacilityHoursFacilityPayload => ({
    facilityId: 1186,
    facilityName: "Nick",
    sections,
});

test("keeps overnight Friday hours open after midnight on Saturday", () => {
    const schedule = buildSchedule([
        {
            title: "Facility Hours",
            rows: [
                {label: "Friday", hours: "8:00 PM - 1:00 AM"},
                {label: "Saturday", hours: "9:00 AM - 11:00 PM"},
            ],
        },
    ]);

    const status = getFacilityOpenStatus(
        schedule,
        new Date("2026-03-07T06:30:00Z")
    );

    assert.equal(status.state, "open");
    assert.equal(status.matchedRule?.label, "Friday");
});

test("returns same-day windows plus overnight spillover for the next date", () => {
    const schedule = buildSchedule([
        {
            title: "Facility Hours",
            rows: [
                {label: "Friday", hours: "8:00 PM - 1:00 AM"},
                {label: "Saturday", hours: "9:00 AM - 11:00 PM"},
            ],
        },
    ]);

    const windows = getFacilityOpenWindowsForDate(schedule, "2026-03-07");

    assert.deepEqual(windows, [
        {startMinutes: 0, endMinutes: 60},
        {startMinutes: 540, endMinutes: 1380},
    ]);
});

test("lets a same-day date-specific closure override the previous day's overnight hours", () => {
    const schedule = buildSchedule([
        {
            title: "Spring Break Hours",
            rows: [
                {label: "Mar 7, 2026", hours: "Closed"},
            ],
        },
        {
            title: "Facility Hours",
            rows: [
                {label: "Friday", hours: "8:00 PM - 1:00 AM"},
                {label: "Saturday", hours: "9:00 AM - 11:00 PM"},
            ],
        },
    ]);

    const status = getFacilityOpenStatus(
        schedule,
        new Date("2026-03-07T06:30:00Z")
    );
    const windows = getFacilityOpenWindowsForDate(schedule, "2026-03-07");

    assert.equal(status.state, "closed");
    assert.equal(status.matchedRule?.label, "Mar 7, 2026");
    assert.deepEqual(windows, []);
});
