import {ThemeProvider} from "@mui/material";
import {fireEvent, render, screen, within} from "@testing-library/react";
import {createAppTheme} from "../../app/theme";
import {fixtureForecastDays, fixtureThresholds} from "../../test/fixtures/dashboard";
import ForecastCard from "./ForecastCard";

const renderCard = (overrides: Partial<React.ComponentProps<typeof ForecastCard>> = {}) => {
    const props: React.ComponentProps<typeof ForecastCard> = {
        day: fixtureForecastDays[0],
        comparisonDays: fixtureForecastDays,
        occupancyThresholds: fixtureThresholds,
        dayOffset: 0,
        totalDays: 3,
        canPrev: false,
        canNext: true,
        onPrev: vi.fn(),
        onNext: vi.fn(),
        isLoading: false,
        error: null,
        ...overrides,
    };

    return {
        ...render(
            <ThemeProvider theme={createAppTheme("light")}>
                <ForecastCard {...props}/>
            </ThemeProvider>
        ),
        props,
    };
};

describe("ForecastCard", () => {
    beforeEach(() => {
        vi.useFakeTimers();
        vi.setSystemTime(new Date("2026-08-31T13:15:00Z"));
    });

    it("hides a failed forecast without rendering diagnostics or old forecast values", () => {
        renderCard({error: "Forecast service temporarily unavailable."});
        expect(screen.queryByText("Forecast Today")).not.toBeInTheDocument();
        expect(screen.queryByRole("alert")).not.toBeInTheDocument();
        expect(screen.queryByText(/Forecast service/)).not.toBeInTheDocument();
        expect(screen.queryByRole("button", {name: "Show crowd chart"})).not.toBeInTheDocument();
    });

    it("hides a forecast day with no usable forecast information", () => {
        renderCard({day: {...fixtureForecastDays[0], totalHours: [], categories: [], crowdBands: [], bestWindows: [], avoidWindows: []}});
        expect(screen.queryByText("Forecast Today")).not.toBeInTheDocument();
        expect(screen.queryByText(/crowd bands are unavailable/)).not.toBeInTheDocument();
    });

    it("keeps the crowd chart usable without internal threshold explanations", () => {
        renderCard({occupancyThresholds: null});
        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        expect(screen.getByRole("img", {name: "People by half hour"})).toBeVisible();
        expect(screen.queryByText(/half-hour colors|occupancy thresholds were missing/)).not.toBeInTheDocument();
    });

    it("preserves day controls, horizontal swipe, filtering, keyboard bars, and the current marker", () => {
        const {container, props} = renderCard();

        expect(screen.getByText("Forecast Today")).toBeInTheDocument();
        expect(screen.getByRole("button", {name: "Previous forecast day"})).toBeDisabled();
        fireEvent.click(screen.getByRole("button", {name: "Next forecast day"}));
        expect(props.onNext).toHaveBeenCalledTimes(1);

        const swipeArea = container.querySelector('[data-disable-pull-refresh="true"]');
        expect(swipeArea).not.toBeNull();
        fireEvent.touchStart(swipeArea!, {touches: [{clientX: 180, clientY: 100}]});
        fireEvent.touchMove(swipeArea!, {touches: [{clientX: 100, clientY: 104}]});
        fireEvent.touchEnd(swipeArea!, {changedTouches: [{clientX: 100, clientY: 104}]});
        expect(props.onNext).toHaveBeenCalledTimes(2);

        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        expect(screen.getByRole("img", {name: "People by half hour"})).toBeInTheDocument();
        expect(screen.getByText("Now")).toBeInTheDocument();

        const bar = screen.getAllByRole("button").find((element) => /people$/.test(element.getAttribute("aria-label") ?? ""));
        expect(bar).toBeDefined();
        fireEvent.keyDown(bar!, {key: "Enter"});
        expect(screen.getByText(/AM – .*AM: \d+ people \(Forecast\)$/)).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", {name: "PEAK"}));
        expect(screen.getByText("No matching intervals.")).toBeInTheDocument();
        expect(screen.getByText("Tap a bar for its count and source")).toBeInTheDocument();
    });

    it("keeps a short peak visible with the matching half-hour count and range", () => {
        renderCard({
            day: {date: "2026-08-31", dayName: "Monday", totalHours: [
                {hourStart: "2026-08-31T10:00:00-05:00", expectedCount: 237, expectedPct: 0.237},
                {hourStart: "2026-08-31T10:30:00-05:00", expectedCount: 261, expectedPct: 0.261},
                {hourStart: "2026-08-31T11:00:00-05:00", expectedCount: 237, expectedPct: 0.237},
            ]},
            comparisonDays: [],
            occupancyThresholds: {lowMax: 11, peakMin: 26},
        });
        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        const chart = screen.getByRole("img", {name: "People by half hour"});
        const medium = within(chart).getByRole("button", {name: "10:00 AM – 10:30 AM, Forecast, 237 people"});
        const peak = within(chart).getByRole("button", {name: "10:30 AM – 11:00 AM, Forecast, 261 people"});
        const mediumRect = medium.querySelector("rect")!;
        const peakRect = peak.querySelector("rect")!;
        expect(Number(peakRect.getAttribute("height")) / Number(mediumRect.getAttribute("height"))).toBeCloseTo(261 / 237);
        expect(peakRect.getAttribute("fill")).not.toBe(mediumRect.getAttribute("fill"));
        fireEvent.click(peak);
        expect(screen.getByText("10:30 AM – 11:00 AM: 261 people (Forecast)")).toBeVisible();

        fireEvent.click(screen.getByRole("button", {name: "PEAK"}));
        expect(screen.getByText("10:30 AM – 11:00 AM")).toBeVisible();
        expect(within(chart).getAllByRole("button")).toEqual([peak]);
    });

    it("identifies completed observations and unavailable observations in bar details", () => {
        renderCard({day: {date: "2026-08-31", dayName: "Monday", totalHours: [
            {hourStart: "2026-08-31T07:00:00-05:00", expectedCount: 90, expectedPct: 0.45, actualCount: 40, actualPct: 0.2},
            {hourStart: "2026-08-31T07:30:00-05:00", expectedCount: 90, expectedPct: 0.45, actualCount: null, actualPct: null},
        ]}});
        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        const chart = screen.getByRole("img", {name: "People by half hour"});
        fireEvent.click(within(chart).getByRole("button", {name: "7:00 AM – 7:30 AM, Actual, 40 people"}));
        expect(screen.getByText("7:00 AM – 7:30 AM: 40 people (Actual)")).toBeVisible();
        fireEvent.click(within(chart).getByRole("button", {name: "7:30 AM – 8:00 AM, Forecast, 90 people"}));
        expect(screen.getByText("7:30 AM – 8:00 AM: 90 people (Forecast)")).toBeVisible();
    });

    it("does not show forecast crowd ranges for an unclassified actual bar", () => {
        renderCard({day: {...fixtureForecastDays[0], totalHours: [{
            hourStart: "2026-08-31T07:00:00-05:00", expectedCount: 90, actualCount: 40,
        }], crowdBands: [{start: "2026-08-31T07:00:00-05:00", end: "2026-08-31T08:00:00-05:00", level: "peak"}]}});

        expect(screen.queryByText("PEAK CROWD")).not.toBeInTheDocument();
        expect(screen.queryByText(/\(PEAK CROWD\)/)).not.toBeInTheDocument();
        expect(screen.queryByRole("button", {name: "PEAK"})).not.toBeInTheDocument();
        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        expect(screen.getByRole("button", {name: "7:00 AM – 7:30 AM, Actual, 40 people"})).toBeVisible();
    });

    it("keeps refreshed actual bars visible when the selected crowd filter becomes unavailable", () => {
        const forecastHour = {hourStart: "2026-08-31T07:00:00-05:00", expectedCount: 20, expectedPct: 0.1};
        const day = {...fixtureForecastDays[0], totalHours: [], categories: [{
            key: "fitness floors", title: "Fitness Floors", hours: [forecastHour],
        }], crowdBands: [{start: "2026-08-31T07:00:00-05:00", end: "2026-08-31T07:30:00-05:00", level: "low" as const}]};
        const {props, rerender} = renderCard({day, comparisonDays: []});
        fireEvent.click(screen.getByRole("button", {name: "LOW"}));
        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        fireEvent.click(screen.getByRole("button", {name: "7:00 AM – 7:30 AM, Forecast, 20 people"}));

        const actualDay = {...day, categories: [{...day.categories[0], hours: [{
            ...forecastHour, actualCount: 40, actualPct: 0.2,
            observedCount: 40, observedCapacity: 200, expectedCapacity: 200,
            actualCoverage: 1, temporalCoverage: 1, coverageThreshold: 0.75,
        }]}]};
        rerender(
            <ThemeProvider theme={createAppTheme("light")}>
                <ForecastCard {...props} day={actualDay}/>
            </ThemeProvider>,
        );

        expect(screen.queryByRole("button", {name: "LOW"})).not.toBeInTheDocument();
        expect(screen.getByRole("button", {name: "7:00 AM – 7:30 AM, Actual, 40 people"})).toBeVisible();
        expect(screen.getByText("7:00 AM – 7:30 AM: 40 people (Actual)")).toBeVisible();
    });

    it("selects the correct count when two bars share a fall-back clock time", () => {
        vi.setSystemTime(new Date("2026-11-01T01:15:00-06:00"));
        renderCard({day: {date: "2026-11-01", dayName: "Sunday", totalHours: [
            {hourStart: "2026-11-01T01:00:00-05:00", expectedCount: 90, actualCount: 30, actualPct: 0.15},
            {hourStart: "2026-11-01T01:00:00-06:00", expectedCount: 90, expectedPct: 0.45, actualCount: 40, actualPct: 0.2},
        ]}, comparisonDays: []});
        fireEvent.click(screen.getByRole("button", {name: "Show crowd chart"}));
        fireEvent.click(screen.getByRole("button", {name: "1:00 AM – 1:30 AM, Actual, 30 people"}));
        expect(screen.getByText("1:00 AM – 1:30 AM: 30 people (Actual)")).toBeVisible();
        fireEvent.click(screen.getByRole("button", {name: "1:00 AM – 1:30 AM, Forecast, 90 people"}));
        expect(screen.getByText("1:00 AM – 1:30 AM: 90 people (Forecast)")).toBeVisible();
    });

    it("preserves best and avoid window fallback copy", () => {
        renderCard({
            day: {
                ...fixtureForecastDays[0],
                totalHours: [],
                categories: [],
                crowdBands: [],
            },
        });

        expect(screen.getByText(/\(LOW CROWD\)$/)).toBeInTheDocument();
        expect(screen.getByText(/\(PEAK CROWD\)$/)).toBeInTheDocument();
    });
});
