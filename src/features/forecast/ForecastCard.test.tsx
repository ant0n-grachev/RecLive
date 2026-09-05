import {ThemeProvider} from "@mui/material";
import {fireEvent, render, screen} from "@testing-library/react";
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

        fireEvent.click(screen.getByRole("button", {name: "Show hourly chart"}));
        expect(screen.getByRole("img", {name: "People histogram by hourly forecast bar"})).toBeInTheDocument();
        expect(screen.getByText("Now")).toBeInTheDocument();

        const bar = screen.getAllByRole("button").find((element) => /people$/.test(element.getAttribute("aria-label") ?? ""));
        expect(bar).toBeDefined();
        fireEvent.keyDown(bar!, {key: "Enter"});
        expect(screen.getByText(/AM – .*AM: \d+$/)).toBeInTheDocument();

        fireEvent.click(screen.getByRole("button", {name: "PEAK"}));
        expect(screen.getByText("No matching intervals.")).toBeInTheDocument();
        expect(screen.getByText("Tap a bar to inspect the hourly trend")).toBeInTheDocument();
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
