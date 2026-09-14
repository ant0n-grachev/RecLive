import {render, screen} from "@testing-library/react";
import {ThemeProvider} from "@mui/material";
import {createAppTheme} from "../../app/theme";
import ForecastWindowsList from "./ForecastWindowsList";

const rgb = (color: string) => color.match(/[\d.]+/g)!.map(Number);
const luminance = (channels: number[]) => channels.slice(0, 3).map((n) => {
    const c = n / 255;
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}).reduce((sum, c, i) => sum + c * [0.2126, 0.7152, 0.0722][i], 0);

it.each(["light", "dark"] as const)("all populated crowd captions meet 4.5:1 in %s theme", (mode) => {
    const theme = createAppTheme(mode);
    const bands = (["low", "medium", "peak"] as const).map((level, i) => ({level,
        start: `2026-08-31T${10 + i}:00:00-05:00`, end: `2026-08-31T${11 + i}:00:00-05:00`,
    }));
    render(<ThemeProvider theme={theme}><ForecastWindowsList workingHoursBands={bands} displayBands={bands}
        filteredBestWindows={[]} filteredAvoidWindows={[]} nowTs={0} isDark={mode === "dark"}/></ThemeProvider>);
    // Actual caption background is composited over the card's paper surface.
    const paper = document.createElement("div");
    paper.style.color = theme.palette.background.paper;
    document.body.append(paper);
    const surface = rgb(getComputedStyle(paper).color);
    paper.remove();
    for (const level of ["LOW", "MEDIUM", "PEAK"]) {
        const style = getComputedStyle(screen.getByText(`${level} CROWD`).parentElement!);
        const foreground = rgb(style.color);
        const background = rgb(style.backgroundColor);
        const opacity = background[3] ?? 1;
        const composite = background.slice(0, 3).map((channel, i) => channel * opacity + surface[i] * (1 - opacity));
        const lights = [luminance(foreground), luminance(composite)].sort((a, b) => b - a);
        expect((lights[0] + 0.05) / (lights[1] + 0.05), level).toBeGreaterThanOrEqual(4.5);
    }
});
