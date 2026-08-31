import type {ReactElement} from "react";
import {render, type RenderOptions, type RenderResult} from "@testing-library/react";
import {CssBaseline, ThemeProvider} from "@mui/material";
import {MemoryRouter} from "react-router-dom";
import {createAppTheme} from "../app/theme";

export function renderWithApp(
    ui: ReactElement,
    {route = "/nick", ...options}: RenderOptions & {route?: string} = {},
): RenderResult {
    return render(
        <ThemeProvider theme={createAppTheme("light")}>
            <CssBaseline />
            <MemoryRouter initialEntries={[route]}>{ui}</MemoryRouter>
        </ThemeProvider>,
        options,
    );
}
