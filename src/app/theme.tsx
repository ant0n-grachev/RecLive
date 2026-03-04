import {createTheme, type PaletteMode} from "@mui/material/styles";
import {OCCUPANCY_MAIN_HEX} from "../shared/utils/styles";

const baseFontFamily = "\"Roboto\", \"Helvetica\", \"Arial\", sans-serif";
const LIGHT_BACKGROUND_COLOR = "#f5f5f5";
const DARK_BACKGROUND_COLOR = "#0b1220";
const DARK_SURFACE_COLOR = "#111827";

export const createAppTheme = (mode: PaletteMode) => {
    const isDark = mode === "dark";
    const backgroundColor = isDark ? DARK_BACKGROUND_COLOR : LIGHT_BACKGROUND_COLOR;
    const surfaceColor = isDark ? DARK_SURFACE_COLOR : "#ffffff";

    return createTheme({
        palette: {
            mode,
            success: {
                light: "#66bb6a",
                main: OCCUPANCY_MAIN_HEX.success,
                dark: "#1b5e20",
                contrastText: "#ffffff",
            },
            warning: {
                light: "#facc15",
                main: OCCUPANCY_MAIN_HEX.warning,
                dark: "#a16207",
                contrastText: "#111827",
            },
            error: {
                light: "#ef5350",
                main: OCCUPANCY_MAIN_HEX.error,
                dark: "#b71c1c",
                contrastText: "#ffffff",
            },
            text: isDark
                ? {
                    primary: "#e2e8f0",
                    secondary: "#94a3b8",
                }
                : {
                    primary: "#0f172a",
                    secondary: "#475569",
                },
            divider: isDark ? "rgba(148, 163, 184, 0.24)" : "rgba(15, 23, 42, 0.14)",
            background: {
                default: backgroundColor,
                paper: surfaceColor,
            },
        },
        typography: {
            fontFamily: baseFontFamily,
            h4: {
                fontWeight: 700,
            },
            h5: {
                fontWeight: 700,
            },
            h6: {
                fontWeight: 700,
                fontSize: "1.2rem",
                lineHeight: 1.2,
            },
            subtitle2: {
                fontWeight: 600,
                textTransform: "uppercase",
                letterSpacing: 0.4,
            },
            body2: {
                fontSize: "0.95rem",
                lineHeight: 1.4,
            },
        },
        components: {
            MuiCssBaseline: {
                styleOverrides: {
                    html: {
                        margin: 0,
                        padding: 0,
                        overflowX: "hidden",
                    },
                    body: {
                        margin: 0,
                        padding: 0,
                        backgroundColor,
                        color: isDark ? "#e2e8f0" : "#0f172a",
                        overflowX: "hidden",
                    },
                    "#root": {
                        height: "100%",
                        overflowX: "hidden",
                    },
                },
            },
            MuiCard: {
                styleOverrides: {
                    root: {
                        borderRadius: 16,
                    },
                },
            },
        },
    });
};
