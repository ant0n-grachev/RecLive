import {createTheme} from "@mui/material/styles";

const baseFontFamily = "\"Roboto\", \"Helvetica\", \"Arial\", sans-serif";
const backgroundColor = "#f5f5f5";

export const theme = createTheme({
    palette: {
        background: {
            default: backgroundColor,
            paper: "#ffffff",
        },
    },
    typography: {
        fontFamily: baseFontFamily,
        h4: {
            fontWeight: 700,
        },
        h5: {
            fontWeight: 600,
        },
        h6: {
            fontWeight: 600,
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
                },
                body: {
                    margin: 0,
                    padding: 0,
                    backgroundColor,
                },
                "#root": {
                    height: "100%",
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
