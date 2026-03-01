import React from "react";
import ReactDOM from "react-dom/client";
import App from "./app/App";

import {ThemeProvider, CssBaseline} from "@mui/material";
import {theme} from "./app/theme";

import "@fontsource/roboto/400.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/700.css";

ReactDOM.createRoot(document.getElementById("root")!).render(
    <React.StrictMode>
        <ThemeProvider theme={theme}>
            <CssBaseline/>
            <App/>
        </ThemeProvider>
    </React.StrictMode>
);
