import {useEffect, useMemo, useState} from "react";
import ReactDOM from "react-dom/client";
import {BrowserRouter} from "react-router-dom";
import {useRegisterSW} from "virtual:pwa-register/react";
import {ThemeProvider, CssBaseline} from "@mui/material";
import type {PaletteMode} from "@mui/material/styles";
import {createAppTheme} from "./app/theme";
import AppRoutes from "./app/AppRoutes";
import {PwaUpdatePrompt} from "./facilities/PwaUpdatePrompt";
import {reportServiceWorkerRegistered, reportServiceWorkerRegistrationError} from "./lib/serviceWorkerReadiness";

import "@fontsource/roboto/400.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/700.css";

const THEME_MODE_STORAGE_KEY = "reclive:themeMode";

const getStoredThemeMode = (): PaletteMode => {
    if (typeof window === "undefined") return "light";

    try {
        const stored = window.localStorage.getItem(THEME_MODE_STORAGE_KEY);
        return stored === "dark" ? "dark" : "light";
    } catch {
        // Ignore storage read failures (private mode/quota exceeded).
        return "light";
    }
};

export function Root() {
    const {
        needRefresh: [needRefresh, setNeedRefresh],
        offlineReady: [offlineReady, setOfflineReady],
        updateServiceWorker,
    } = useRegisterSW({
        immediate: true,
        onRegisteredSW: (_url, registration) => {
            if (registration) reportServiceWorkerRegistered();
            else reportServiceWorkerRegistrationError();
        },
        onRegisterError: reportServiceWorkerRegistrationError,
    });
    const [themeMode, setThemeMode] = useState<PaletteMode>(() => getStoredThemeMode());
    const theme = useMemo(() => createAppTheme(themeMode), [themeMode]);

    const dismissPwaPrompt = () => {
        setNeedRefresh(false);
        setOfflineReady(false);
    };

    useEffect(() => {
        if (typeof window === "undefined") return;
        try {
            window.localStorage.setItem(THEME_MODE_STORAGE_KEY, themeMode);
        } catch {
            // Ignore storage write failures (private mode/quota exceeded).
        }
    }, [themeMode]);

    return (
        <ThemeProvider theme={theme}>
            <CssBaseline/>
            <BrowserRouter>
                <AppRoutes themeMode={themeMode} onThemeModeChange={setThemeMode}/>
                <PwaUpdatePrompt
                    needRefresh={needRefresh}
                    offlineReady={offlineReady}
                    updateServiceWorker={updateServiceWorker}
                    onDismiss={dismissPwaPrompt}
                />
            </BrowserRouter>
        </ThemeProvider>
    );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<Root/>);
