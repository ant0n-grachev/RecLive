import {useMediaQuery} from "@mui/material";
import {useTheme, type PaletteMode} from "@mui/material/styles";
import {DashboardPage} from "../features/dashboard/DashboardPage";
import {useDashboardState} from "../features/dashboard/useDashboardState";
import type {FacilityId} from "../lib/types/facility";
import {useFacilitySeo} from "./seo";

export interface AppProps {
    initialFacility?: FacilityId;
    onFacilityRouteChange?: (facility: FacilityId) => void;
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}

export default function App({
    initialFacility,
    onFacilityRouteChange,
    themeMode,
    onThemeModeChange,
}: AppProps) {
    const theme = useTheme();
    const isPhoneViewport = useMediaQuery(theme.breakpoints.down("sm"));
    const state = useDashboardState({initialFacility, onFacilityRouteChange, isPhoneViewport});
    useFacilitySeo(state.facility);

    return (
        <DashboardPage
            state={state}
            view={state.view}
            themeMode={themeMode}
            onThemeModeChange={onThemeModeChange}
        />
    );
}
