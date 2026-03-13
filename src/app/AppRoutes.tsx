import {Suspense, lazy} from "react";
import {Box, CircularProgress} from "@mui/material";
import {Navigate, Route, Routes, useLocation, useNavigate} from "react-router-dom";
import type {PaletteMode} from "@mui/material/styles";
import type {FacilityId} from "../lib/types/facility";
import {facilityPath} from "./seo";

const FACILITY_STORAGE_KEY = "reclive:selectedFacility";
const DashboardApp = lazy(() => import("./App"));

const parseFacilityQuery = (search: string): FacilityId | null => {
    const params = new URLSearchParams(search);
    const raw = Number(params.get("facility"));
    if (raw === 1186 || raw === 1656) return raw;
    return null;
};

const storedFacility = (): FacilityId => {
    if (typeof window === "undefined") return 1186;
    try {
        const raw = Number(window.localStorage.getItem(FACILITY_STORAGE_KEY));
        return raw === 1656 ? 1656 : 1186;
    } catch {
        return 1186;
    }
};

interface AppRoutesProps {
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}

function RootRedirect() {
    const location = useLocation();
    const fromQuery = parseFacilityQuery(location.search);
    const target = fromQuery ?? storedFacility();
    return <Navigate replace to={facilityPath(target)}/>;
}

function FacilityRoutePage({
    facility,
    themeMode,
    onThemeModeChange,
}: {
    facility: FacilityId;
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}) {
    const navigate = useNavigate();
    return (
        <Suspense
            fallback={(
                <Box sx={{minHeight: "100vh", display: "grid", placeItems: "center"}}>
                    <CircularProgress size={28} thickness={5}/>
                </Box>
            )}
        >
            <DashboardApp
                initialFacility={facility}
                onFacilityRouteChange={(next) => navigate(facilityPath(next))}
                themeMode={themeMode}
                onThemeModeChange={onThemeModeChange}
            />
        </Suspense>
    );
}

export default function AppRoutes({themeMode, onThemeModeChange}: AppRoutesProps) {
    return (
        <Routes>
            <Route path="/" element={<RootRedirect/>}/>
            <Route
                path="/nick"
                element={(
                    <FacilityRoutePage
                        facility={1186}
                        themeMode={themeMode}
                        onThemeModeChange={onThemeModeChange}
                    />
                )}
            />
            <Route
                path="/bakke"
                element={(
                    <FacilityRoutePage
                        facility={1656}
                        themeMode={themeMode}
                        onThemeModeChange={onThemeModeChange}
                    />
                )}
            />
            <Route path="*" element={<Navigate replace to="/nick"/>}/>
        </Routes>
    );
}
