import AcUnitIcon from "@mui/icons-material/AcUnit";
import AppsIcon from "@mui/icons-material/Apps";
import DirectionsRunIcon from "@mui/icons-material/DirectionsRun";
import FitnessCenterIcon from "@mui/icons-material/FitnessCenter";
import PoolIcon from "@mui/icons-material/Pool";
import SportsBasketballIcon from "@mui/icons-material/SportsBasketball";
import SportsEsportsIcon from "@mui/icons-material/SportsEsports";
import SportsGolfIcon from "@mui/icons-material/SportsGolf";
import SportsTennisIcon from "@mui/icons-material/SportsTennis";
import TerrainIcon from "@mui/icons-material/Terrain";
import type {ReactNode} from "react";

const normalize = (value: string): string => value.trim().toLowerCase();

export interface SectionVisual {
    icon: ReactNode;
    color: string;
    bg: string;
}

const SECTION_VISUAL_MAP: Record<string, SectionVisual> = {
    "fitness floors": {
        icon: <FitnessCenterIcon sx={{fontSize: 18}}/>,
        color: "#9a3412",
        bg: "rgba(234, 88, 12, 0.14)",
    },
    "basketball courts": {
        icon: <SportsBasketballIcon sx={{fontSize: 18}}/>,
        color: "#1d4ed8",
        bg: "rgba(37, 99, 235, 0.14)",
    },
    "running track": {
        icon: <DirectionsRunIcon sx={{fontSize: 18}}/>,
        color: "#7c3aed",
        bg: "rgba(124, 58, 237, 0.14)",
    },
    "swimming pool": {
        icon: <PoolIcon sx={{fontSize: 18}}/>,
        color: "#0369a1",
        bg: "rgba(14, 116, 144, 0.14)",
    },
    "racquetball courts": {
        icon: <SportsTennisIcon sx={{fontSize: 18}}/>,
        color: "#047857",
        bg: "rgba(16, 185, 129, 0.14)",
    },
    "rock climbing": {
        icon: <TerrainIcon sx={{fontSize: 18}}/>,
        color: "#92400e",
        bg: "rgba(217, 119, 6, 0.16)",
    },
    "ice skating": {
        icon: <AcUnitIcon sx={{fontSize: 18}}/>,
        color: "#0f766e",
        bg: "rgba(13, 148, 136, 0.14)",
    },
    "esports room": {
        icon: <SportsEsportsIcon sx={{fontSize: 18}}/>,
        color: "#6d28d9",
        bg: "rgba(147, 51, 234, 0.14)",
    },
    "sports simulators": {
        icon: <SportsGolfIcon sx={{fontSize: 18}}/>,
        color: "#854d0e",
        bg: "rgba(202, 138, 4, 0.16)",
    },
    "other spaces": {
        icon: <AppsIcon sx={{fontSize: 18}}/>,
        color: "#334155",
        bg: "rgba(71, 85, 105, 0.14)",
    },
};

export const getSectionVisual = (title: string): SectionVisual | null => {
    return SECTION_VISUAL_MAP[normalize(title)] ?? null;
};
