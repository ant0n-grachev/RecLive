import {useCallback} from "react";
import {Button} from "@mui/material";
import DarkModeRoundedIcon from "@mui/icons-material/DarkModeRounded";
import LightModeRoundedIcon from "@mui/icons-material/LightModeRounded";
import {alpha, useTheme, type PaletteMode} from "@mui/material/styles";

interface ThemeModeToggleProps {
    themeMode: PaletteMode;
    onThemeModeChange: (mode: PaletteMode) => void;
}

export default function ThemeModeToggle({themeMode, onThemeModeChange}: ThemeModeToggleProps) {
    const theme = useTheme();
    const hoverThemeButtonBg = alpha(theme.palette.text.primary, theme.palette.mode === "dark" ? 0.14 : 0.06);
    const handleThemeToggle = useCallback(
        () => onThemeModeChange(themeMode === "dark" ? "light" : "dark"),
        [onThemeModeChange, themeMode]
    );
    const activeThemeLabel = themeMode === "dark" ? "Dark" : "Light";
    const activeThemeIcon = themeMode === "dark"
        ? <DarkModeRoundedIcon sx={{fontSize: 16, mr: 0.5}}/>
        : <LightModeRoundedIcon sx={{fontSize: 16, mr: 0.5}}/>;

    return (
        <Button
            size="small"
            onClick={handleThemeToggle}
            aria-label={`Switch to ${themeMode === "dark" ? "light" : "dark"} theme`}
            sx={{
                minHeight: 36,
                px: 1.2,
                border: 0,
                borderRadius: 999,
                textTransform: "none",
                fontWeight: 700,
                color: "text.primary",
                bgcolor: "action.selected",
                "&:hover": {
                    border: 0,
                    bgcolor: hoverThemeButtonBg,
                },
            }}
        >
            {activeThemeIcon}
            {activeThemeLabel}
        </Button>
    );
}
