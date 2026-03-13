import {Box} from "@mui/material";
import {alpha, useTheme} from "@mui/material/styles";
import {motion, useReducedMotion} from "framer-motion";
import type {FacilityId} from "../lib/types/facility";

interface Props {
    facility: FacilityId;
    onSelect: (f: FacilityId) => void;
}

const OPTIONS: {value: FacilityId; label: string}[] = [
    {value: 1186, label: "Nick"},
    {value: 1656, label: "Bakke"},
];

export default function FacilitySelector({facility, onSelect}: Props) {
    const reduceMotion = useReducedMotion();
    const theme = useTheme();
    const isDark = theme.palette.mode === "dark";
    const selectorTrackBg = isDark ? alpha(theme.palette.common.white, 0.08) : "#ececec";
    const selectorTrackBorder = isDark
        ? alpha(theme.palette.common.white, 0.14)
        : alpha(theme.palette.common.black, 0.06);
    const focusOutline = alpha(theme.palette.primary.main, isDark ? 0.75 : 0.7);
    const selectedPillShadow = isDark
        ? "0 1px 2px rgba(2, 6, 23, 0.42), 0 8px 20px rgba(2, 6, 23, 0.33)"
        : "0 1px 2px rgba(0, 0, 0, 0.08), 0 8px 20px rgba(0, 0, 0, 0.06)";

    return (
        <Box
            role="group"
            aria-label="Facility selector"
            sx={{
                width: "100%",
                position: "relative",
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: 0.5,
                p: 0.5,
                borderRadius: 999,
                bgcolor: selectorTrackBg,
                border: "1px solid",
                borderColor: selectorTrackBorder,
            }}
        >
            {OPTIONS.map((option) => {
                const isSelected = option.value === facility;

                return (
                    <Box
                        key={option.value}
                        component="button"
                        type="button"
                        onClick={() => onSelect(option.value)}
                        aria-pressed={isSelected}
                        sx={{
                            position: "relative",
                            appearance: "none",
                            border: 0,
                            background: "transparent",
                            cursor: "pointer",
                            borderRadius: 999,
                            flex: 1,
                            minHeight: 44,
                            px: 2,
                            py: 0.85,
                            fontWeight: 600,
                            fontSize: "0.95rem",
                            lineHeight: 1,
                            color: isSelected ? "text.primary" : "text.secondary",
                            transition: "color 180ms ease",
                            zIndex: 1,
                            "&:focus-visible": {
                                outline: `2px solid ${focusOutline}`,
                                outlineOffset: 1,
                            },
                        }}
                    >
                        {isSelected && (
                            <Box
                                component={motion.span}
                                layoutId="facility-selector-pill"
                                transition={reduceMotion
                                    ? {duration: 0.12}
                                    : {
                                        type: "spring",
                                        stiffness: 520,
                                        damping: 38,
                                        mass: 0.8,
                                    }}
                                sx={{
                                    position: "absolute",
                                    inset: 0,
                                    borderRadius: 999,
                                    bgcolor: "background.paper",
                                    boxShadow: selectedPillShadow,
                                    zIndex: -1,
                                }}
                            />
                        )}
                        {option.label}
                    </Box>
                );
            })}
        </Box>
    );
}
