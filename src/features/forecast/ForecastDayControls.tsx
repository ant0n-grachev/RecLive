import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import {Box, IconButton, Stack} from "@mui/material";

interface Props {
    variant: "buttons" | "pagination";
    canPrev: boolean;
    canNext: boolean;
    isLoading: boolean;
    onPrev: () => void;
    onNext: () => void;
    totalDays: number;
    dayOffset: number;
}

export default function ForecastDayControls({
    variant,
    canPrev,
    canNext,
    isLoading,
    onPrev,
    onNext,
    totalDays,
    dayOffset,
}: Props) {
    if (variant === "buttons") {
        return (
            <Stack direction="row" spacing={0.25} sx={{display: {xs: "none", sm: "flex"}}}>
                <IconButton
                    size="medium"
                    onClick={onPrev}
                    disabled={!canPrev || isLoading}
                    aria-label="Previous forecast day"
                    sx={{width: 44, height: 44}}
                >
                    <ChevronLeftIcon fontSize="small"/>
                </IconButton>
                <IconButton
                    size="medium"
                    onClick={onNext}
                    disabled={!canNext || isLoading}
                    aria-label="Next forecast day"
                    sx={{width: 44, height: 44}}
                >
                    <ChevronRightIcon fontSize="small"/>
                </IconButton>
            </Stack>
        );
    }

    if (totalDays <= 1) return null;

    return (
        <Stack direction="row" spacing={0.75} justifyContent="center" sx={{mt: 1.5}}>
            {Array.from({length: totalDays}).map((_, index) => {
                const isActive = index === dayOffset;
                return (
                    <Box
                        key={`day-dot-${index}`}
                        sx={{
                            width: isActive ? 16 : 6,
                            height: 6,
                            borderRadius: 999,
                            bgcolor: isActive ? "text.primary" : "divider",
                            transition: "all 180ms ease",
                        }}
                    />
                );
            })}
        </Stack>
    );
}
