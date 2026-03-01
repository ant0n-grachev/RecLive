export const getOccupancyColor = (percent: number): string => {
    if (percent < 40) return "success.main";
    if (percent < 70) return "warning.main";
    if (percent < 90) return "warning.dark";
    return "error.main";
};

export const clampPercent = (value?: number | null): number => {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return 0;
    }

    return Math.max(0, Math.round(value));
};
