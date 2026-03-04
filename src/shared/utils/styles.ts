import type {Theme} from "@mui/material/styles";

export type OccupancyTone = "success" | "warning" | "error";
export interface OccupancyThresholds {
    lowMax: number;
    peakMin: number;
}

export const OCCUPANCY_MAIN_HEX: Record<OccupancyTone, string> = {
    success: "#2e7d32",
    warning: "#ca8a04",
    error: "#d32f2f",
};

export const OCCUPANCY_SOFT_BG: Record<OccupancyTone, string> = {
    success: "rgba(46, 125, 50, 0.12)",
    warning: "rgba(202, 138, 4, 0.18)",
    error: "rgba(211, 47, 47, 0.12)",
};

const normalizeThresholds = (thresholds?: OccupancyThresholds | null): OccupancyThresholds | null => {
    if (!thresholds) return null;

    const lowRaw = Number.isFinite(thresholds.lowMax) ? thresholds.lowMax : null;
    const peakRaw = Number.isFinite(thresholds.peakMin) ? thresholds.peakMin : null;
    if (lowRaw === null || peakRaw === null) {
        return null;
    }
    const lowMax = Math.max(0, Math.min(99, lowRaw));
    const peakMin = Math.max(lowMax + 1, Math.min(100, peakRaw));

    return {lowMax, peakMin};
};

export const getOccupancyTone = (percent: number, thresholds?: OccupancyThresholds | null): OccupancyTone | null => {
    const normalized = normalizeThresholds(thresholds);
    if (!normalized) {
        return null;
    }
    const {lowMax, peakMin} = normalized;
    if (percent <= lowMax) return "success";
    if (percent < peakMin) return "warning";
    return "error";
};

export const getOccupancyColor = (
    percent: number,
    thresholds?: OccupancyThresholds | null
): "success.main" | "warning.main" | "error.main" | "text.secondary" => {
    const tone = getOccupancyTone(percent, thresholds);
    if (tone === null) return "text.secondary";
    if (tone === "success") return "success.main";
    if (tone === "warning") return "warning.main";
    return "error.main";
};

export const combineOccupancyThresholds = (
    entries: Array<{thresholds?: OccupancyThresholds | null; weight?: number | null}>
): OccupancyThresholds | null => {
    let totalWeight = 0;
    let lowWeightedSum = 0;
    let peakWeightedSum = 0;

    for (const entry of entries) {
        if (!entry?.thresholds) continue;
        const weight = Number(entry.weight);
        if (!Number.isFinite(weight) || weight <= 0) continue;

        totalWeight += weight;
        lowWeightedSum += entry.thresholds.lowMax * weight;
        peakWeightedSum += entry.thresholds.peakMin * weight;
    }

    if (totalWeight <= 0) {
        return null;
    }

    const lowMax = Math.max(0, Math.min(99, Math.round(lowWeightedSum / totalWeight)));
    const peakMin = Math.max(lowMax + 1, Math.min(100, Math.round(peakWeightedSum / totalWeight)));
    return {lowMax, peakMin};
};

export const INNER_SURFACE_SX = {
    borderRadius: 3,
    border: "1px solid",
    borderColor: "divider",
    bgcolor: "background.paper",
} as const;

export const CARD_SHELL_SX = {
    borderRadius: 3,
    border: "1px solid",
    borderColor: "divider",
    bgcolor: "background.paper",
    boxShadow: (theme: Theme) => (
        theme.palette.mode === "dark"
            ? "0 8px 22px rgba(2, 6, 23, 0.42)"
            : "0 6px 14px rgba(0, 0, 0, 0.06), 0 14px 28px rgba(0, 0, 0, 0.05)"
    ),
} as const;

export const CARD_TITLE_SX = {
    fontSize: "1.2rem",
    fontWeight: 700,
    lineHeight: 1.2,
} as const;

export const clampPercent = (value?: number | null): number => {
    if (value === undefined || value === null || Number.isNaN(value)) {
        return 0;
    }

    return Math.max(0, Math.round(value));
};
