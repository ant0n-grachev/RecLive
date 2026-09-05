import type {OccupancySummary} from "../../shared/occupancy/computeOccupancySummary";

export interface AlertSectionOption {
    key: string;
    label: string;
    summary: OccupancySummary;
}
