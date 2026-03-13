import rawCapacities from "../../../shared/facility_capacities.json";

const FACILITY_CAPACITIES = rawCapacities as Record<string, number>;

export const getFacilityCapacity = (locationId: number): number | null => (
    FACILITY_CAPACITIES[String(locationId)] ?? null
);
