export type FacilityId = 1186 | 1656;

export interface Location {
    facilityId: FacilityId;
    locationId: number;
    locationName: string;
    floor: number;
    isClosed: boolean | null;
    currentCapacity: number | null;
    maxCapacity: number | null;
    lastUpdated: string | null;
}

export interface FacilityPayload {
    facilityId: FacilityId;
    facilityName: string;
    floors: Record<number, Location[]>;
    locations: Location[];
}
