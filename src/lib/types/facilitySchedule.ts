export interface FacilityHoursRow {
    label: string;
    hours: string;
}

export interface FacilityHoursSection {
    title: string;
    rows: FacilityHoursRow[];
    note?: string | null;
}

export interface FacilityHoursFacilityPayload {
    generatedAt?: string | null;
    sourceSite?: string | null;
    facilityId: number;
    facilityName: string;
    slug?: string | null;
    url?: string | null;
    resolvedUrl?: string | null;
    status?: string | null;
    source?: string | null;
    sections: FacilityHoursSection[];
    error?: string | null;
    updatedAt?: string | null;
}
