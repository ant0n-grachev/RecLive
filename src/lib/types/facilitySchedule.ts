import type {FacilityId} from "./facility";

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
    generatedAt: string | null;
    sourceSite: string | null;
    sourceFetchedAt: string | null;
    lastSuccessfulAt: string | null;
    stale: boolean;
    errorCategory: "anti_bot" | "upstream_timeout" | "upstream_http" | "wp_payload_invalid" | "parse_empty" | "schema_invalid" | "io_error" | null;
    facilityId: FacilityId;
    facilityName: string;
    slug: string;
    url: string;
    resolvedUrl?: string | null;
    status: "ok" | "stale";
    source: "direct_html" | "wp_json";
    sourceModifiedGmt?: string | null;
    sections: FacilityHoursSection[];
    error: string | null;
    updatedAt: string | null;
}
