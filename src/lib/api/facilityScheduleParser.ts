import axios from "axios";
import {env} from "../config/env";
import type {FacilityId} from "../types/facility";
import type {FacilityHoursFacilityPayload} from "../types/facilitySchedule";

const FORECAST_API_BASE_URL = env.forecastApiBaseUrl;
const SCHEDULE_REQUEST_TIMEOUT_MS = 10_000;

const asText = (value: unknown): string => (
    typeof value === "string" ? value.trim() : ""
);

const normalizeRows = (value: unknown): Array<{label: string; hours: string}> => {
    if (!Array.isArray(value)) return [];

    return value
        .map((row) => {
            if (!row || typeof row !== "object") {
                return null;
            }

            const label = asText((row as Record<string, unknown>).label);
            const hours = asText((row as Record<string, unknown>).hours);
            if (!label || !hours) {
                return null;
            }

            return {label, hours};
        })
        .filter((row): row is {label: string; hours: string} => Boolean(row));
};

const normalizeSections = (value: unknown): FacilityHoursFacilityPayload["sections"] => {
    if (!Array.isArray(value)) return [];

    const output: FacilityHoursFacilityPayload["sections"] = [];
    for (const section of value) {
        if (!section || typeof section !== "object") {
            continue;
        }

        const sectionRecord = section as Record<string, unknown>;
        const title = asText(sectionRecord.title);
        if (!title) {
            continue;
        }

        const rows = normalizeRows(sectionRecord.rows);
        const note = asText(sectionRecord.note) || null;
        output.push({
            title,
            rows,
            note,
        });
    }
    return output;
};

export async function fetchFacilityHours(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<FacilityHoursFacilityPayload> {
    const url = `${FORECAST_API_BASE_URL}/api/facility-hours/facilities/${facilityId}`;
    const resp = await axios.get<unknown>(url, {
        signal,
        timeout: SCHEDULE_REQUEST_TIMEOUT_MS,
    });

    const payload = resp.data;
    if (!payload || typeof payload !== "object") {
        throw new Error("Facility hours payload is invalid");
    }

    const row = payload as Record<string, unknown>;
    const parsedFacilityId = Number(row.facilityId);
    if (!Number.isInteger(parsedFacilityId)) {
        throw new Error("Facility hours facilityId is missing");
    }

    return {
        generatedAt: asText(row.generatedAt) || null,
        sourceSite: asText(row.sourceSite) || null,
        facilityId: parsedFacilityId,
        facilityName: asText(row.facilityName) || `Facility ${parsedFacilityId}`,
        slug: asText(row.slug) || null,
        url: asText(row.url) || null,
        resolvedUrl: asText(row.resolvedUrl) || null,
        status: asText(row.status) || null,
        source: asText(row.source) || null,
        sections: normalizeSections(row.sections),
        error: asText(row.error) || null,
        updatedAt: asText(row.updatedAt) || null,
    };
}
