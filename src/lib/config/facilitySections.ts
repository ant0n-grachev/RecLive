import sharedConfigRaw from "../../../server/facility_sections.json";
import type {FacilityId} from "../types/facility";

interface SharedSectionRaw {
    key?: unknown;
    title?: unknown;
    ids?: unknown;
}

interface SharedFacilityRaw {
    shortName?: unknown;
    facilityName?: unknown;
    sections?: unknown;
}

interface SharedConfigRaw {
    facilities?: Record<string, SharedFacilityRaw>;
}

export interface SharedSection {
    key: string;
    title: string;
    ids: number[];
}

export interface SharedFacilityConfig {
    shortName: string;
    facilityName: string;
    sections: SharedSection[];
}

const normalizeKey = (value: string): string => value.trim().toLowerCase().replace(/\s+/g, " ");

const parseSection = (facilityId: FacilityId, raw: SharedSectionRaw, index: number): SharedSection => {
    const keyText = typeof raw.key === "string" ? normalizeKey(raw.key) : "";
    const titleText = typeof raw.title === "string" ? raw.title.trim() : "";
    const rawIds = Array.isArray(raw.ids) ? raw.ids : [];
    const ids = rawIds
        .map((item) => Number(item))
        .filter((item) => Number.isInteger(item) && item > 0);

    if (!keyText) {
        throw new Error(`Invalid shared section key at facility ${facilityId}, index ${index}`);
    }
    if (!titleText) {
        throw new Error(`Invalid shared section title at facility ${facilityId}, index ${index}`);
    }
    if (ids.length === 0) {
        throw new Error(`Invalid shared section ids at facility ${facilityId}, index ${index}`);
    }

    return {key: keyText, title: titleText, ids};
};

const parseFacility = (facilityId: FacilityId, raw: SharedFacilityRaw): SharedFacilityConfig => {
    const shortName = typeof raw.shortName === "string" ? raw.shortName.trim() : "";
    const facilityName = typeof raw.facilityName === "string" ? raw.facilityName.trim() : "";
    const rawSections = Array.isArray(raw.sections) ? raw.sections : [];
    const sections = rawSections.map((section, index) => parseSection(facilityId, section as SharedSectionRaw, index));

    if (!shortName) {
        throw new Error(`Missing shortName for facility ${facilityId}`);
    }
    if (!facilityName) {
        throw new Error(`Missing facilityName for facility ${facilityId}`);
    }
    if (sections.length === 0) {
        throw new Error(`Missing sections for facility ${facilityId}`);
    }

    return {shortName, facilityName, sections};
};

const parseSharedConfig = (): Record<FacilityId, SharedFacilityConfig> => {
    const config = sharedConfigRaw as SharedConfigRaw;
    const facilities = config.facilities ?? {};

    return {
        1186: parseFacility(1186, facilities["1186"] ?? {}),
        1656: parseFacility(1656, facilities["1656"] ?? {}),
    };
};

export const FACILITY_SHARED_CONFIG = parseSharedConfig();

export const FACILITY_SHORT_NAMES: Record<FacilityId, string> = {
    1186: FACILITY_SHARED_CONFIG[1186].shortName,
    1656: FACILITY_SHARED_CONFIG[1656].shortName,
};

export const FACILITY_DISPLAY_NAMES: Record<FacilityId, string> = {
    1186: FACILITY_SHARED_CONFIG[1186].facilityName,
    1656: FACILITY_SHARED_CONFIG[1656].facilityName,
};
