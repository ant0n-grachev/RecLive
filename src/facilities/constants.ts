import type {FacilityId} from "../lib/types/facility";
import {FACILITY_SHARED_CONFIG} from "../lib/config/facilitySections";

export interface SectionConfig {
    title: string;
    ids: readonly number[];
}

export type SectionLayout = SectionConfig | readonly [SectionConfig, SectionConfig];

export const isSectionRow = (
    layout: SectionLayout
): layout is readonly [SectionConfig, SectionConfig] => Array.isArray(layout);

export interface FacilityDashboardConfig {
    sections: readonly SectionLayout[];
    otherTitle: string;
}

const buildSectionLookup = (facilityId: FacilityId): Record<string, SectionConfig> => {
    const sections = FACILITY_SHARED_CONFIG[facilityId].sections;
    const lookup: Record<string, SectionConfig> = {};

    for (const section of sections) {
        lookup[section.key] = {
            title: section.title,
            ids: section.ids,
        };
    }

    return lookup;
};

const sectionOrThrow = (
    lookup: Record<string, SectionConfig>,
    facilityId: FacilityId,
    key: string
): SectionConfig => {
    const section = lookup[key];
    if (!section) {
        throw new Error(`Missing shared section '${key}' for facility ${facilityId}`);
    }
    return section;
};

const nickLookup = buildSectionLookup(1186);
const bakkeLookup = buildSectionLookup(1656);

const nickSections: readonly SectionLayout[] = [
    sectionOrThrow(nickLookup, 1186, "fitness floors"),
    sectionOrThrow(nickLookup, 1186, "basketball courts"),
    sectionOrThrow(nickLookup, 1186, "racquetball courts"),
    [
        sectionOrThrow(nickLookup, 1186, "running track"),
        sectionOrThrow(nickLookup, 1186, "swimming pool"),
    ],
];

const bakkeSections: readonly SectionLayout[] = [
    sectionOrThrow(bakkeLookup, 1656, "fitness floors"),
    sectionOrThrow(bakkeLookup, 1656, "basketball courts"),
    [
        sectionOrThrow(bakkeLookup, 1656, "running track"),
        sectionOrThrow(bakkeLookup, 1656, "swimming pool"),
    ],
    [
        sectionOrThrow(bakkeLookup, 1656, "rock climbing"),
        sectionOrThrow(bakkeLookup, 1656, "ice skating"),
    ],
    [
        sectionOrThrow(bakkeLookup, 1656, "esports room"),
        sectionOrThrow(bakkeLookup, 1656, "sports simulators"),
    ],
];

export const FACILITY_DASHBOARD_CONFIG: Record<FacilityId, FacilityDashboardConfig> = {
    1186: {
        sections: nickSections,
        otherTitle: "Other Spaces",
    },
    1656: {
        sections: bakkeSections,
        otherTitle: "Other Spaces",
    },
};

const collectKnownIds = (sections: readonly SectionLayout[]): number[] =>
    sections.flatMap((section) =>
        isSectionRow(section)
            ? section.flatMap((item) => [...item.ids])
            : [...section.ids]
    );

export const FACILITY_KNOWN_IDS: Record<FacilityId, number[]> = {
    1186: collectKnownIds(nickSections),
    1656: collectKnownIds(bakkeSections),
};
