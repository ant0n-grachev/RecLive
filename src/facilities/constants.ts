import type {FacilityId} from "../lib/types/facility";

export const NICK_FITNESS = [5761, 5760, 5762, 5758] as const;
export const NICK_POOL = [5764] as const;
export const NICK_TRACK = [5763] as const;
export const NICK_COURTS = [7089, 7090, 5766] as const;
export const NICK_RACQUETBALL = [5753, 5754] as const;

export const BAKKE_FITNESS = [8718, 8717, 8705, 8700, 8699, 8696] as const;
export const BAKKE_ICE = [10550] as const;
export const BAKKE_POOL = [8716] as const;
export const BAKKE_TRACK = [8694] as const;
export const BAKKE_MENDOTA = [8701] as const;
export const BAKKE_COURTS = [8720, 8714, 8698] as const;
export const BAKKE_ESPORTS = [8712] as const;
export const BAKKE_SKYBOX = [8695] as const;

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

const nickSections: readonly SectionLayout[] = [
    {title: "🏋️ Fitness Floors", ids: NICK_FITNESS},
    {title: "🏀 Basketball Courts", ids: NICK_COURTS},
    [
        {title: "👟 Running Track", ids: NICK_TRACK},
        {title: "🏊‍♀️ Swimming Pool", ids: NICK_POOL},
    ],
    {title: "🎾 Racquetball Courts", ids: NICK_RACQUETBALL},
];

const bakkeSections: readonly SectionLayout[] = [
    {title: "🏋️ Fitness Floors", ids: BAKKE_FITNESS},
    {title: "🏀 Basketball Courts", ids: BAKKE_COURTS},
    [
        {title: "👟 Running Track", ids: BAKKE_TRACK},
        {title: "🏊‍♂️ Swimming Pool", ids: BAKKE_POOL},
    ],
    [
        {title: "🧗 Rock Climbing", ids: BAKKE_MENDOTA},
        {title: "🧊 Ice Skating", ids: BAKKE_ICE},
    ],
    [
        {title: "🎮 Esports Room", ids: BAKKE_ESPORTS},
        {title: "⛳ Sports Simulators", ids: BAKKE_SKYBOX},
    ],
];

export const FACILITY_DASHBOARD_CONFIG: Record<FacilityId, FacilityDashboardConfig> = {
    1186: {
        sections: nickSections,
        otherTitle: "🧩 Other Spaces",
    },
    1656: {
        sections: bakkeSections,
        otherTitle: "🧩 Other Spaces",
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
