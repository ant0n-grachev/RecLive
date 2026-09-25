import type {FacilityId} from "../../lib/types/facility";

export interface Point {
    x: number;
    y: number;
}

export interface ZoneConfig {
    label: string;
    ids: readonly number[];
    corners: readonly [Point, Point, Point, ...Point[]];
}

export interface FloorMapConfig {
    image: string;
    aspectRatio: string;
    zoom: number;
    zones: readonly ZoneConfig[];
}

export const FLOOR_MAPS: Record<FacilityId, Partial<Record<number, FloorMapConfig>>> = {
    1186: {
        0: {
            image: "/floor-maps/nick_page_01.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Power House",
                    ids: [5761],
                    corners: [
                        {x: 40, y: 60.5},
                        {x: 70.5, y: 60.5},
                        {x: 70.5, y: 72.6},
                        {x: 40, y: 72.6},
                    ],
                },
                {
                    label: "Pool",
                    ids: [5764],
                    corners: [
                        {x: 50.5, y: 32.6},
                        {x: 70.5, y: 32.6},
                        {x: 70.5, y: 60.5},
                        {x: 50.5, y: 60.5},
                    ],
                },
            ],
        },
        1: {
            image: "/floor-maps/nick_page_02.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Level 1 Fitness",
                    ids: [5760],
                    corners: [
                        {x: 40, y: 32.8},
                        {x: 45, y: 32.7},
                        {x: 45, y: 57.5},
                        {x: 36.5, y: 57.5},
                    ],
                },
                {
                    label: "Courts 1 & 2",
                    ids: [7089],
                    corners: [
                        {x: 50.5, y: 72.5},
                        {x: 71, y: 72.5},
                        {x: 71, y: 59.5},
                        {x: 50.5, y: 59.5},
                    ],
                },
            ],
        },
        2: {
            image: "/floor-maps/nick_page_03.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Level 2 Fitness",
                    ids: [5762],
                    corners: [
                        {x: 39.5, y: 32},
                        {x: 45.5, y: 32},
                        {x: 45.5, y: 73},
                        {x: 38.5, y: 73},
                        {x: 36, y: 55.5},
                    ],
                },
            ],
        },
        3: {
            image: "/floor-maps/nick_page_04.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Level 3 Fitness",
                    ids: [5758],
                    corners: [
                        {x: 38.5, y: 32.5},
                        {x: 50.5, y: 32.5},
                        {x: 50.5, y: 73.5},
                        {x: 35.5, y: 73.5},
                        {x: 38.5, y: 73.5},
                        {x: 35.5, y: 56},
                    ],
                },
                {
                    label: "Courts 3-6",
                    ids: [7090],
                    corners: [
                        {x: 50.5, y: 31.5},
                        {x: 70.5, y: 31.5},
                        {x: 70.5, y: 59.7},
                        {x: 50.5, y: 59.7},
                    ],
                },
                {
                    label: "Courts 7 & 8",
                    ids: [5766],
                    corners: [
                        {x: 70.5, y: 59.7},
                        {x: 50.5, y: 59.7},
                        {x: 50.5, y: 72.5},
                        {x: 70.5, y: 72.5},
                    ],
                },
            ],
        },
        4: {
            image: "/floor-maps/nick_page_05.png",
            aspectRatio: "9 / 16",
            zoom: 2,
            zones: [
                {
                    label: "Track",
                    ids: [5763],
                    corners: [
                        {x: 38.5, y: 31},
                        {x: 70.5, y: 31},
                        {x: 70.5, y: 60},
                        {x: 35, y: 60},
                    ],
                },
                {
                    label: "Racquetball",
                    ids: [5753, 5754],
                    corners: [
                        {x: 40, y: 47},
                        {x: 44, y: 47},
                        {x: 44, y: 56.5},
                        {x: 40, y: 56.5},
                    ],
                },
            ],
        },
    },
    1656: {
        1: {
            image: "/floor-maps/bakke_page_01.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                // {
                //     label: "The Point",
                //     ids: [8718],
                //     corners: [
                //         {x: 22.5, y: 33},
                //         {x: 45.5, y: 33},
                //         {x: 45.5, y: 53},
                //         {x: 22.5, y: 53},
                //     ],
                // },
                {
                    label: "Level 1 Fitness",
                    ids: [8717],
                    corners: [
                        {x: 28.7, y: 38.5},
                        {x: 36, y: 38.5},
                        {x: 44.5, y: 30},
                        {x: 44.5, y: 48.5},
                        {x: 48, y: 48.5},
                        {x: 48, y: 51},
                        {x: 28.7, y: 51},
                    ],
                },
                {
                    label: "Courts 1 & 2",
                    ids: [8720],
                    corners: [
                        {x: 28.7, y: 51},
                        {x: 40, y: 51},
                        {x: 40, y: 72.5},
                        {x: 28.7, y: 72.5},
                    ],
                },
                {
                    label: "Courts 5-8",
                    ids: [8698],
                    corners: [
                        {x: 60.4, y: 30},
                        {x: 73, y: 30},
                        {x: 73, y: 48.5},
                        {x: 60.4, y: 48.5},
                    ],
                },
                {
                    label: "Cove Pool",
                    ids: [8716],
                    corners: [
                        {x: 44.5, y: 30},
                        {x: 60.4, y: 30},
                        {x: 60.4, y: 48.5},
                        {x: 44.5, y: 48.5},
                    ],
                },
                {
                    label: "Ice Center",
                    ids: [10550],
                    corners: [
                        {x: 48, y: 48.5},
                        {x: 70.5, y: 48.5},
                        {x: 70.5, y: 71.5},
                        {x: 48, y: 71.5},
                    ],
                },
            ],
        },
        2: {
            image: "/floor-maps/bakke_page_02.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                {
                    label: "Level 2 Fitness",
                    ids: [8705],
                    corners: [
                        {x: 28, y: 38.5},
                        {x: 37, y: 38.5},
                        {x: 45, y: 30},
                        {x: 45, y: 49},
                        {x: 42, y: 51.5},
                        {x: 41.5, y: 58},
                        {x: 28, y: 58},
                    ],
                },
                // {
                //     label: "Esports Room",
                //     ids: [8712],
                //     corners: [
                //         {x: 26.5, y: 37.5},
                //         {x: 41.5, y: 37.5},
                //         {x: 41.5, y: 52.5},
                //         {x: 26.5, y: 52.5},
                //     ],
                // },
            ],
        },
        3: {
            image: "/floor-maps/bakke_page_03.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                {
                    label: "Level 3 Fitness",
                    ids: [8700],
                    corners: [
                        {x: 27, y: 45},
                        {x: 38, y: 37.5},
                        {x: 45, y: 28},
                        {x: 48.5, y: 28},
                        {x: 48.5, y: 47},
                        {x: 42.5, y: 50},
                        {x: 39, y: 69},
                        {x: 28.5, y: 69},
                    ],
                },
                {
                    label: "Courts 3 & 4",
                    ids: [8714],
                    corners: [
                        {x: 48.5, y: 28},
                        {x: 72.5, y: 28},
                        {x: 72.5, y: 47},
                        {x: 48.5, y: 47},
                    ],
                },
                // {
                //     label: "Mount Mendota",
                //     ids: [8701],
                //     corners: [
                //         {x: 30, y: 37.5},
                //         {x: 48, y: 37.5},
                //         {x: 48, y: 52.5},
                //         {x: 30, y: 52.5},
                //     ],
                // },
            ],
        },
        4: {
            image: "/floor-maps/bakke_page_04.png",
            aspectRatio: "5 / 3",
            zoom: 2,
            zones: [
                {
                    label: "Track",
                    ids: [8694],
                    corners: [
                        {x: 48.5, y: 31.5},
                        {x: 72.7, y: 31.5},
                        {x: 72.7, y: 47.5},
                        {x: 48.5, y: 47.5},
                    ],
                },
                {
                    label: "Level 4 Fitness",
                    ids: [8699],
                    corners: [
                        {x: 28, y: 44},
                        {x: 38, y: 38},
                        {x: 45, y: 28.5},
                        {x: 55, y: 28.5},
                        {x: 55, y: 31.5},
                        {x: 48.5, y: 31.5},
                        {x: 45, y: 32.9},
                        {x: 45, y: 47.5},
                        {x: 39.3, y: 61.5},
                        {x: 28, y: 61.5},
                    ],
                },
                {
                    label: "Orbit",
                    ids: [8696],
                    corners: [
                        {x: 34, y: 48.8},
                        {x: 38.5, y: 48.8},
                        {x: 38.5, y: 57},
                        {x: 34, y: 57},
                    ],
                },
                {
                    label: "Skybox",
                    ids: [8695],
                    corners: [
                        {x: 45, y: 32.9},
                        {x: 48.5, y: 31.5},
                        {x: 48.5, y: 47.5},
                        {x: 45, y: 47.5},
                    ],
                },
            ],
        },
    },
};
