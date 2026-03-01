export interface ForecastWindow {
    start: string;
    end: string;
    startHour?: number;
    endHour?: number;
    windowHours?: number;
    expectedTotal?: number;
    expectedAvg?: number;
    sampleCountMin?: number;
}

export interface ForecastBand {
    start: string;
    end: string;
    level: "low" | "medium" | "peak";
}

export interface ForecastHour {
    hourStart: string;
    expectedCount: number;
    expectedPct?: number | null;
}

export interface ForecastCategoryDay {
    key: string;
    title: string;
    maxCapacity?: number | null;
    hours: ForecastHour[];
}

export interface ForecastDay {
    dayName: string;
    date: string;
    categories?: ForecastCategoryDay[];
    avoidWindows?: ForecastWindow[];
    bestWindows?: ForecastWindow[];
    crowdBands?: ForecastBand[];
}

export interface FacilityForecastResponse {
    facilityId: number;
    facilityName: string;
    weeklyForecast: ForecastDay[];
}
