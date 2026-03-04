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

export interface ForecastOccupancyThresholds {
    lowMax: number;
    peakMin: number;
}

export interface ForecastHour {
    hourStart: string;
    expectedCount: number;
    expectedPct?: number | null;
    actualCount?: number | null;
    actualPct?: number | null;
    actualSampleCount?: number;
    actualCoverage?: number | null;
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
    forecastDayStartHour?: number;
    forecastDayEndHour?: number;
    occupancyThresholds?: ForecastOccupancyThresholds | null;
    sectionOccupancyThresholds?: Record<string, ForecastOccupancyThresholds> | null;
    locationOccupancyThresholds?: Record<string, ForecastOccupancyThresholds> | null;
    weeklyForecast: ForecastDay[];
}
