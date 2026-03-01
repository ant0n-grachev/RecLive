import axios from "axios";
import type {FacilityId} from "../types/facility";
import type {FacilityForecastResponse, ForecastDay} from "../types/forecast";

const DEFAULT_FORECAST_API_BASE_URL = "https://api.grachev.us";

const FORECAST_API_BASE_URL = (
    import.meta.env.VITE_FORECAST_API_BASE_URL || DEFAULT_FORECAST_API_BASE_URL
).replace(/\/+$/, "");

const CHICAGO_TIMEZONE = "America/Chicago";

const getChicagoDateISO = (value = new Date()): string => {
    const parts = new Intl.DateTimeFormat("en-US", {
        timeZone: CHICAGO_TIMEZONE,
        year: "numeric",
        month: "2-digit",
        day: "2-digit",
    }).formatToParts(value);

    const year = parts.find((part) => part.type === "year")?.value;
    const month = parts.find((part) => part.type === "month")?.value;
    const day = parts.find((part) => part.type === "day")?.value;

    if (!year || !month || !day) {
        return value.toISOString().slice(0, 10);
    }

    return `${year}-${month}-${day}`;
};

export async function fetchForecastDays(
    facilityId: FacilityId,
    signal?: AbortSignal
): Promise<ForecastDay[]> {
    const today = getChicagoDateISO();
    const url = `${FORECAST_API_BASE_URL}/api/forecast/facilities/${facilityId}`;

    const resp = await axios.get<FacilityForecastResponse>(url, {signal});
    const weekly = resp.data?.weeklyForecast;
    if (!Array.isArray(weekly)) {
        throw new Error("Forecast weekly payload missing");
    }

    return weekly
        .filter((day) => day?.date >= today)
        .sort((a, b) => a.date.localeCompare(b.date));
}
