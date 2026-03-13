const CHICAGO_TIMEZONE = "America/Chicago";
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const DATE_TIME_PATTERN =
    /^(\d{4})-(\d{2})-(\d{2})(?:[T\s](\d{2})(?::(\d{2})(?::(\d{2})(?:\.\d+)?)?)?)?/;
const EXPLICIT_OFFSET_PATTERN = /(Z|[+-]\d{2}:?\d{2})$/i;

const chicagoDateFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: CHICAGO_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
});

const chicagoDateTimePartsFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: CHICAGO_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hourCycle: "h23",
});

const chicagoTimeFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: CHICAGO_TIMEZONE,
    hour: "numeric",
    minute: "2-digit",
});
const chicagoHourFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: CHICAGO_TIMEZONE,
    hour: "2-digit",
    hourCycle: "h23",
});

const chicagoDateTimeFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: CHICAGO_TIMEZONE,
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
});

const getUtcDayIndex = (year: number, month: number, day: number): number | null => {
    const utc = Date.UTC(year, month - 1, day);
    const check = new Date(utc);
    if (
        check.getUTCFullYear() !== year
        || check.getUTCMonth() !== month - 1
        || check.getUTCDate() !== day
    ) {
        return null;
    }
    return Math.floor(utc / MS_PER_DAY);
};

const getChicagoDayIndexFromDate = (value: Date): number | null => {
    if (Number.isNaN(value.getTime())) return null;

    const parts = chicagoDateFormatter.formatToParts(value);
    const year = Number(parts.find((part) => part.type === "year")?.value);
    const month = Number(parts.find((part) => part.type === "month")?.value);
    const day = Number(parts.find((part) => part.type === "day")?.value);

    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
        return null;
    }

    return getUtcDayIndex(year, month, day);
};

interface NaiveDateTimeParts {
    year: number;
    month: number;
    day: number;
    hour: number;
    minute: number;
    second: number;
}

const parseNaiveDateTimeParts = (value: string): NaiveDateTimeParts | null => {
    const match = value.match(DATE_TIME_PATTERN);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    const hour = Number(match[4] ?? "0");
    const minute = Number(match[5] ?? "0");
    const second = Number(match[6] ?? "0");

    if (
        !Number.isInteger(year)
        || !Number.isInteger(month)
        || !Number.isInteger(day)
        || !Number.isInteger(hour)
        || !Number.isInteger(minute)
        || !Number.isInteger(second)
    ) {
        return null;
    }

    if (
        getUtcDayIndex(year, month, day) === null
        || hour < 0
        || hour > 23
        || minute < 0
        || minute > 59
        || second < 0
        || second > 59
    ) {
        return null;
    }

    return {year, month, day, hour, minute, second};
};

const getChicagoOffsetMinutes = (utcMs: number): number | null => {
    const parts = chicagoDateTimePartsFormatter.formatToParts(new Date(utcMs));
    const year = Number(parts.find((part) => part.type === "year")?.value);
    const month = Number(parts.find((part) => part.type === "month")?.value);
    const day = Number(parts.find((part) => part.type === "day")?.value);
    const hour = Number(parts.find((part) => part.type === "hour")?.value);
    const minute = Number(parts.find((part) => part.type === "minute")?.value);
    const second = Number(parts.find((part) => part.type === "second")?.value);

    if (
        !Number.isInteger(year)
        || !Number.isInteger(month)
        || !Number.isInteger(day)
        || !Number.isInteger(hour)
        || !Number.isInteger(minute)
        || !Number.isInteger(second)
    ) {
        return null;
    }

    const chicagoAsUtc = Date.UTC(year, month - 1, day, hour, minute, second);
    return (chicagoAsUtc - utcMs) / (60 * 1000);
};

const chicagoLocalPartsToUtcMs = (parts: NaiveDateTimeParts): number | null => {
    const baseUtcMs = Date.UTC(parts.year, parts.month - 1, parts.day, parts.hour, parts.minute, parts.second);
    let candidateMs = baseUtcMs;

    for (let i = 0; i < 4; i += 1) {
        const offsetMinutes = getChicagoOffsetMinutes(candidateMs);
        if (offsetMinutes === null) return null;

        const nextMs = baseUtcMs - offsetMinutes * 60 * 1000;
        if (nextMs === candidateMs) {
            break;
        }
        candidateMs = nextMs;
    }

    return candidateMs;
};

export const getChicagoTimestampMs = (value: string | null | undefined): number | null => {
    if (!value) return null;

    const text = value.trim();
    if (!text) return null;

    if (EXPLICIT_OFFSET_PATTERN.test(text)) {
        const parsed = new Date(text);
        return Number.isNaN(parsed.getTime()) ? null : parsed.getTime();
    }

    const naiveParts = parseNaiveDateTimeParts(text);
    if (naiveParts) {
        return chicagoLocalPartsToUtcMs(naiveParts);
    }
    return null;
};

const getChicagoDayIndexFromTimestamp = (value: string): number | null => {
    const timestampMs = getChicagoTimestampMs(value);
    if (timestampMs === null) return null;
    return getChicagoDayIndexFromDate(new Date(timestampMs));
};

export const getChicagoDayAge = (value: string | null | undefined, now = new Date()): number | null => {
    if (!value) return null;

    const nowDayIndex = getChicagoDayIndexFromDate(now);
    const valueDayIndex = getChicagoDayIndexFromTimestamp(value);
    if (nowDayIndex === null || valueDayIndex === null) return null;

    return Math.max(0, nowDayIndex - valueDayIndex);
};

export const formatChicagoTime = (value: string | null | undefined): string | null => {
    const timestampMs = getChicagoTimestampMs(value);
    if (timestampMs === null) return null;
    return chicagoTimeFormatter.format(new Date(timestampMs));
};

export const formatChicagoUpdatedRelative = (
    value: string | null | undefined,
    now: Date = new Date()
): string | null => {
    const formattedTime = formatChicagoTime(value);
    if (!formattedTime) return null;

    const daysAgo = getChicagoDayAge(value, now);
    if (daysAgo === 0) return `Updated today at ${formattedTime}`;
    if (daysAgo === 1) return `Updated yesterday at ${formattedTime}`;
    if (typeof daysAgo === "number") return `Updated ${daysAgo} days ago at ${formattedTime}`;
    return `Updated at ${formattedTime}`;
};

export const formatChicagoDateTime = (value: string | null | undefined): string | null => {
    const timestampMs = getChicagoTimestampMs(value);
    if (timestampMs === null) return null;
    return chicagoDateTimeFormatter.format(new Date(timestampMs));
};

export const getChicagoHour = (value: Date = new Date()): number | null => {
    const parts = chicagoHourFormatter.formatToParts(value);
    const hourText = parts.find((part) => part.type === "hour")?.value;
    if (!hourText) return null;

    const hour = Number(hourText);
    if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
        return null;
    }

    return hour;
};

export const isWithinChicagoHours = (
    startHourInclusive: number,
    endHourExclusive: number,
    value: Date = new Date()
): boolean => {
    const hour = getChicagoHour(value);
    if (hour === null) return false;
    return hour >= startHourInclusive && hour < endHourExclusive;
};
