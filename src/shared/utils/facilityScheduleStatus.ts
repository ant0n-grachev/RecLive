import type {FacilityHoursFacilityPayload, FacilityHoursSection} from "../../lib/types/facilitySchedule";

const CHICAGO_TIMEZONE = "America/Chicago";
const MINUTES_PER_DAY = 24 * 60;

const chicagoPartsFormatter = new Intl.DateTimeFormat("en-US", {
    timeZone: CHICAGO_TIMEZONE,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    weekday: "long",
    hour: "2-digit",
    minute: "2-digit",
    hourCycle: "h23",
});

const weekdayByName: Record<string, number> = {
    sunday: 0,
    monday: 1,
    tuesday: 2,
    wednesday: 3,
    thursday: 4,
    friday: 5,
    saturday: 6,
};

const monthByName: Record<string, number> = {
    jan: 1,
    january: 1,
    feb: 2,
    february: 2,
    mar: 3,
    march: 3,
    apr: 4,
    april: 4,
    may: 5,
    jun: 6,
    june: 6,
    jul: 7,
    july: 7,
    aug: 8,
    august: 8,
    sep: 9,
    sept: 9,
    september: 9,
    oct: 10,
    october: 10,
    nov: 11,
    november: 11,
    dec: 12,
    december: 12,
};

interface ChicagoNowParts {
    year: number;
    month: number;
    day: number;
    weekday: number;
    minutes: number;
}

interface DateRange {
    startKey: string;
    endKey: string;
    spanDays: number;
}

interface TimeWindow {
    startMinutes: number;
    endMinutes: number;
    isClosed: boolean;
}

interface MatchedScheduleRule {
    sectionTitle: string;
    label: string;
    hours: string;
    isOpen: boolean;
}

export interface FacilityOpenStatus {
    state: "open" | "closed" | "unknown";
    matchedRule: MatchedScheduleRule | null;
}

export interface FacilityOpenWindow {
    startMinutes: number;
    endMinutes: number;
}

const pad2 = (value: number): string => String(value).padStart(2, "0");

const toDateKey = (year: number, month: number, day: number): string =>
    `${year}-${pad2(month)}-${pad2(day)}`;

const fromDateKey = (key: string): Date | null => {
    const match = key.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!match) return null;

    const year = Number(match[1]);
    const month = Number(match[2]);
    const day = Number(match[3]);
    if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
        return null;
    }

    const date = new Date(Date.UTC(year, month - 1, day));
    if (
        date.getUTCFullYear() !== year
        || date.getUTCMonth() !== month - 1
        || date.getUTCDate() !== day
    ) {
        return null;
    }
    return date;
};

const diffDaysFromDateKeys = (startKey: string, endKey: string): number | null => {
    const startDate = fromDateKey(startKey);
    const endDate = fromDateKey(endKey);
    if (!startDate || !endDate) return null;
    const ms = endDate.getTime() - startDate.getTime();
    return Math.floor(ms / (24 * 60 * 60 * 1000));
};

const normalizeText = (value: string): string =>
    value
        .toLowerCase()
        .replace(/[–—]/g, "-")
        .replace(/\s+/g, " ")
        .trim();

const parseChicagoNowParts = (value: Date = new Date()): ChicagoNowParts | null => {
    if (Number.isNaN(value.getTime())) return null;

    const parts = chicagoPartsFormatter.formatToParts(value);
    const year = Number(parts.find((part) => part.type === "year")?.value);
    const month = Number(parts.find((part) => part.type === "month")?.value);
    const day = Number(parts.find((part) => part.type === "day")?.value);
    const weekdayName = (parts.find((part) => part.type === "weekday")?.value ?? "").toLowerCase();
    const hour = Number(parts.find((part) => part.type === "hour")?.value);
    const minute = Number(parts.find((part) => part.type === "minute")?.value);

    const weekday = weekdayByName[weekdayName];
    if (
        !Number.isInteger(year)
        || !Number.isInteger(month)
        || !Number.isInteger(day)
        || !Number.isInteger(hour)
        || !Number.isInteger(minute)
        || !Number.isInteger(weekday)
    ) {
        return null;
    }

    return {
        year,
        month,
        day,
        weekday,
        minutes: Math.max(0, Math.min(MINUTES_PER_DAY - 1, hour * 60 + minute)),
    };
};

const parseTokenDate = (
    token: string,
    fallbackYear: number,
    fallbackMonth?: number
): {year: number; month: number; day: number} | null => {
    const normalized = normalizeText(token);
    if (!normalized) return null;

    const isoMatch = normalized.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (isoMatch) {
        const year = Number(isoMatch[1]);
        const month = Number(isoMatch[2]);
        const day = Number(isoMatch[3]);
        if (!Number.isInteger(year) || !Number.isInteger(month) || !Number.isInteger(day)) {
            return null;
        }
        const key = toDateKey(year, month, day);
        return fromDateKey(key) ? {year, month, day} : null;
    }

    const monthDayMatch = normalized.match(/^([a-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?$/);
    if (monthDayMatch) {
        const monthName = monthDayMatch[1];
        const month = monthByName[monthName];
        const day = Number(monthDayMatch[2]);
        const explicitYear = monthDayMatch[3] ? Number(monthDayMatch[3]) : fallbackYear;
        if (!Number.isInteger(month) || !Number.isInteger(day) || !Number.isInteger(explicitYear)) {
            return null;
        }
        const key = toDateKey(explicitYear, month, day);
        return fromDateKey(key) ? {year: explicitYear, month, day} : null;
    }

    const dayOnlyMatch = normalized.match(/^(\d{1,2})$/);
    if (dayOnlyMatch && typeof fallbackMonth === "number" && Number.isInteger(fallbackMonth)) {
        const day = Number(dayOnlyMatch[1]);
        const key = toDateKey(fallbackYear, fallbackMonth, day);
        return fromDateKey(key) ? {year: fallbackYear, month: fallbackMonth, day} : null;
    }

    return null;
};

const parseDateRangeLabel = (
    value: string,
    fallbackYear: number
): DateRange | null => {
    const normalized = normalizeText(value);
    if (!normalized) return null;

    const dateLike = /(\d{4}-\d{2}-\d{2})|([a-z]{3,9}\s+\d{1,2})/.test(normalized);
    if (!dateLike) return null;

    const parts = normalized.split(/\s*-\s*/).filter(Boolean);
    if (parts.length === 0) return null;

    const start = parseTokenDate(parts[0], fallbackYear);
    if (!start) return null;

    const endToken = parts.length > 1 ? parts.slice(1).join("-") : parts[0];
    const end = parseTokenDate(endToken, start.year, start.month) ?? parseTokenDate(endToken, fallbackYear);
    if (!end) return null;

    let startKey = toDateKey(start.year, start.month, start.day);
    let endKey = toDateKey(end.year, end.month, end.day);
    if (endKey < startKey) {
        const shiftedEnd = parseTokenDate(endToken, start.year + 1, start.month);
        if (shiftedEnd) {
            endKey = toDateKey(shiftedEnd.year, shiftedEnd.month, shiftedEnd.day);
        } else {
            const shiftedStart = parseTokenDate(parts[0], fallbackYear - 1);
            if (shiftedStart) {
                startKey = toDateKey(shiftedStart.year, shiftedStart.month, shiftedStart.day);
            }
        }
    }

    const span = diffDaysFromDateKeys(startKey, endKey);
    if (span === null || span < 0) return null;
    return {
        startKey,
        endKey,
        spanDays: span + 1,
    };
};

const parseWeekdaySet = (label: string): Set<number> | null => {
    const normalized = normalizeText(label);
    if (!normalized) return null;

    if (normalized.includes("daily")) {
        return new Set([0, 1, 2, 3, 4, 5, 6]);
    }
    if (normalized.includes("weekdays")) {
        return new Set([1, 2, 3, 4, 5]);
    }
    if (normalized.includes("weekends")) {
        return new Set([0, 6]);
    }

    const weekdayNames = Object.keys(weekdayByName);
    const rangeRegex = new RegExp(`(${weekdayNames.join("|")})\\s*-\\s*(${weekdayNames.join("|")})`);
    const rangeMatch = normalized.match(rangeRegex);
    if (rangeMatch) {
        const start = weekdayByName[rangeMatch[1]];
        const end = weekdayByName[rangeMatch[2]];
        const output = new Set<number>();
        if (start <= end) {
            for (let idx = start; idx <= end; idx += 1) {
                output.add(idx);
            }
        } else {
            for (let idx = start; idx <= 6; idx += 1) output.add(idx);
            for (let idx = 0; idx <= end; idx += 1) output.add(idx);
        }
        return output;
    }

    const output = new Set<number>();
    for (const [name, index] of Object.entries(weekdayByName)) {
        if (normalized.includes(name)) {
            output.add(index);
        }
    }
    return output.size > 0 ? output : null;
};

const parseClockToken = (token: string, isEnd: boolean): number | null => {
    const normalized = normalizeText(token).replace(/\./g, "");
    if (!normalized) return null;
    if (normalized === "midnight") {
        return isEnd ? MINUTES_PER_DAY : 0;
    }
    if (normalized === "noon") {
        return 12 * 60;
    }

    const match = normalized.match(/^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$/);
    if (!match) return null;

    let hour = Number(match[1]);
    const minute = Number(match[2] ?? "0");
    const suffix = match[3] ?? "";
    if (!Number.isInteger(hour) || !Number.isInteger(minute)) return null;
    if (minute < 0 || minute > 59) return null;

    if (suffix === "am") {
        if (hour === 12) hour = 0;
    } else if (suffix === "pm") {
        if (hour < 12) hour += 12;
    }

    if (hour < 0 || hour > 24) return null;
    if (hour === 24 && minute > 0) return null;
    return hour * 60 + minute;
};

const parseHoursWindow = (value: string): TimeWindow | null => {
    const normalized = normalizeText(value);
    if (!normalized) return null;

    if (normalized.includes("closed")) {
        return {
            startMinutes: 0,
            endMinutes: 0,
            isClosed: true,
        };
    }
    if (normalized.includes("24 hours")) {
        return {
            startMinutes: 0,
            endMinutes: MINUTES_PER_DAY,
            isClosed: false,
        };
    }

    const parts = normalized.split(/\s*-\s*/).filter(Boolean);
    if (parts.length !== 2) return null;

    const startMinutes = parseClockToken(parts[0], false);
    const endMinutesRaw = parseClockToken(parts[1], true);
    if (startMinutes === null || endMinutesRaw === null) return null;

    let endMinutes = endMinutesRaw;
    if (endMinutes <= startMinutes) {
        endMinutes += MINUTES_PER_DAY;
    }

    return {
        startMinutes,
        endMinutes,
        isClosed: false,
    };
};

const isWithinDateRange = (dayKey: string, range: DateRange): boolean =>
    dayKey >= range.startKey && dayKey <= range.endKey;

const isFacilityWideSection = (sectionTitle: string): boolean => {
    const title = normalizeText(sectionTitle);
    if (!title) return false;
    if (title === "seasonal notice") return false;
    if (title.includes("maintenance closures")) return true;

    const areaKeywords = [
        "ice rink",
        "sub zero",
        "pool",
        "court",
        "track",
        "climbing",
        "esports",
        "simulator",
        "fitness",
        "room",
    ];
    if (areaKeywords.some((keyword) => title.includes(keyword))) {
        return false;
    }
    return true;
};

const isOpenForWindow = (window: TimeWindow, chicagoMinutes: number): boolean => {
    if (window.isClosed) return false;

    if (window.endMinutes > MINUTES_PER_DAY) {
        return (
            chicagoMinutes >= window.startMinutes
            || chicagoMinutes < (window.endMinutes - MINUTES_PER_DAY)
        );
    }

    return chicagoMinutes >= window.startMinutes && chicagoMinutes < window.endMinutes;
};

interface CandidateRule {
    rule: MatchedScheduleRule;
    specificity: number;
    spanDays: number;
    sectionIndex: number;
    rowIndex: number;
}

interface CandidateRuleWindow {
    sectionTitle: string;
    label: string;
    hours: string;
    window: TimeWindow;
    specificity: number;
    spanDays: number;
    sectionIndex: number;
    rowIndex: number;
}

interface ResolvedRuleCandidate {
    rule: MatchedScheduleRule;
    window: TimeWindow;
    specificity: number;
    spanDays: number;
    sectionIndex: number;
    rowIndex: number;
}

const candidateComparator = (left: CandidateRule, right: CandidateRule): number => {
    if (left.specificity !== right.specificity) {
        return right.specificity - left.specificity;
    }
    if (left.spanDays !== right.spanDays) {
        return left.spanDays - right.spanDays;
    }
    if (left.sectionIndex !== right.sectionIndex) {
        return left.sectionIndex - right.sectionIndex;
    }
    return left.rowIndex - right.rowIndex;
};

const shiftDateKeyByDays = (dateKey: string, dayDelta: number): string | null => {
    const date = fromDateKey(dateKey);
    if (!date) return null;

    date.setUTCDate(date.getUTCDate() + dayDelta);
    return toDateKey(
        date.getUTCFullYear(),
        date.getUTCMonth() + 1,
        date.getUTCDate()
    );
};

const sectionRangeFromTitle = (section: FacilityHoursSection, fallbackYear: number): DateRange | null =>
    parseDateRangeLabel(section.title, fallbackYear);

const collectRuleWindowCandidates = (
    sections: FacilityHoursSection[],
    dayKey: string,
    weekday: number,
    fallbackYear: number
): CandidateRuleWindow[] => {
    const candidates: CandidateRuleWindow[] = [];

    sections.forEach((section, sectionIndex) => {
        if (!isFacilityWideSection(section.title)) return;

        const titleRange = sectionRangeFromTitle(section, fallbackYear);
        for (let rowIndex = 0; rowIndex < section.rows.length; rowIndex += 1) {
            const row = section.rows[rowIndex];
            const rowRange = parseDateRangeLabel(row.label, fallbackYear);
            const weekdaySet = rowRange ? null : parseWeekdaySet(row.label);

            let isDateMatch = false;
            let specificity = 1;
            let spanDays = 9999;

            if (rowRange) {
                isDateMatch = isWithinDateRange(dayKey, rowRange);
                specificity = 3;
                spanDays = Math.max(1, rowRange.spanDays);
            } else if (weekdaySet) {
                const titleRangeMatch = titleRange ? isWithinDateRange(dayKey, titleRange) : true;
                isDateMatch = titleRangeMatch && weekdaySet.has(weekday);
                specificity = titleRange ? 2 : 1;
                spanDays = titleRange ? Math.max(1, titleRange.spanDays) : 9998;
            }

            if (!isDateMatch) continue;

            const window = parseHoursWindow(row.hours);
            if (!window) continue;

            candidates.push({
                sectionTitle: section.title,
                label: row.label,
                hours: row.hours,
                window,
                specificity,
                spanDays,
                sectionIndex,
                rowIndex,
            });
        }
    });

    return candidates;
};

const resolveWinnerCandidate = (
    sections: FacilityHoursSection[],
    dayKey: string,
    weekday: number,
    fallbackYear: number,
    chicagoMinutes: number
): ResolvedRuleCandidate | null => {
    const candidates = collectRuleWindowCandidates(
        sections,
        dayKey,
        weekday,
        fallbackYear
    ).map((candidate) => ({
        rule: {
            sectionTitle: candidate.sectionTitle,
            label: candidate.label,
            hours: candidate.hours,
            isOpen: isOpenForWindow(candidate.window, chicagoMinutes),
        },
        window: candidate.window,
        specificity: candidate.specificity,
        spanDays: candidate.spanDays,
        sectionIndex: candidate.sectionIndex,
        rowIndex: candidate.rowIndex,
    }));

    if (candidates.length === 0) {
        return null;
    }

    candidates.sort(candidateComparator);
    return candidates[0];
};

const getPreviousDayWinnerCandidate = (
    sections: FacilityHoursSection[],
    dayKey: string,
    chicagoMinutes: number
): ResolvedRuleCandidate | null => {
    const previousDayKey = shiftDateKeyByDays(dayKey, -1);
    if (!previousDayKey) return null;

    const previousDate = fromDateKey(previousDayKey);
    if (!previousDate) return null;

    return resolveWinnerCandidate(
        sections,
        previousDayKey,
        previousDate.getUTCDay(),
        previousDate.getUTCFullYear(),
        chicagoMinutes
    );
};

const canUsePreviousDaySpillover = (
    todayCandidate: ResolvedRuleCandidate | null,
    previousCandidate: ResolvedRuleCandidate | null
): previousCandidate is ResolvedRuleCandidate => {
    if (!previousCandidate) return false;
    if (previousCandidate.window.isClosed || previousCandidate.window.endMinutes <= MINUTES_PER_DAY) {
        return false;
    }
    if (!todayCandidate) return true;

    return todayCandidate.specificity <= previousCandidate.specificity;
};

const windowToSameDaySegments = (window: TimeWindow): FacilityOpenWindow[] => {
    if (window.isClosed) return [];

    const startMinutes = Math.max(0, Math.min(MINUTES_PER_DAY, window.startMinutes));
    const endMinutes = Math.max(startMinutes, Math.min(MINUTES_PER_DAY, window.endMinutes));
    if (endMinutes <= startMinutes) return [];

    return [{startMinutes, endMinutes}];
};

const getPreviousDaySpilloverSegments = (
    previousCandidate: ResolvedRuleCandidate | null
): FacilityOpenWindow[] => {
    if (!previousCandidate) return [];
    if (previousCandidate.window.isClosed || previousCandidate.window.endMinutes <= MINUTES_PER_DAY) {
        return [];
    }

    const endMinutes = Math.max(
        0,
        Math.min(MINUTES_PER_DAY, previousCandidate.window.endMinutes - MINUTES_PER_DAY)
    );
    if (endMinutes <= 0) return [];

    return [{startMinutes: 0, endMinutes}];
};

const mergeOpenWindows = (windows: FacilityOpenWindow[]): FacilityOpenWindow[] => {
    if (windows.length <= 1) return windows;

    const sorted = windows
        .slice()
        .sort((left, right) => left.startMinutes - right.startMinutes || left.endMinutes - right.endMinutes);
    const merged: FacilityOpenWindow[] = [sorted[0]];

    for (let index = 1; index < sorted.length; index += 1) {
        const current = sorted[index];
        const last = merged[merged.length - 1];

        if (current.startMinutes <= last.endMinutes) {
            last.endMinutes = Math.max(last.endMinutes, current.endMinutes);
            continue;
        }

        merged.push({...current});
    }

    return merged;
};

export const getFacilityOpenStatus = (
    schedule: FacilityHoursFacilityPayload | null | undefined,
    now: Date = new Date()
): FacilityOpenStatus => {
    if (!schedule || !Array.isArray(schedule.sections) || schedule.sections.length === 0) {
        return {state: "unknown", matchedRule: null};
    }

    const chicagoNow = parseChicagoNowParts(now);
    if (!chicagoNow) {
        return {state: "unknown", matchedRule: null};
    }

    const todayKey = toDateKey(chicagoNow.year, chicagoNow.month, chicagoNow.day);
    const todayCandidate = resolveWinnerCandidate(
        schedule.sections,
        todayKey,
        chicagoNow.weekday,
        chicagoNow.year,
        chicagoNow.minutes
    );
    const previousDayCandidate = getPreviousDayWinnerCandidate(
        schedule.sections,
        todayKey,
        chicagoNow.minutes
    );

    if (todayCandidate?.rule.isOpen) {
        return {
            state: "open",
            matchedRule: todayCandidate.rule,
        };
    }

    if (
        canUsePreviousDaySpillover(todayCandidate, previousDayCandidate)
        && previousDayCandidate.rule.isOpen
    ) {
        return {
            state: "open",
            matchedRule: previousDayCandidate.rule,
        };
    }

    if (!todayCandidate) {
        return {state: "unknown", matchedRule: null};
    }

    return {
        state: todayCandidate.rule.isOpen ? "open" : "closed",
        matchedRule: todayCandidate.rule,
    };
};

export const getFacilityOpenWindowsForDate = (
    schedule: FacilityHoursFacilityPayload | null | undefined,
    dateKey: string | null | undefined
): FacilityOpenWindow[] => {
    if (!schedule || !Array.isArray(schedule.sections) || schedule.sections.length === 0) {
        return [];
    }
    if (!dateKey) return [];

    const parsed = fromDateKey(dateKey);
    if (!parsed) return [];

    const year = parsed.getUTCFullYear();
    const weekday = parsed.getUTCDay();
    const todayCandidate = resolveWinnerCandidate(
        schedule.sections,
        dateKey,
        weekday,
        year,
        0
    );
    const previousDayCandidate = getPreviousDayWinnerCandidate(
        schedule.sections,
        dateKey,
        0
    );

    const windows: FacilityOpenWindow[] = [];
    if (canUsePreviousDaySpillover(todayCandidate, previousDayCandidate)) {
        windows.push(...getPreviousDaySpilloverSegments(previousDayCandidate));
    }
    if (todayCandidate) {
        windows.push(...windowToSameDaySegments(todayCandidate.window));
    }

    return mergeOpenWindows(windows);
};
