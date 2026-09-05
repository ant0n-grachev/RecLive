import {closedSummary, liveSummary, partialSummary, fixtureLocations, fixtureNowTs} from "../../test/fixtures/dashboard";
import {computeOccupancySummary} from "../../shared/occupancy/computeOccupancySummary";
import {readStoredSubscriptions, resolveDefaultThresholdInput, resolveInitialSectionKey, writeStoredSubscriptions} from "./alertSubscriptionStorage";

const storageKey = "reclive:crowd-alert-subscriptions";
const overall = {key: "overall", label: "Entire Facility", summary: liveSummary};
const partial = {key: "partial", label: "Partial", summary: partialSummary};
const closed = {key: "closed", label: "Closed", summary: closedSummary};
const unknown = {key: "unknown", label: "Unknown", summary: computeOccupancySummary([], {nowMs: fixtureNowTs})};

beforeEach(() => localStorage.clear());
afterEach(() => vi.restoreAllMocks());

it("roundtrips only supported facility keys and stored selection fields", () => {
    const input = {1186: {sectionKey: "overall", threshold: 40, secret: "discard"}, 1656: {sectionKey: "partial", threshold: 12}, 9999: {sectionKey: "other", threshold: 10}};
    writeStoredSubscriptions(input);
    const expected = {1186: {sectionKey: "overall", threshold: 40}, 1656: {sectionKey: "partial", threshold: 12}};
    expect(JSON.parse(localStorage.getItem(storageKey)!)).toEqual(expected);
    expect(readStoredSubscriptions()).toEqual(expected);
});

it.each(["{", "null", "[]", "42", '{"1186":{"sectionKey":"","threshold":40}}', '{"1186":{"sectionKey":"overall","threshold":"40"}}'])("recovers malformed storage: %s", (text) => {
    localStorage.setItem(storageKey, text);
    expect(readStoredSubscriptions()).toEqual({});
});

it("sanitizes legacy data on read", () => {
    localStorage.setItem(storageKey, JSON.stringify({1186: {sectionKey: "overall", threshold: 40, endpoint: "discard"}, 9999: {sectionKey: "overall", threshold: 30}}));
    expect(readStoredSubscriptions()).toEqual({1186: {sectionKey: "overall", threshold: 40}});
});

it("recovers unavailable storage", () => {
    vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => { throw new Error("unavailable"); });
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => { throw new Error("unavailable"); });
    expect(readStoredSubscriptions()).toEqual({});
    expect(() => writeStoredSubscriptions({})).not.toThrow();
});

it("preserves matching live/partial selection and falls back past closed/unknown", () => {
    writeStoredSubscriptions({1186: {sectionKey: "partial", threshold: 12}});
    expect(resolveInitialSectionKey(1186, [closed, unknown, overall, partial])).toBe("partial");
    expect(resolveInitialSectionKey(1656, [closed, unknown, overall, partial])).toBe("overall");
    expect(resolveInitialSectionKey(1186, [closed, unknown])).toBe("");
    expect(resolveInitialSectionKey(1186, [overall])).toBe("overall");
});

it("clamps and rounds matching defaults while using the fallback for other sections", () => {
    writeStoredSubscriptions({1186: {sectionKey: "overall", threshold: 40}, 1656: {sectionKey: "overall", threshold: 8.6}});
    expect(resolveDefaultThresholdInput(1186, overall)).toBe("19");
    expect(resolveDefaultThresholdInput(1656, overall)).toBe("9");
    expect(resolveDefaultThresholdInput(1656, partial)).toBe("19");
    writeStoredSubscriptions({1186: {sectionKey: "overall", threshold: -5}});
    expect(resolveDefaultThresholdInput(1186, overall)).toBe("1");
});

it("caps the range at 100 and leaves closed, unknown, absent and zero occupancy blank", () => {
    const crowded = { ...overall, summary: computeOccupancySummary(fixtureLocations.map((row) => ({...row, currentCapacity: 150})), {nowMs: fixtureNowTs})};
    const empty = { ...overall, summary: computeOccupancySummary(fixtureLocations.map((row) => ({...row, currentCapacity: 0})), {nowMs: fixtureNowTs})};
    writeStoredSubscriptions({1186: {sectionKey: "overall", threshold: 180}});
    expect(resolveDefaultThresholdInput(1186, crowded)).toBe("100");
    expect(closedSummary.count).toBeNull();
    expect(closedSummary.percent).toBeNull();
    for (const section of [closed, unknown, empty, null]) expect(resolveDefaultThresholdInput(1186, section)).toBe("");
});
