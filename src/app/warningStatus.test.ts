import test from "node:test";
import assert from "node:assert/strict";
import {resolveDashboardWarning} from "./warningStatus.ts";

test("suppresses the banner for expected scheduled closures", () => {
    const result = resolveDashboardWarning({
        hasAnyError: false,
        isOffline: false,
        liveOutageState: "none",
        liveDataSource: "facility_api",
        forecastError: null,
        isScheduledClosedNow: true,
        isScheduledOpenButDataNotLive: false,
        isDataLikelyStale: false,
    });

    assert.deepEqual(result, {
        kind: "scheduled_closed",
        text: null,
        hidePredictions: true,
    });
});

test("keeps predictions visible when the schedule says open but live counts have not resumed", () => {
    const result = resolveDashboardWarning({
        hasAnyError: false,
        isOffline: false,
        liveOutageState: "none",
        liveDataSource: "facility_api",
        forecastError: null,
        isScheduledClosedNow: false,
        isScheduledOpenButDataNotLive: true,
        isDataLikelyStale: false,
    });

    assert.equal(result.kind, "scheduled_open_not_live");
    assert.equal(result.hidePredictions, false);
    assert.match(result.text ?? "", /open according to the official schedule/i);
});
