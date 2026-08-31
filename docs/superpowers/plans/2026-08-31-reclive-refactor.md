# RecLive Behavior-Preserving Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Split the protected RecLive frontend and backend modules into focused feature and service modules without changing routes, APIs, stored artifacts, forecasting behavior, or user-visible behavior.

**Architecture:** Execute only after Phases 1-8 are green. Characterize every boundary first, move pure functions before stateful entry points, and keep compatibility wrappers at all four top-level backend scripts. `src/app/App.tsx` becomes orchestration only; feature modules own dashboard state, forecast calculation/rendering, heat-map geometry/modeling, and alerts. `server/reclive/` owns validated settings, connections, repositories, routers, ingestion, schedules, push, and forecasting; the model’s data, feature, training, and prediction algorithms move verbatim.

**Tech Stack:** React 19, TypeScript 5.9, Vitest, React Testing Library, Playwright; Python 3, FastAPI, PyMySQL, pytest, XGBoost.

**Spec:** `docs/superpowers/specs/2026-08-31-reclive-security-data-trust-design.md`

## Global Constraints

- Execute Phase 9 only after Phases 1-8 are green; preserve their behavior, contracts, tests, route paths, data schemas, and security controls.
- Do not alter facility IDs `1186`/`1656`, `/nick`, `/bakke`, install behavior, push behavior, forecast algorithm, model inputs, artifact paths, saved-artifact compatibility, or public API contracts.
- Do not run full XGBoost training in ordinary test or CI commands. Use deterministic fixtures, fixed clocks, fake repositories, and saved small fixtures.
- Keep `server/gym_fetch.py`, `server/forecast_api.py`, `server/forecast_job.py`, and `server/facility_hours_fetch.py` executable and import-compatible as thin delegation entry points.
- No React component may be declared inside another component. Public UI copy and visual layout remain unchanged except for metric label replacement required below.
- Report explicit forecast metrics: MAE in people, MAE in capacity percentage points, RMSE in people, prediction-interval coverage, simple-baseline MAE in people, and rolling holdout by facility. Remove `precisionPct` rather than relabeling its existing ratio-derived value.
- Retain frontend tests under `src/**/*.test.ts(x)`, backend tests under `tests/backend/`, fixtures under `tests/fixtures/`, and browser tests under `tests/e2e/`.
- Do not merge, deploy, change credentials, contact providers, or commit unrelated working-tree changes.

---

## File Structure

- Create: `src/features/dashboard/DashboardPage.tsx`, `src/features/dashboard/useDashboardState.ts`, `src/features/dashboard/dashboardSelectors.ts`, `src/features/dashboard/dashboardTypes.ts`, and `src/test/fixtures/dashboard.ts`.
- Create: `src/features/forecast/ForecastCard.tsx`, `src/features/forecast/ForecastChart.tsx`, `src/features/forecast/ForecastDayControls.tsx`, `src/features/forecast/ForecastWindowsList.tsx`, `src/features/forecast/forecastBands.ts`, `src/features/forecast/forecastHistogram.ts`, `src/features/forecast/forecastTime.ts`.
- Create: `src/features/heatmap/HeatmapCard.tsx`, `src/features/heatmap/HeatmapSvg.tsx`, `src/features/heatmap/floorMaps.ts`, `src/features/heatmap/heatmapGeometry.ts`, `src/features/heatmap/heatmapModel.ts`.
- Create: `src/features/alerts/AlertsDialog.tsx`, `src/features/alerts/AlertRuleForm.tsx`, `src/features/alerts/ActiveAlertRules.tsx`, `src/features/alerts/alertSubscriptionStorage.ts`, `src/features/alerts/alertTypes.ts`.
- Modify: `src/app/App.tsx` and existing `src/facilities/ForecastWindowsCard.tsx`, `src/facilities/FloorHeatMapCard.tsx`, `src/facilities/CrowdAlertSubscriptionCard.tsx` as re-export compatibility modules.
- Create: `server/reclive/settings.py`, `server/reclive/db.py`, `server/reclive/sections.py`, `server/reclive/api/app.py`, `server/reclive/api/dependencies.py`, `server/reclive/api/lifespan_compat.py`, `server/reclive/api/health.py`, `server/reclive/api/live_counts.py`, `server/reclive/api/forecasts.py`, `server/reclive/api/schedules.py`, and `server/reclive/api/push.py`.
- Create: `server/reclive/repositories/push_rules.py` and move existing Phase 2-8 repository/service code into focused modules without replacing public imports: `server/reclive/occupancy_repository.py`, `server/reclive/ingestion.py`, `server/reclive/actual_hours.py`, `server/reclive/facility_schedule.py`, and `server/reclive/push.py`.
- Create: `server/reclive/forecasting/config.py`, `server/reclive/forecasting/data.py`, `server/reclive/forecasting/features.py`, `server/reclive/forecasting/training.py`, `server/reclive/forecasting/prediction.py`, `server/reclive/forecasting/metrics.py`, `server/reclive/forecasting/reporting.py`, and `server/reclive/forecasting/job.py`.
- Modify: `server/forecast_api.py`, `server/gym_fetch.py`, `server/facility_hours_fetch.py`, `server/forecast_job.py`, and `server/forecast_shared.py` as thin compatibility delegators.
- Create: `tests/backend/test_compatibility_entrypoints.py`, `tests/backend/test_forecasting_metrics.py`, `tests/backend/test_router_contracts.py`, `src/features/dashboard/dashboardSelectors.test.ts`, `src/features/forecast/forecastHistogram.test.ts`, `src/features/heatmap/heatmapModel.test.ts`, `src/features/alerts/alertSubscriptionStorage.test.ts`, and `tests/fixtures/refactor/forecast-contract.json`.

### Task 1: Freeze behavior with frontend/backend characterization fixtures

**Files:**
- Create: `tests/fixtures/refactor/forecast-contract.json`
- Create: `tests/backend/test_compatibility_entrypoints.py`
- Create: `tests/backend/test_router_contracts.py`
- Modify: `tests/e2e/route-smoke.spec.ts`

**Interfaces:**
- Consumes: Phase 1-8 API fixtures and fixed values for Nick `1186` and Bakke `1656`.
- Produces: pre-refactor characterization assertions for returned payload shape, route status/body, importable script functions, retained Nick/Bakke dashboard controls, and a deterministic sanitized forecast fixture.

- [ ] **Step 1: Write characterization tests against the existing entry points**

```python
import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import forecast_api

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "refactor" / "forecast-contract.json"


@pytest.fixture
def api_client(monkeypatch) -> TestClient:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    monkeypatch.setattr(forecast_api, "load_forecast", lambda: payload)
    return TestClient(forecast_api.app)


def test_legacy_backend_entrypoints_export_their_existing_callables() -> None:
    assert callable(importlib.import_module("server.gym_fetch").main)
    assert callable(importlib.import_module("server.forecast_api").app)
    assert callable(importlib.import_module("server.forecast_job").build_forecast)
    assert callable(importlib.import_module("server.facility_hours_fetch").main)


def test_forecast_router_preserves_compact_contract(api_client) -> None:
    response = api_client.get("/api/forecast/facilities/1186?compact=1")

    assert response.status_code == 200
    assert set(response.json()) >= {
        "facilityId", "facilityName", "weeklyForecast",
        "forecastDayStartHour", "forecastDayEndHour",
        "occupancyThresholds", "sectionOccupancyThresholds",
        "locationOccupancyThresholds",
    }
```

In the existing `tests/e2e/route-smoke.spec.ts` route loop, retain its final-schema API mocks and add these assertions after navigation so the browser contract is protected before extraction:

```ts
await expect(page.getByText("Live Occupancy")).toBeVisible();
await expect(page.getByRole("button", {name: "Alerts"})).toBeVisible();
```

- [ ] **Step 2: Run characterization tests to lock the green baseline**

Run: `pytest tests/backend/test_compatibility_entrypoints.py tests/backend/test_router_contracts.py -q && npm run test:e2e -- tests/e2e/route-smoke.spec.ts`

Expected: PASS against the Phase 1-8 code before any module move. If an assertion fails, correct the characterization to the approved current contract or repair the earlier phase before refactoring; do not bless a changed baseline.

- [ ] **Step 3: Record deterministic fixture contents**

Create `tests/fixtures/refactor/forecast-contract.json` in the same first edit as the tests, before the Step 2 command. Its root has `forecastDayStartHour`, `forecastDayEndHour`, and `facilities`; include Nick `1186` and Bakke `1656`, and give each facility `facilityName`, threshold objects, and one `weeklyForecast` day with category keys, two timezone-aware `hourStart` values, expected counts, qualified actual fields, crowd bands, interval fields, and no secret/configuration values. The `api_client` fixture above is therefore runnable at the first baseline. Write tests that compare only sorted JSON values and view-model fields, never full generated timestamps or model binaries. In Task 7, rewire only this fixture to construct `create_app(Settings.for_test(...))` with the same fixture path; do not weaken its route/body assertions.

- [ ] **Step 4: Re-run the green baseline and commit it before any extraction**

Run: `pytest tests/backend/test_compatibility_entrypoints.py tests/backend/test_router_contracts.py -q && npm run test:e2e -- tests/e2e/route-smoke.spec.ts`

Expected: PASS with no temporary adapter or new production module. Retain these assertions through every later task.

```bash
git add tests/backend/test_compatibility_entrypoints.py tests/backend/test_router_contracts.py tests/fixtures/refactor/forecast-contract.json tests/e2e/route-smoke.spec.ts
git commit -m "test: characterize pre-refactor contracts"
```

### Task 2: Extract pure dashboard selection and lifecycle state

**Files:**
- Create: `src/features/dashboard/dashboardTypes.ts`
- Create: `src/features/dashboard/dashboardSelectors.ts`
- Create: `src/features/dashboard/useDashboardState.ts`
- Create: `src/features/dashboard/dashboardSelectors.test.ts`
- Create: `src/test/fixtures/dashboard.ts`
- Modify: `src/app/App.tsx`

**Interfaces:**
- Consumes: `FacilityPayload`, `ForecastDay[]`, `FacilitySchedule | null`, `OccupancySummary`, and current clock milliseconds.
- Produces: `DashboardViewModel`, `buildSectionForecastMap(days, sections)`, `deriveForecastBounds(days)`, and `useDashboardState(args)`.

- [ ] **Step 1: Write failing selector tests**

```ts
import {buildDashboardViewModel} from "./dashboardSelectors";
import {fixtureForecastDays, fixtureLive, fixtureSchedule} from "../../test/fixtures/dashboard";

it("keeps the active facility, selected forecast day, warnings, and alert sections stable", () => {
    const view = buildDashboardViewModel({
        facility: 1186, nowMs: Date.parse("2026-08-31T12:00:00Z"),
        live: fixtureLive, forecastDays: fixtureForecastDays, schedule: fixtureSchedule,
        selectedForecastDay: {key: null, offset: 0},
    });

    expect(view.facilitySummary.status).toBe("live");
    expect(view.alertSections[0]).toMatchObject({key: "overall", label: "Entire Facility"});
    expect(view.visibleForecastDays.map((day) => day.date)).toEqual(["2026-08-31"]);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test:run -- src/features/dashboard/dashboardSelectors.test.ts`

Expected: FAIL because `dashboardSelectors.ts` does not exist.

- [ ] **Step 3: Extract exact dashboard interfaces and pure selectors**

```ts
export interface ForecastDaySelection {
    key: string | null;
    offset: number;
}

export interface DashboardViewModel {
    facilitySummary: OccupancySummary;
    sectionSummaries: ReadonlyMap<string, OccupancySummary>;
    otherSummary: OccupancySummary | null;
    alertSections: readonly AlertSectionOption[];
    visibleForecastDays: readonly ForecastDay[];
    selectedForecastDay: ForecastDay | null;
    forecastHourBounds: ForecastHourBounds;
    warning: WarningResolution;
}

export function buildDashboardViewModel(input: DashboardSelectorInput): DashboardViewModel;
export function buildSectionForecastMap(days: readonly ForecastDay[], sections: readonly SectionConfig[]): Record<string, ForecastHour[]>;
export function deriveForecastBounds(days: readonly ForecastDay[]): ForecastHourBounds;
```

Move `normalizeSectionTitle`, Chicago time/date keys, forecast-bound derivation, `buildSectionForecastMap`, section/Other/alert summary calculation, and warning selection from `App.tsx` into selector modules unchanged. Put storage-backed selected-facility and debug-time initialization, refresh keys, clock tick, controlled modal state, visibility-aware refresh wiring, and route callback into `useDashboardState.ts`. Do not declare JSX components in that hook. Create `src/test/fixtures/dashboard.ts` with fully typed exports `fixtureLive: FacilityPayload`, `fixtureForecastDays: ForecastDay[]`, `fixtureSchedule: FacilitySchedule`, `fixtureThresholds: OccupancyThresholds`, `fixtureLocations: Location[]`, `liveSummary: OccupancySummary`, `closedSummary: OccupancySummary`, and `createDashboardStateForTest(overrides?: Partial<DashboardState>): DashboardState`; use fixed ISO timestamps and both facility IDs, and import these names in Tasks 2-6 rather than relying on undeclared globals.

- [ ] **Step 4: Run focused verification**

Run: `npm run test:run -- src/features/dashboard/dashboardSelectors.test.ts src/app/warningStatus.test.ts && npm run build`

Expected: PASS; selector output matches pre-extraction fixture fields.

- [ ] **Step 5: Commit**

```bash
git add src/features/dashboard/dashboardTypes.ts src/features/dashboard/dashboardSelectors.ts src/features/dashboard/useDashboardState.ts src/features/dashboard/dashboardSelectors.test.ts src/test/fixtures/dashboard.ts src/app/App.tsx
git commit -m "refactor: extract dashboard state and selectors"
```

### Task 3: Split forecast computation and the oversized forecast view

**Files:**
- Create: `src/features/forecast/forecastTime.ts`
- Create: `src/features/forecast/forecastBands.ts`
- Create: `src/features/forecast/forecastHistogram.ts`
- Create: `src/features/forecast/ForecastCard.tsx`, `src/features/forecast/ForecastChart.tsx`, `src/features/forecast/ForecastDayControls.tsx`, and `src/features/forecast/ForecastWindowsList.tsx`
- Modify: `src/facilities/ForecastWindowsCard.tsx`
- Create: `src/features/forecast/forecastHistogram.test.ts`

**Interfaces:**
- Consumes: existing `ForecastDay`, `FacilityOpenWindow[]`, `OccupancyThresholds`, and fixed `nowTs`.
- Produces: `buildForecastDisplaySlots`, `buildCrowdBandsFromDisplaySlots`, `buildHistogramModel`, and the unchanged default `ForecastWindowsCard` props.

- [ ] **Step 1: Write failing histogram characterization test**

```ts
import {fixtureThresholds} from "../../test/fixtures/dashboard";
import {buildHistogramModel, type ForecastDisplaySlot} from "./forecastHistogram";

const fixtureSlots: ForecastDisplaySlot[] = [
    {startMinute: 360, endMinute: 390, startTs: 0, endTs: 1, count: 40, percent: 20, source: "actual", level: "low"},
    {startMinute: 390, endMinute: 420, startTs: 1, endTs: 2, count: 44, percent: 22, source: "actual", level: "low"},
    {startMinute: 420, endMinute: 450, startTs: 2, endTs: 3, count: 100, percent: 50, source: "predicted", level: "medium"},
    {startMinute: 450, endMinute: 480, startTs: 3, endTs: 4, count: 104, percent: 52, source: "predicted", level: "medium"},
    {startMinute: 480, endMinute: 510, startTs: 4, endTs: 5, count: 120, percent: 60, source: "actual", level: "medium"},
    {startMinute: 510, endMinute: 540, startTs: 5, endTs: 6, count: 124, percent: 62, source: "predicted", level: "medium"},
    {startMinute: 540, endMinute: 570, startTs: 6, endTs: 7, count: 0, percent: null, source: "predicted", level: "unknown"},
    {startMinute: 570, endMinute: 600, startTs: 7, endTs: 8, count: 0, percent: null, source: "predicted", level: "unknown"},
];

it("preserves actual, predicted, mixed, and unknown bar counts", () => {
    const model = buildHistogramModel(fixtureSlots, 240, fixtureThresholds);

    expect(model).toMatchObject({
        actualBarCount: 1,
        predictedBarCount: 2,
        mixedBarCount: 1,
        unknownBarCount: 1,
        yMax: 250,
    });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test:run -- src/features/forecast/forecastHistogram.test.ts`

Expected: FAIL because `forecastHistogram.ts` does not exist.

- [ ] **Step 3: Move pure forecast groups unchanged**

```ts
export interface ForecastDisplaySlot {
    startMinute: number;
    endMinute: number;
    startTs: number;
    endTs: number;
    count: number;
    percent: number | null;
    source: "actual" | "predicted" | "mixed";
    level: CrowdBandLevel | "unknown";
}

export interface HistogramModel {
    bars: readonly HistogramBar[];
    yTicks: readonly HistogramTick[];
    yMax: number;
    actualBarCount: number;
    predictedBarCount: number;
    mixedBarCount: number;
    unknownBarCount: number;
}

export function buildForecastDisplaySlots(day: ForecastDay, nowTs: number, schedule: FacilityOpenWindow[] | null, thresholds: OccupancyThresholds | null): ForecastDisplaySlot[];
export function buildHistogramModel(slots: readonly ForecastDisplaySlot[], maxCapacity: number | null, thresholds: OccupancyThresholds | null): HistogramModel;
```

Move time parsing/range clipping to `forecastTime.ts`, crowd-band sort/merge/smoothing to `forecastBands.ts`, and histogram smoothing/bar/tick/model construction to `forecastHistogram.ts` without changing arithmetic or constants. Put card-level state and composition in `ForecastCard.tsx`, SVG histogram markup only in `ForecastChart.tsx`, swipe/previous-next controls only in `ForecastDayControls.tsx`, and avoid/best window cards only in `ForecastWindowsList.tsx`. No file declares another React component inside its function body. Replace the legacy file with `export {default} from "../features/forecast/ForecastCard";` so existing imports stay valid.

- [ ] **Step 4: Run focused verification**

Run: `npm run test:run -- src/features/forecast/forecastHistogram.test.ts && npm run build`

Expected: PASS; bar classes and display slots equal the fixture.

- [ ] **Step 5: Commit**

```bash
git add src/features/forecast src/facilities/ForecastWindowsCard.tsx
git commit -m "refactor: split forecast display model"
```

### Task 4: Split heat-map configuration, geometry, model, and SVG view

**Files:**
- Create: `src/features/heatmap/floorMaps.ts`
- Create: `src/features/heatmap/heatmapModel.ts`
- Create: `src/features/heatmap/HeatmapCard.tsx`, `src/features/heatmap/HeatmapSvg.tsx`, and `src/features/heatmap/heatmapGeometry.ts`
- Modify: `src/facilities/FloorHeatMapCard.tsx`
- Create: `src/features/heatmap/heatmapModel.test.ts`

**Interfaces:**
- Consumes: `FacilityId`, `Location[]`, summary/status data, and location occupancy thresholds.
- Produces: `FLOOR_MAPS`, `buildFloorRenderData`, `isInsidePolygon`, and unchanged default card props.

- [ ] **Step 1: Write failing map model test**

```ts
import {fixtureLocations, fixtureThresholds} from "../../test/fixtures/dashboard";
import {buildFloorRenderData} from "./heatmapModel";

it("retains closed zones, neutral unknown zones, and colored observed zones", () => {
    const result = buildFloorRenderData(1656, 1, fixtureLocations, fixtureThresholds, {});

    expect(result.closedZones.map((zone) => zone.label)).toEqual(["Ice Center"]);
    expect(result.zoneSummaries.find((zone) => zone.zone.label === "The Point")?.status).toBe("partial");
    expect(result.heatCells.length).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test:run -- src/features/heatmap/heatmapModel.test.ts`

Expected: FAIL because `heatmapModel.ts` does not exist.

- [ ] **Step 3: Extract immutable geometry and pure render data**

```ts
export interface FloorRenderData {
    floorMap: FloorMapConfig | null;
    gridCols: number;
    gridRows: number;
    heatCells: readonly HeatCell[];
    closedZones: readonly ZoneConfig[];
    zoneSummaries: readonly ZoneSummary[];
}

export function buildFloorRenderData(
    facilityId: FacilityId, floor: number, locations: readonly Location[],
    occupancyThresholds: OccupancyThresholds | null,
    locationOccupancyThresholds: Partial<Record<number, OccupancyThresholds>>,
): FloorRenderData;
```

Move `FLOOR_MAPS` and immutable map types to `floorMaps.ts`; move polygon containment/bounds to `heatmapGeometry.ts`; move zone summary/coverage, overlay fill, and grid model to `heatmapModel.ts`. `HeatmapCard.tsx` owns expanded/floor/popover state and controls. `HeatmapSvg.tsx` owns the image, patterns, cell/closure layers, hit polygons, keyboard interaction, and accessible labels. Replace the old facility file with `export {default} from "../features/heatmap/HeatmapCard";`. This leaves no relocated thousand-line component and no nested components.

- [ ] **Step 4: Run focused verification**

Run: `npm run test:run -- src/features/heatmap/heatmapModel.test.ts && npm run build`

Expected: PASS; fixture zone statuses and geometry are unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/features/heatmap src/facilities/FloorHeatMapCard.tsx
git commit -m "refactor: separate heatmap model and view"
```

### Task 5: Split alert storage, dialog, form, and active-rule management

**Files:**
- Create: `src/features/alerts/alertTypes.ts`
- Create: `src/features/alerts/alertSubscriptionStorage.ts`
- Create: `src/features/alerts/AlertsDialog.tsx`, `src/features/alerts/AlertRuleForm.tsx`, and `src/features/alerts/ActiveAlertRules.tsx`
- Modify: `src/facilities/CrowdAlertSubscriptionCard.tsx`
- Modify: `src/app/App.tsx`
- Create: `src/features/alerts/alertSubscriptionStorage.test.ts`

**Interfaces:**
- Consumes: existing browser push API functions and `AlertSectionOption[]`.
- Produces: `readStoredSubscriptions`, `writeStoredSubscriptions`, `resolveInitialSectionKey`, `resolveDefaultThresholdInput`, `AlertSectionOption`, and unchanged default card props.

- [ ] **Step 1: Write failing storage/selection tests**

```ts
import {closedSummary, liveSummary} from "../../test/fixtures/dashboard";
import {resolveDefaultThresholdInput, writeStoredSubscriptions} from "./alertSubscriptionStorage";

it("preserves stored rule selection and clamps default threshold", () => {
    writeStoredSubscriptions({1186: {sectionKey: "overall", threshold: 40}});

    expect(resolveDefaultThresholdInput(1186, {key: "overall", label: "Entire Facility", summary: liveSummary})).toBe("40");
    expect(resolveDefaultThresholdInput(1186, {key: "closed", label: "Closed", summary: closedSummary})).toBe("");
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test:run -- src/features/alerts/alertSubscriptionStorage.test.ts`

Expected: FAIL because the storage module does not exist.

- [ ] **Step 3: Extract explicit alert interfaces**

```ts
export interface AlertSectionOption {
    key: string;
    label: string;
    summary: OccupancySummary;
}

export interface StoredSubscription {
    sectionKey: string;
    threshold: number;
}

export type StoredSubscriptions = Partial<Record<FacilityId, StoredSubscription>>;

export function resolveDefaultThresholdInput(
    facility: FacilityId, selectedSection: AlertSectionOption | null
): string;
export function writeStoredSubscriptions(value: StoredSubscriptions): void;
```

Move localStorage parsing/writing, stored-rule selection, threshold clamping, and standalone labels to `alertSubscriptionStorage.ts`. `AlertsDialog.tsx` owns modal state/availability/subscription loading, `AlertRuleForm.tsx` owns selected section and submit validation, and `ActiveAlertRules.tsx` owns visible rule list, individual cancel, and cancel-all behavior. Preserve browser calls `ensurePushSubscription`, `getExistingPushSubscription`, `getPushAvailability`, `subscribePushRule`, `listPushRules`, `cancelPushRule`, and `cancelAllPushRules` plus current copy and callback contract. Replace the old facility component with exports of `AlertSectionOption` and default `AlertsDialog`.

- [ ] **Step 4: Run focused verification**

Run: `npm run test:run -- src/features/alerts/alertSubscriptionStorage.test.ts && npm run build`

Expected: PASS; old localStorage data and alert threshold selection remain compatible.

- [ ] **Step 5: Commit**

```bash
git add src/features/alerts src/facilities/CrowdAlertSubscriptionCard.tsx src/app/App.tsx
git commit -m "refactor: isolate alert subscription state"
```

### Task 6: Reduce App.tsx to dashboard orchestration with no nested components

**Files:**
- Create: `src/features/dashboard/DashboardPage.tsx`
- Modify: `src/app/App.tsx`
- Create: `src/features/dashboard/DashboardPage.test.tsx`

**Interfaces:**
- Consumes: `AppProps`, `useDashboardState`, and `DashboardViewModel`.
- Produces: unchanged `App` default export and top-level `DashboardPage(props: DashboardPageProps): JSX.Element`.

- [ ] **Step 1: Write failing render characterization test**

```tsx
const fixtureDashboardProps: DashboardPageProps = {
    state: createDashboardStateForTest({facility: 1186, isCrowdAlertOpen: false}),
    view: buildDashboardViewModel({
        facility: 1186,
        nowMs: Date.parse("2026-08-31T12:00:00Z"),
        live: fixtureLive,
        forecastDays: fixtureForecastDays,
        schedule: fixtureSchedule,
        selectedForecastDay: {key: null, offset: 0},
    }),
    themeMode: "light",
    onThemeModeChange: vi.fn(),
};

it("renders Nick route content, alerts control, and forecast card through DashboardPage", () => {
    render(
        <DashboardPage
            state={fixtureDashboardProps.state}
            view={fixtureDashboardProps.view}
            themeMode={fixtureDashboardProps.themeMode}
            onThemeModeChange={fixtureDashboardProps.onThemeModeChange}
        />,
    );

    expect(screen.getByText("Live Occupancy")).toBeVisible();
    expect(screen.getByRole("button", {name: "Alerts"})).toBeVisible();
    expect(screen.getByText("Forecast")).toBeVisible();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm run test:run -- src/features/dashboard/DashboardPage.test.tsx`

Expected: FAIL because `DashboardPage.tsx` does not exist.

- [ ] **Step 3: Move page JSX and retain App route contract**

```ts
export interface DashboardPageProps {
    state: DashboardState;
    view: DashboardViewModel;
    themeMode: PaletteMode;
    onThemeModeChange(mode: PaletteMode): void;
}

export function DashboardPage(props: DashboardPageProps): JSX.Element;
```

Move dashboard layout JSX and imported top-level components to `DashboardPage.tsx`. Every child is imported at module scope; define no component inside `DashboardPage` or `App`. `App` retains only prop acceptance, `useDashboardState`, `useFacilitySeo`, and `return <DashboardPage state={state} view={view} themeMode={themeMode} onThemeModeChange={onThemeModeChange} />`. Preserve lazy heat-map loading and responsive alert modal choice as state/view fields so timing and accessibility remain unchanged.

- [ ] **Step 4: Run frontend regression checks**

Run: `npm run test:run -- src/features/dashboard/DashboardPage.test.tsx src/features/dashboard/dashboardSelectors.test.ts && npm run lint && npm run build`

Expected: PASS; `App.tsx` contains no JSX child component declarations.

- [ ] **Step 5: Commit**

```bash
git add src/features/dashboard/DashboardPage.tsx src/features/dashboard/DashboardPage.test.tsx src/app/App.tsx
git commit -m "refactor: make app dashboard orchestration only"
```

### Task 7: Establish backend settings, DB, repositories, routers, and services behind stable APIs

**Files:**
- Create: `server/reclive/settings.py`, `server/reclive/db.py`, `server/reclive/sections.py`, and `server/reclive/repositories/push_rules.py`
- Modify: existing Phase 2-8 `server/reclive/occupancy_repository.py`, `server/reclive/ingestion.py`, `server/reclive/actual_hours.py`, `server/reclive/facility_schedule.py`, and `server/reclive/push.py` through moves and re-exports only
- Create: `server/reclive/api/app.py`, `server/reclive/api/dependencies.py`, `server/reclive/api/lifespan_compat.py`, `server/reclive/api/health.py`, `server/reclive/api/live_counts.py`, `server/reclive/api/forecasts.py`, `server/reclive/api/schedules.py`, and `server/reclive/api/push.py`
- Modify: `server/forecast_api.py`
- Modify: `tests/backend/test_router_contracts.py`

**Interfaces:**
- Consumes: validated `Settings` and repositories through FastAPI dependencies.
- Produces: `create_app(settings: Settings | None = None) -> FastAPI` and route-equivalent router functions.

- [ ] **Step 1: Write failing app/dependency tests**

```python
from server.reclive.api.app import create_app
from server.reclive.settings import Settings


def test_create_app_preserves_all_public_routes() -> None:
    app = create_app(Settings.for_test())
    routes = {route.path for route in app.routes}

    assert {"/health", "/api/live-counts", "/api/facility-hours",
            "/api/forecast/facilities/{facility_id}",
            "/api/forecast/facilities/{facility_id}/actual-hours",
            "/api/push/subscribe"}.issubset(routes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_router_contracts.py -q`

Expected: FAIL because `server.reclive.api.app` does not exist.

- [ ] **Step 3: Extract validated boundaries without changing route semantics**

```python
@dataclass(frozen=True)
class Settings:
    forecast_json_path: str
    facility_hours_json_path: str
    actual_hour_min_coverage: float
    cors_origins: tuple[str, ...]
    database: DatabaseSettings
    push: PushSettings

    @classmethod
    def from_environment(cls) -> "Settings":
        return build_settings_from_environment(os.environ)
    @classmethod
    def for_test(cls, *, forecast_json_path: str | None = None) -> "Settings":
        return build_test_settings(forecast_json_path=forecast_json_path)


# server/reclive/api/dependencies.py
from fastapi import Request

from server.reclive.db import open_db_connection
from server.reclive.occupancy_repository import SnapshotRepository
from server.reclive.repositories.push_rules import PushRuleRepository


def get_occupancy_repository(request: Request) -> SnapshotRepository:
    return SnapshotRepository(open_db_connection(request.app.state.settings.database))


def get_push_rule_repository(request: Request) -> PushRuleRepository:
    return PushRuleRepository(open_db_connection(request.app.state.settings.database))


# server/reclive/api/app.py
from fastapi import FastAPI

from server.reclive.api.forecasts import router as forecasts_router
from server.reclive.api.health import router as health_router
from server.reclive.api.lifespan_compat import build_lifespan
from server.reclive.api.live_counts import router as live_counts_router
from server.reclive.api.push import router as push_router
from server.reclive.api.schedules import router as schedules_router
from server.reclive.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    effective_settings = settings or Settings.from_environment()
    app = FastAPI(title="RecLive Forecast API", version="1.1.0", lifespan=build_lifespan(effective_settings))
    app.state.settings = effective_settings
    app.include_router(health_router)
    app.include_router(live_counts_router)
    app.include_router(forecasts_router)
    app.include_router(schedules_router)
    app.include_router(push_router)
    return app
```

Create `server/reclive/api/lifespan_compat.py` in this task and move the existing Phase 5 evaluator startup/cancellation context into `build_lifespan(settings: Settings)`; it returns the FastAPI lifespan callable and preserves the tested enablement, one-task, cancellation, and cleanup behavior. Move environment parsing and production validation from `forecast_api.py` into `settings.py`; move `open_db_connection` into `db.py`; move section config load/normalization into `sections.py`. Move current snapshot/history reads from the existing `occupancy_repository.py` only into subfunctions while preserving the module’s current public imports through re-exports. Move push database queries into `repositories/push_rules.py` and re-export their service-facing names from `reclive.push`. Move route decorators into `health.py`, `live_counts.py`, `forecasts.py`, `schedules.py`, and `push.py`; each module imports `APIRouter` explicitly and exports `router = APIRouter()`. Phase 9's `health.router` contains the already-tested transitional health behavior from Phases 3 and 7, including the top-level sanitized schedule evidence, so no route disappears during extraction. Phase 10 will replace that router with its final repository-backed factory. Retain exact HTTP methods, paths, query names, request/response models, status codes, CORS behavior, and evaluator startup. Replace `server/forecast_api.py` with `from server.reclive.api.app import app` plus a `main()` that starts unchanged configured host/port. Update the Task 1 `api_client` fixture to use `create_app(Settings.for_test(forecast_json_path=str(FIXTURE_PATH)))`; this is a seam-only change and its characterization assertions remain identical.

- [ ] **Step 4: Run backend regression checks**

Run: `pytest tests/backend/test_router_contracts.py tests/backend/test_live_counts_api.py tests/backend/test_actual_hours.py -q && ruff check server/reclive server/forecast_api.py`

Expected: PASS; every public path and tested error status remains identical.

- [ ] **Step 5: Commit**

```bash
git add server/reclive/settings.py server/reclive/db.py server/reclive/sections.py server/reclive/repositories server/reclive/api server/forecast_api.py tests/backend/test_router_contracts.py
git commit -m "refactor: separate api services and repositories"
```

### Task 8: Extract ingestion, schedule, and push services; preserve scripts

**Files:**
- Modify: existing `server/reclive/ingestion.py`, `server/reclive/facility_schedule.py`, and `server/reclive/push.py`; preserve each module’s Phase 2-8 public imports with direct re-exports after moves
- Modify: `server/gym_fetch.py`, `server/facility_hours_fetch.py`, `server/forecast_shared.py`
- Modify: `tests/backend/test_compatibility_entrypoints.py`

**Interfaces:**
- Consumes: `Settings`, repositories, and existing data contracts.
- Produces: `run_configured_ingestion(settings: Settings) -> IngestionRunResult`, `run_facility_hours_fetch(settings: Settings) -> int`, `evaluate_rules_once(services: PushServices) -> PushEvaluationResult`, and `normalize_section_key(value: str) -> str`; the Phase 2 `run_ingestion(fetch_payload, connect, capacities, now)` interface remains import-compatible.

- [ ] **Step 1: Write failing script-delegation tests**

```python
from unittest.mock import patch

import server.gym_fetch as gym_fetch
from server.reclive.ingestion import IngestionRunResult


def test_gym_fetch_main_delegates_to_reclive_ingestion() -> None:
    result = IngestionRunResult("succeeded", 0, 0, 0, 0, None)
    with patch("server.gym_fetch.run_configured_ingestion", return_value=result) as run:
        assert gym_fetch.main() == 0
    run.assert_called_once()


def test_forecast_shared_reexports_section_normalizer() -> None:
    from server.forecast_shared import normalize_section_key

    assert normalize_section_key(" Fitness_Floors ") == "fitness floors"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_compatibility_entrypoints.py -q`

Expected: FAIL because the compatibility wrapper has not imported `run_ingestion`.

- [ ] **Step 3: Move service logic verbatim and thin wrappers**

```python
def run_configured_ingestion(settings: Settings) -> IngestionRunResult:
    return IngestionService(settings, SnapshotRepository(open_db_connection(settings.database))).run()
def run_facility_hours_fetch(settings: Settings) -> int:
    return FacilityScheduleService(settings).run()
def evaluate_rules_once(services: PushServices) -> PushEvaluationResult:
    return services.evaluator.evaluate_once()
```

Move Phase 2 ingestion subfunctions within existing `reclive/ingestion.py` without changing their public names or the four-argument `run_ingestion` signature; add `run_configured_ingestion` as the settings-bound command seam used by `gym_fetch.main`. Move Phase 7 HTML/WordPress parsing, partial preservation, atomic publish, and command main behavior into the existing `reclive/facility_schedule.py`, preserving its existing imports. Move Phase 5 subscription validation, claim/evaluation, locks, delivery, expiration, and rate-limit logic within `reclive/push.py`, preserving its existing imports. Leave each top-level script as imports, preserved exported helpers required by tests/cron, and `if __name__ == "__main__": raise SystemExit(main())`. Make `forecast_shared.py` re-export `normalize_section_key` from `reclive.sections`.

- [ ] **Step 4: Run compatibility verification**

Run: `pytest tests/backend/test_compatibility_entrypoints.py tests/backend/test_ingestion.py tests/backend/test_facility_hours.py tests/backend/test_push_lifecycle.py -q && ruff check server/reclive server/gym_fetch.py server/facility_hours_fetch.py server/forecast_shared.py`

Expected: PASS; direct script imports and CLI entry points remain callable.

- [ ] **Step 5: Commit**

```bash
git add server/reclive/ingestion.py server/reclive/facility_schedule.py server/reclive/push.py server/gym_fetch.py server/facility_hours_fetch.py server/forecast_shared.py tests/backend/test_compatibility_entrypoints.py
git commit -m "refactor: extract ingestion schedules and push services"
```

### Task 9: Move forecasting unchanged and publish explicit metrics

**Files:**
- Create: `server/reclive/forecasting/config.py`, `data.py`, `features.py`, `training.py`, `prediction.py`, `metrics.py`, `reporting.py`, and `job.py` under `server/reclive/forecasting/`
- Modify: `server/forecast_job.py`
- Create: `tests/backend/test_forecasting_metrics.py`
- Modify: `tests/backend/test_compatibility_entrypoints.py`

**Interfaces:**
- Consumes: existing history rows, schedule/weather inputs, saved XGBoost artifacts, existing environment variables, and fixed small fixture arrays.
- Produces: `build_forecast() -> dict[str, object]`, `write_forecast(payload: dict[str, object]) -> None`, `main() -> int`, and `ForecastMetrics`.

- [ ] **Step 1: Write failing metric and compatibility tests**

```python
from server.reclive.forecasting.metrics import compute_forecast_metrics


def test_explicit_metrics_have_correct_units_and_no_precision_pct() -> None:
    metrics = compute_forecast_metrics(
        actual_people=[100.0, 50.0], predicted_people=[90.0, 70.0],
        capacity_people=[200.0, 100.0], lower_people=[80.0, 40.0],
        upper_people=[110.0, 80.0], baseline_people=[110.0, 40.0],
    )

    assert metrics.mae_people == 15.0
    assert metrics.mae_capacity_percentage_points == 12.5
    assert metrics.rmse_people == pytest.approx((250.0) ** 0.5)
    assert metrics.prediction_interval_coverage == 1.0
    assert metrics.simple_baseline_mae_people == 10.0
    assert "precisionPct" not in metrics.to_payload()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/backend/test_forecasting_metrics.py -q`

Expected: FAIL because `server.reclive.forecasting.metrics` does not exist.

- [ ] **Step 3: Move algorithm groups verbatim, then add reporting-only metrics**

```python
@dataclass(frozen=True)
class ForecastMetrics:
    mae_people: float | None
    mae_capacity_percentage_points: float | None
    rmse_people: float | None
    prediction_interval_coverage: float | None
    simple_baseline_mae_people: float | None
    rolling_holdout_by_facility: Mapping[int, "FacilityHoldoutMetrics"]

    def to_payload(self) -> dict[str, object]:
        return {
            "maePeople": self.mae_people,
            "maeCapacityPercentagePoints": self.mae_capacity_percentage_points,
            "rmsePeople": self.rmse_people,
            "predictionIntervalCoverage": self.prediction_interval_coverage,
            "simpleBaselineMaePeople": self.simple_baseline_mae_people,
            "rollingHoldoutByFacility": {str(key): value.to_payload() for key, value in self.rolling_holdout_by_facility.items()},
        }


def compute_forecast_metrics(
    actual_people: Sequence[float], predicted_people: Sequence[float],
    capacity_people: Sequence[float], lower_people: Sequence[float],
    upper_people: Sequence[float], baseline_people: Sequence[float],
) -> ForecastMetrics:
    valid = [(actual, predicted, capacity, lower, upper, baseline) for actual, predicted, capacity, lower, upper, baseline in zip(actual_people, predicted_people, capacity_people, lower_people, upper_people, baseline_people) if all(math.isfinite(value) for value in (actual, predicted, capacity, lower, upper, baseline))]
    if not valid:
        return ForecastMetrics(None, None, None, None, None, {})
    errors = [abs(predicted - actual) for actual, predicted, _, _, _, _ in valid]
    capacity_errors = [abs(predicted - actual) / capacity * 100 for actual, predicted, capacity, _, _, _ in valid if capacity > 0]
    return ForecastMetrics(sum(errors) / len(errors), sum(capacity_errors) / len(capacity_errors) if capacity_errors else None, math.sqrt(sum(error * error for error in errors) / len(errors)), sum(lower <= actual <= upper for actual, _, _, lower, upper, _ in valid) / len(valid), sum(abs(baseline - actual) for actual, _, _, _, _, baseline in valid) / len(valid), {})

def compute_rolling_holdout_by_facility(
    rows: Sequence[ForecastEvaluationRow], windows: Sequence[RollingHoldoutWindow],
) -> Mapping[int, FacilityHoldoutMetrics]:
    return {
        facility_id: FacilityHoldoutMetrics.from_windows(facility_id, rows, windows)
        for facility_id in (1186, 1656)
    }
```

Move environment constants/path resolution to `config.py`; DB/history/artifact loading to `data.py`; calendar/weather/quality/features unchanged to `features.py`; XGBoost fitting, quantiles, tuning, champion gate, and artifact persistence unchanged to `training.py`; saved model prediction, blending, intervals, and forecast target assembly unchanged to `prediction.py`; output payload assembly/file write/main to `job.py`. Use mechanical moves with unchanged function bodies before adding `metrics.py` and `reporting.py`.

`compute_forecast_metrics` is reporting only: MAE is mean absolute people error; capacity percentage-point MAE is mean of `abs(predicted_people - actual_people) / capacity_people * 100` for positive finite capacity; RMSE is people units; interval coverage is fraction of actual values between inclusive lower/upper bounds; baseline MAE uses the simple baseline vector. `compute_rolling_holdout_by_facility` separately accepts rows that include facility ID and explicit chronological windows, computes the same metrics per 1186/1656 window, and returns `FacilityHoldoutMetrics` keyed by facility ID. Remove `precisionPct` from payload and `forecast_job.main` logs; retain existing `valMae` and `valRmse` keys only as temporary compatibility aliases during Phase 9, mapping them to `maePeople` and `rmsePeople`, and emit the new explicit names in `modelInfo.metrics`.

- [ ] **Step 4: Run no-training verification**

Run: `pytest tests/backend/test_forecasting_metrics.py tests/backend/test_compatibility_entrypoints.py -q && ruff check server/reclive/forecasting server/forecast_job.py`

Expected: PASS without an XGBoost training run; `precisionPct` is absent from serialized fixture output.

- [ ] **Step 5: Commit**

```bash
git add server/reclive/forecasting server/forecast_job.py tests/backend/test_forecasting_metrics.py tests/backend/test_compatibility_entrypoints.py
git commit -m "refactor: modularize forecasting and name metrics"
```

### Task 10: Verify full behavior and remove only refactor scaffolding

**Files:**
- Modify: refactor files from Tasks 2-9 only when verification exposes import/path regressions.
- Test: existing `tests/e2e/route-smoke.spec.ts` and all protected Phase 1-8 tests.

**Interfaces:**
- Consumes: all compatibility wrappers, routers, frontend features, and serialized forecast fixtures.
- Produces: proof that refactoring changed organization and metric naming only.

- [ ] **Step 1: Re-run the pre-refactor browser characterization after all extraction tasks**

Run: `npm run test:e2e -- tests/e2e/route-smoke.spec.ts`

Expected: PASS. The Nick/Bakke control assertions were added and committed in Task 1 before extraction; any failure here is an extraction regression, not an expected RED state.

- [ ] **Step 2: Correct imports and exports only if verification exposes a regression**

Keep every legacy frontend wrapper as a re-export and every backend top-level script as a delegator. Correct only module paths, exported type names, dependency injection bindings, and test fixture imports. Do not change calculations, route names, database queries, component copy, model hyperparameters, feature order, artifact format, or retry/security policy in this step.

- [ ] **Step 3: Run the complete required verification**

Run: `npm run lint && npm run build && npm run test:run && npm run test:e2e -- tests/e2e/route-smoke.spec.ts && pytest tests/backend -q && ruff check server && git diff --check`

Expected: PASS. If a credential-backed external check is unavailable, report it as unexecuted; do not infer a pass.

- [ ] **Step 4: Commit only if Step 2 required corrections**

```bash
git add src server tests
git commit -m "test: verify behavior-preserving refactor"
```

## Plan Self-Review

**Spec coverage:** Tasks 1-6 protect and split dashboard, forecast, heatmap, and alert behavior with top-level components only. Tasks 7-8 create the prescribed backend settings, DB, repositories, API routers, ingestion, schedules, and push boundaries while retaining compatibility scripts. Task 9 moves forecasting without rewriting the algorithm and replaces misleading `precisionPct` with explicit unit-named metrics, baseline comparison, interval coverage, and per-facility rolling holdout reports. Task 10 verifies routes, scripts, API contracts, browser behavior, lint, type check, tests, and diff hygiene.

**Placeholder scan:** Every task names files, interfaces, concrete tests, RED result, extraction boundary, GREEN command, and commit. No deferred implementation marker appears.

**Type consistency:** `DashboardPage` consumes `DashboardState`/`DashboardViewModel`; legacy feature modules re-export unchanged default components; `create_app` remains the FastAPI owner; top-level scripts delegate to named `reclive` functions; `ForecastMetrics.to_payload` is the sole metric serialization contract.
