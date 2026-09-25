# Forecasting

RecLive uses NumPy and XGBoost to turn historical location counts, calendar patterns, official opening hours, and Open-Meteo weather into seven-day crowd forecasts. The job loads history, prepares features, trains or reuses models, applies calibration and prediction intervals, and publishes a JSON artifact for the API. The browser renders the forecast as charts and suggested visit windows.

The implementation lives in `server/reclive/forecasting/`: `data.py` loads inputs, `features.py` prepares features, `training.py` fits and selects models, `prediction.py` produces forecasts, and `reporting.py` and `metrics.py` qualify diagnostics. `job.py` coordinates the pipeline; `server/forecast_job.py` remains the command entry point.

## Forecast reporting diagnostics

The generated forecast's `modelInfo.metrics` and `modelInfo.metricContext`
describe qualified reporting diagnostics:

- `metrics.maePeople` is mean absolute error in people for location rows where
  both the cleaned observed people target and prediction are finite.
- `metrics.rmsePeople` is root mean square error in people over the same valid
  actual-plus-prediction rows.
- `metrics.maeCapacityPercentagePoints` divides absolute people error by that
  row's finite, positive per-location normalization capacity and multiplies by
  100. Its valid population can be smaller than the people-error population.
- `metrics.predictionIntervalCoverage` is a fraction from 0 to 1 for valid
  observed targets within a finite, ordered lower/upper interval; it is not a
  0-to-100 percentage field.
- `metrics.simpleBaselineMaePeople` is mean absolute error in people against a
  per-location raw last-observation persistence baseline frozen at each UTC
  window start. Observation and fetched/availability times must both strictly
  precede the window start; late, absent, invalid, or ambiguous observations
  provide no baseline for that row.
- `metricContext.observationCounts` and each window's `observationCounts` give
  separate samples for every metric; the valid populations are not assumed to
  match.

The overall and per-facility method is `fixed_model_terminal_holdout`, and
`independentBacktest` is `false`. One fitted model is evaluated across
non-overlapping UTC windows anchored at the actual terminal split, normally 24
hours with a possibly shorter final window. This is not rolling-origin
retraining. Retrospective preprocessing, full-history priors, tuning/selection,
and later calibration/champion selection prevent an independent-backtest
claim.

Rows are location observations from freshly trained-and-saved selected
facility-wide `__all__` models for facilities `1186` and `1656`; they are not
facility totals or final served/blended forecasts. The target is the cleaned,
schedule-adjusted bucket mean retained before ratio clipping. Model ratios,
predictions, and intervals are converted to people with each row's normalization
capacity: the maximum capacity encountered in loaded model history, not an
as-of historical capacity.

The `metricContext.timestampAlignment` reporting guard requires every
contributing location's canonical DB observation instant to match its preserved
model instant. Missing, unparseable, mismatched, or mixed alignment suppresses
that selected model's rows and windows instead of publishing uncertain
coverage. This reporting guard does not repair or reinterpret the preserved DB
timezone/source-time integration.

When a valid population is empty, scalar metrics are JSON `null`, its count is
zero, and `rollingHoldoutByFacility` is empty when no facility has qualified
evidence. A missing baseline can leave only the baseline metric/count
unavailable, and one qualifying facility may still contribute when the other
is suppressed. `metricContext.compatibilityAliases` maps public `valMae` and
`valRmse` to the people metrics. Legacy `byFacility`/`byModel` `valMae`,
`valRmse`, `holdoutMae`, and `holdoutRmse` remain weighted occupancy ratios;
guardrail, drift, and blend telemetry retain their algorithm-specific units.

## Display and serving correctness

Each chart bar covers one hour. Its height, count, and crowd color describe
that same interval; the display does not smooth counts or recolor short periods.
Completed hours use qualified actual attendance across every contained forecast
timestamp. Current and future hours remain forecasts. The interface shows only
the hour and people count; missing actual coverage falls back to a forecast.

Crowd thresholds are learned relative to historical attendance, rather than
fixed percentages meaning a building is nearly full. Live and forecast use the
same facility thresholds, but can disagree because their counts differ.

The forecast reads current location snapshots separately from training history.
Successful collection time controls the live correction's freshness; an unchanged
count does not make a successful fetch stale. Source measurement time remains
separate and can be unknown. Observed lag values take precedence over recursive
predictions, and only future targets seed that recursion. Live correction never
changes targets before the run's observation cutoff. Each future timestamp is
fully corrected, including schedule boundary zeros, before the next timestamp
builds its lag features. Locations remain batched within each timestamp.

Saved XGBoost models are written through temporary filenames retaining the JSON
suffix. Readers also accept earlier UBJSON models mistakenly named `.json`,
including rollback copies, so a valid saved model does not trigger unnecessary
retraining merely because of its filename.

## Published forecast evidence

After publishing `FORECAST_JSON_PATH`, the job records a compact gzip snapshot in
the adjacent `forecast-history/` directory. Each record contains the final
facility counts, their timestamps and thresholds, `generatedAt`, and a separate
`publishedAt` captured after publication. Records are immutable and retained for
90 days; unrelated files are not pruned. These are backend artifacts and must
never be included in the frontend deployment.

For genuine forecast evaluation, compare only targets after `publishedAt` with
qualified actuals, grouped by lead time. Compare the final served counts with a
simple baseline. The historical chart and the terminal model metrics documented above are
not substitutes for that evaluation. Archiving makes future verification possible;
it does not itself establish an accuracy score.

## Hourly verification

`server/verify_forecasts.py` compares immutable published forecasts with completed
hourly attendance in a read-only database transaction. It produces private,
cumulative `hourly-comparisons.txt` and `summary.txt` files, plus
`verification-state.json` with the exact archive and timestamps behind each row.
These files are operational records, not browser assets.

Two separate groups measure predictions published **1–2 hours** and **24–25
hours** before each target hour starts (upper bounds exclusive). The newest
eligible publication is selected. Four quarter-hour predictions are averaged
and rounded like the hourly chart; forecasts revised after the cutoff cannot
replace them. Scored records remain unchanged on subsequent runs. Crowd-category
agreement uses each prediction's archived thresholds for both counts.

Only completed, fully open hours with qualified attendance are scored. Closed,
partly open, unknown-schedule, missing-forecast, and insufficient-observation
hours remain visible with an exclusion status. Missing data is never treated as
zero or perfect accuracy. Attendance represents recorded hourly average
occupancy, not unique visitors or an independent validation of the source counter.
Official opening-hour eligibility uses the current fresh schedule for the target
date; historical changes to that schedule are not independently archived.

The summary reports sample counts, mean absolute error in people, signed bias
(positive means overprediction), root mean squared error, and crowd-category
agreement, separately for each gym and lead group. It does not yet compare the
model with a simple baseline or establish an overall accuracy percentage.

Use the backend environment and existing Python runtime:

```bash
python server/verify_forecasts.py \
  --archive-dir /path/to/shared/forecast-history \
  --output-dir /path/to/shared/forecast-verification \
  --hourly
```

Invoke from one existing scheduler each minute, or hourly after minute 07.
`--hourly` uses a process lock and persistent successful-run marker to perform
one check per UTC hour, after seven minutes of ingestion grace. Failed runs
remain retryable; retries cannot duplicate rows. Each run revisits the last
three Chicago dates for delayed observations or brief outages. For a longer
outage, omit `--hourly` and use `--lookback-days N` (up to 60). Records already
scored remain unchanged. Text reports and state are replaced atomically, with
state committed last so interrupted report writes are retried. Keep the output
directory private and preserve it across releases; reports have no automatic
pruning. Source forecast archives retain their existing 90-day policy.
