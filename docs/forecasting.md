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
