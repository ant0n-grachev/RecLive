# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- Explain local setup, data flow, and testing in the README and keep deployment
  procedures in the operator runbook.
- Clarify the distinction between training diagnostics and the hourly
  comparisons of published forecasts, including the missing baseline comparison.

### Fixed

- Keep all occupancy observations from a timestamp together during model tuning
  so that validation always uses later observations than training.
