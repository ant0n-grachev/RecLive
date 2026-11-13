# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## Unreleased

### Changed

- Prepare `v0.0.1` for the tuning fix and check the application version against
  both npm lockfile root fields in CI.
- Explain local setup, data flow, and testing in the README and keep deployment
  procedures in the operator runbook.
- Clarify the distinction between training diagnostics and the hourly
  comparisons of published forecasts, including the missing baseline comparison.

### Fixed

- Keep all occupancy observations from a timestamp together during model tuning
  so that validation always uses later observations than training.
- Run frontend CI on Ubuntu 24.04 so the pinned Playwright version can install
  Chromium and run browser checks.
