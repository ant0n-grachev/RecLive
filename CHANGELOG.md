# Changelog

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/).

## Unreleased

### Fixed

- Keep all occupancy observations from a timestamp together during model tuning
  so that validation always uses later observations than training.
