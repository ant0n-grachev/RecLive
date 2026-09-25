"""The scheduled command excludes closed hours and preserves catch-up evidence."""
import gzip
import json
from datetime import datetime, timezone

from server.reclive.forecasting import verification_runner as runner


def test_run_keeps_closed_hours_out_of_metrics_and_deduplicates_retry(tmp_path):
    archives = tmp_path / 'archives'
    archives.mkdir()
    archive = {'schemaVersion': 1, 'timezone': 'America/Chicago',
        'generatedAt': '2026-09-25T07:40:00-05:00',
        'publishedAt': '2026-09-25T07:50:00-05:00',
        'facilities': [{'facilityId': 1186, 'occupancyThresholds': {'lowMax': 11, 'peakMin': 26},
            'weeklyForecast': [{'date': '2026-09-25', 'totalHours': [
                {'hourStart': f'2026-09-25T09:{minute:02}:00-05:00',
                 'expectedCount': 120, 'expectedPct': .12} for minute in (0, 15, 30, 45)]}]}]}
    (archives / 'forecast-test.json.gz').write_bytes(gzip.compress(json.dumps(archive).encode()))
    actual = {'facilityId': 1186, 'hourStart': '2026-09-25T09:00:00-05:00',
        'actualCount': 100, 'actualPct': .10, 'actualCoverage': 1,
        'temporalCoverage': 1, 'coverageThreshold': .75, 'expectedCapacity': 1000,
        'observationStatus': 'ready'}
    def observations(settings, dates, now):
        return [actual, dict(actual, hourStart='2026-09-25T08:00:00-05:00',
                             observationStatus='closed', actualCount=None)]
    now = datetime(2026, 9, 25, 15, 7, tzinfo=timezone.utc)
    for _ in range(2):
        result = runner.run_verification(archives, tmp_path / 'out', None, now,
                                        observations_loader=observations)
    assert result['scored'] == 1
    state = json.loads((tmp_path / 'out/verification-state.json').read_text())
    assert len(state['records']) == 4
    assert sum(r['status'] == 'closed' for r in state['records']) == 2
    assert 'MAE=20.00' in (tmp_path / 'out/summary.txt').read_text()


def test_hourly_retry_within_hour_skips_expensive_observation_read(tmp_path):
    from server.reclive.forecasting.verification_report import save_report
    now = datetime(2026, 9, 25, 15, 7, tzinfo=timezone.utc)
    save_report(tmp_path, [], now)
    def forbidden(*args):
        raise AssertionError('already completed hour must not query database')
    assert runner.run_verification(tmp_path, tmp_path, None, now, hourly=True,
                                   observations_loader=forbidden)['status'] == 'not_due'


def test_failed_read_leaves_hour_retryable(tmp_path):
    import pytest
    def failed(*args):
        raise RuntimeError('database_unavailable')
    now = datetime(2026, 9, 25, 15, 7, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match='database_unavailable'):
        runner.run_verification(tmp_path, tmp_path, None, now, observations_loader=failed)
    assert not (tmp_path / 'verification-state.json').exists()
