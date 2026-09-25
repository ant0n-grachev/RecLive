"""Reports preserve fair comparisons across retries and process restarts."""
import json
from datetime import datetime, timezone

from server.reclive.forecasting import verification_report as report


def row(**changes):
    return dict({
        'facilityId': 1186, 'hourStart': '2026-09-25T10:00:00-05:00',
        'leadHours': 1, 'status': 'scored', 'predictedCount': 120,
        'actualCount': 100, 'errorPeople': 20, 'absoluteError': 20,
        'crowdMatch': True, 'predictedCrowd': 'medium', 'actualCrowd': 'medium',
        'publishedAt': '2026-09-25T08:55:00-05:00', 'actualLeadHours': 1.0833,
    }, **changes)


def test_retry_does_not_duplicate_or_replace_already_scored_forecast(tmp_path):
    now = datetime(2026, 9, 25, 17, tzinfo=timezone.utc)
    report.save_report(tmp_path, [row()], now, invalid_archives=0)
    report.save_report(tmp_path, [row(predictedCount=100, absoluteError=0)], now, invalid_archives=0)
    stored = json.loads((tmp_path / 'verification-state.json').read_text())
    assert len(stored['records']) == 1
    assert stored['records'][0]['predictedCount'] == 120
    assert '20.00' in (tmp_path / 'summary.txt').read_text()
    assert (tmp_path / 'hourly-comparisons.txt').read_text().count('2026-09-25T10:00:00-05:00') == 1


def test_missing_actual_can_be_filled_later_without_losing_other_hours(tmp_path):
    now = datetime(2026, 9, 25, 17, tzinfo=timezone.utc)
    report.save_report(tmp_path, [row(status='insufficient_actual', actualCount=None,
        absoluteError=None, errorPeople=None, crowdMatch=None)], now, invalid_archives=0)
    report.save_report(tmp_path, [row(), row(hourStart='2026-09-25T11:00:00-05:00',
        absoluteError=40, errorPeople=-40, crowdMatch=False)], now, invalid_archives=0)
    summary = (tmp_path / 'summary.txt').read_text()
    assert 'MAE=30.00' in summary
    assert 'bias=-10.00' in summary
    assert 'crowd_match=50.0%' in summary
    assert 'scored=2' in summary


def test_empty_scores_are_unavailable_not_perfect(tmp_path):
    now = datetime(2026, 9, 25, 17, tzinfo=timezone.utc)
    report.save_report(tmp_path, [row(status='no_forecast', predictedCount=None,
        absoluteError=None, errorPeople=None, crowdMatch=None)], now, invalid_archives=2)
    summary = (tmp_path / 'summary.txt').read_text()
    assert 'MAE=n/a' in summary
    assert 'crowd_match=n/a' in summary
    assert 'invalid_archives=2' in summary
    assert (tmp_path / 'verification-state.json').stat().st_mode & 0o777 == 0o600


def test_hourly_gate_waits_for_grace_and_survives_restart(tmp_path):
    first = datetime(2026, 9, 25, 17, 7, tzinfo=timezone.utc)
    assert not report.is_due(tmp_path, first.replace(minute=6))
    assert report.is_due(tmp_path, first)
    report.save_report(tmp_path, [], first, invalid_archives=0)
    assert not report.is_due(tmp_path, first.replace(minute=50))
    assert report.is_due(tmp_path, first.replace(hour=18))


def test_corrupt_existing_state_is_not_silently_erased(tmp_path):
    (tmp_path / 'verification-state.json').write_text('{')
    import pytest
    with pytest.raises(ValueError):
        report.save_report(tmp_path, [], datetime.now(timezone.utc), invalid_archives=0)
    assert (tmp_path / 'verification-state.json').read_text() == '{'
