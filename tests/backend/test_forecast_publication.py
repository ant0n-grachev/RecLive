"""Forecast publication preserves the predictions actually available to users."""
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from server.reclive.forecasting import config, job


def payload():
    return {
        'generatedAt': '2026-09-24T17:20:00-05:00',
        'timezone': 'America/Chicago',
        'modelInfo': {'status': 'using_saved_model', 'privateInternalPath': 'must-not-archive'},
        'facilities': [{
            'facilityId': 1186,
            'occupancyThresholds': {'lowMax': 11, 'peakMin': 26},
            'weeklyForecast': [{
                'date': '2026-09-24',
                'totalHours': [{'hourStart': '2026-09-24T18:00:00-05:00', 'expectedCount': 237, 'expectedPct': .2305}],
                'weatherHours': [{'unrelated': 'omit'}],
            }],
        }],
    }


def read_archive(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return json.load(handle)


def test_publication_retains_final_totals_and_actual_publication_time(monkeypatch, tmp_path):
    destination = tmp_path / 'forecast.json'
    monkeypatch.setattr(config, 'FORECAST_JSON_PATH', str(destination))
    before = datetime.now(timezone.utc)
    job.write_forecast(payload())
    after = datetime.now(timezone.utc)
    saved = list((tmp_path / 'forecast-history').glob('*.json.gz'))
    assert len(saved) == 1
    archive = read_archive(saved[0])
    assert before <= datetime.fromisoformat(archive['publishedAt']) <= after
    assert archive['generatedAt'] == '2026-09-24T17:20:00-05:00'
    assert archive['facilities'][0]['weeklyForecast'][0]['totalHours'] == [
        {'hourStart': '2026-09-24T18:00:00-05:00', 'expectedCount': 237, 'expectedPct': .2305}
    ]
    assert archive['facilities'][0]['occupancyThresholds'] == {'lowMax': 11, 'peakMin': 26}
    assert 'must-not-archive' not in json.dumps(archive)
    assert 'weatherHours' not in archive['facilities'][0]['weeklyForecast'][0]
    assert json.loads(destination.read_text()) == payload()


def test_republication_preserves_old_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'FORECAST_JSON_PATH', str(tmp_path / 'forecast.json'))
    original = payload()
    job.write_forecast(original)
    revised = payload()
    revised['facilities'][0]['weeklyForecast'][0]['totalHours'][0]['expectedCount'] = 400
    job.write_forecast(revised)
    archives = [read_archive(path) for path in (tmp_path / 'forecast-history').glob('*.json.gz')]
    assert sorted(x['facilities'][0]['weeklyForecast'][0]['totalHours'][0]['expectedCount'] for x in archives) == [237, 400]


def test_failed_publication_does_not_archive_unavailable_forecast(monkeypatch, tmp_path):
    destination = tmp_path / 'forecast.json'
    destination.write_text('{"previous":true}')
    monkeypatch.setattr(config, 'FORECAST_JSON_PATH', str(destination))
    original_replace = job.os.replace

    def fail_publish(source, target):
        if Path(target) == destination:
            raise OSError('synthetic publication failure')
        return original_replace(source, target)

    monkeypatch.setattr(job.os, 'replace', fail_publish)
    import pytest
    with pytest.raises(OSError):
        job.write_forecast(payload())
    assert json.loads(destination.read_text()) == {'previous': True}
    assert not list((tmp_path / 'forecast-history').glob('*.json.gz'))


def test_retention_prunes_only_owned_old_archives(monkeypatch, tmp_path):
    monkeypatch.setattr(config, 'FORECAST_JSON_PATH', str(tmp_path / 'forecast.json'))
    from server.reclive.forecasting import publication
    old = datetime.now(timezone.utc) - timedelta(days=91)
    archive = publication.archive_published_forecast(payload(), str(tmp_path / 'forecast.json'), old)
    sentinel = archive.parent / 'operator-note.txt'
    sentinel.write_text('retain')
    job.write_forecast(payload())
    assert not archive.exists()
    assert sentinel.read_text() == 'retain'
    assert len(list(archive.parent.glob('*.json.gz'))) == 1
