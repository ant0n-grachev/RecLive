"""Orchestrate a read-only hourly forecast comparison."""
import fcntl
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .verification_report import is_due, save_report


def run_verification(archive_dir, output_dir, settings, now=None, *, lookback_days=3,
                     hourly=False, observations_loader=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('verification_time_requires_timezone')
    if not 1 <= lookback_days <= 60:
        raise ValueError('lookback_days_must_be_1_to_60')
    directory = Path(output_dir)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (directory / 'verification.lock').open('a') as lock:
        os.chmod(lock.name, 0o600)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'already_running'}
        if hourly and not is_due(directory, now):
            return {'status': 'not_due'}
        from .verification import evaluate_hour, load_predictions
        if observations_loader is None:
            from .verification_observations import collect_actual_hours
            observations_loader = collect_actual_hours
        chicago = ZoneInfo('America/Chicago')
        today = now.astimezone(chicago).date()
        dates = [(today - timedelta(days=i)).isoformat() for i in reversed(range(lookback_days))]
        min_hour = datetime.fromisoformat(dates[0]).replace(tzinfo=chicago)
        predictions, invalid = load_predictions(archive_dir, now, min_hour=min_hour)
        observations = observations_loader(settings, dates, now)
        records = []
        for actual in observations:
            start = datetime.fromisoformat(actual['hourStart'])
            # Eligibility is determined by the official schedule and successful observation coverage.
            rows = evaluate_hour(predictions, actual['facilityId'], start, actual, now)
            observation_status = actual.get('observationStatus', 'ready')
            if observation_status in ('closed', 'partial_open', 'schedule_unavailable'):
                for row in rows:
                    row.update(status=observation_status, actualCount=None, actualPct=None,
                               actualCrowd=None, errorPeople=None, absoluteError=None, crowdMatch=None)
            for row in rows:
                row['observationStatus'] = observation_status
                row['skipReason'] = actual.get('skipReason')
            records.extend(rows)
        state = save_report(directory, records, now, invalid_archives=invalid)
        return {'status': 'completed', 'checkedAt': now.isoformat(),
                'records': len(state['records']),
                'scored': sum(r['status'] == 'scored' for r in state['records']),
                'invalidArchives': invalid}
