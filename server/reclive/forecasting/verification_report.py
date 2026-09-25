"""Private, cumulative hourly verification reports; caller holds the run lock."""
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

FACILITIES = {1186: 'Nick', 1656: 'Bakke'}
STATE_FILE = 'verification-state.json'


def read_state(directory):
    path = Path(directory) / STATE_FILE
    if not path.exists():
        return {'schemaVersion': 1, 'records': [], 'lastRunAt': None}
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('schemaVersion') != 1 or not isinstance(state.get('records'), list):
        raise ValueError('unsupported_verification_state')
    return state


def is_due(directory, now):
    """One successful run per UTC hour, after seven minutes for late ingestion."""
    now = now.astimezone(timezone.utc)
    if now.minute < 7:
        return False
    last = read_state(directory).get('lastRunAt')
    return not last or datetime.fromisoformat(last) < now.replace(minute=0, second=0, microsecond=0)


def _atomic_write(path, content):
    pending = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.pending-', delete=False) as handle:
            pending = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _key(row):
    return (int(row['facilityId']), datetime.fromisoformat(row['hourStart']).astimezone(timezone.utc),
            int(row['leadHours']))


def render_summary(records, now, invalid_archives):
    lines = [
        'RecLive forecast verification', f'Updated: {now.isoformat()}',
        'Counts are hourly average recorded attendance, not unique visitors.',
        '1h: forecast published 1-2 hours before the hour; 24h: published 24-25 hours before it.',
        'Each target uses the latest eligible publication, before attendance was known.',
        'Only completed, fully open hours with qualified attendance are scored.',
        'Recorded source counts may themselves be delayed; this does not validate the counter.',
        'Positive bias means overprediction. Crowd match uses archived forecast thresholds.',
        'No scores means n/a, not zero error. Status counts include excluded hours.',
        f'invalid_archives={invalid_archives}', '',
    ]
    for facility_id, name in FACILITIES.items():
        for horizon in (1, 24):
            subset = [r for r in records if r['facilityId'] == facility_id and r['leadHours'] == horizon]
            scored = [r for r in subset if r['status'] == 'scored']
            if scored:
                mae = sum(r['absoluteError'] for r in scored) / len(scored)
                bias = sum(r['errorPeople'] for r in scored) / len(scored)
                rmse = math.sqrt(sum(r['errorPeople'] ** 2 for r in scored) / len(scored))
                metrics = f'MAE={mae:.2f} people; bias={bias:.2f}; RMSE={rmse:.2f}'
            else:
                metrics = 'MAE=n/a; bias=n/a; RMSE=n/a'
            matches = [r['crowdMatch'] for r in scored if isinstance(r.get('crowdMatch'), bool)]
            accuracy = f'{100 * sum(matches) / len(matches):.1f}%' if matches else 'n/a'
            lines.append(f'{name} | {horizon}h | scored={len(scored)} | {metrics} | crowd_match={accuracy} (n={len(matches)})')
            lines.append('  statuses: ' + ', '.join(f'{k}={v}' for k, v in sorted(Counter(r['status'] for r in subset).items())))
    return '\n'.join(lines) + '\n'


def render_comparisons(records, now):
    columns = ('hourStart', 'facility', 'leadHours', 'predictedCount', 'actualCount',
               'errorPeople', 'absoluteError', 'predictedCrowd', 'actualCrowd', 'crowdMatch',
               'status', 'actualLeadHours', 'publishedAt', 'actualCoverage', 'temporalCoverage')
    lines = ['RecLive hourly forecast comparisons', f'Updated: {now.isoformat()}',
             'Tab-separated; n/a means unavailable. Error = predicted minus recorded attendance.',
             'Times include the Chicago UTC offset. See summary.txt for scoring rules.',
             '\t'.join(columns)]
    for record in records:
        values = dict(record, facility=FACILITIES.get(record['facilityId'], str(record['facilityId'])))
        lines.append('\t'.join('n/a' if values.get(c) is None else str(values[c]) for c in columns))
    return '\n'.join(lines) + '\n'


def save_report(directory, records, now, *, invalid_archives=0):
    directory = Path(directory)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    state = read_state(directory)
    combined = {_key(r): r for r in state['records']}
    for row in records:
        key = _key(row)
        # A later revision, changed threshold, or retry must never improve a scored forecast.
        if combined.get(key, {}).get('status') != 'scored':
            combined[key] = row
    ordered = [combined[k] for k in sorted(combined, key=lambda k: (k[1], k[0], k[2]))]
    updated = dict(state, records=ordered, lastRunAt=now.isoformat(), invalidArchives=invalid_archives)
    # Commit state last: an interrupted text write leaves the run eligible for a safe retry.
    _atomic_write(directory / 'hourly-comparisons.txt', render_comparisons(ordered, now))
    _atomic_write(directory / 'summary.txt', render_summary(ordered, now, invalid_archives))
    _atomic_write(directory / STATE_FILE, json.dumps(updated, allow_nan=False, indent=2) + '\n')
    return updated
