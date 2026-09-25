"""Compare published forecasts with recorded attendance; never write to the DB."""
import argparse
import json
import os

if not __package__:
    import reclive  # noqa: F401

from server.env_loader import load_project_dotenv
from server.reclive.forecasting.verification_runner import run_verification
from server.reclive.settings import Settings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--lookback-days', type=int, default=3,
                        help='Revisit recent dates to recover missed/late hours (1-60; default 3).')
    parser.add_argument('--hourly', action='store_true',
                        help='Run once per UTC hour, after minute 07; safe to invoke every minute.')
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        load_project_dotenv()
        result = run_verification(args.archive_dir, args.output_dir, Settings.from_environment(),
                                  lookback_days=args.lookback_days, hourly=args.hourly)
        if result['status'] == 'completed':
            print(json.dumps(dict(event='forecast_verification.completed', **result)))
        return 0
    except Exception as error:
        # Exception messages may contain connection details or secrets.
        print(json.dumps({'event': 'forecast_verification.failed', 'errorType': type(error).__name__}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
