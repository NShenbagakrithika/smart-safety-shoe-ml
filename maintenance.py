"""Daily retention job. Dry-run by default; deployment/scheduling is manual."""
import argparse
import os
import time
from datetime import datetime, timezone, timedelta


def expired_paths(db, now_ms):
    day_cutoff = (datetime.fromtimestamp(now_ms / 1000, timezone.utc) - timedelta(days=90)).strftime('%Y-%m-%d')
    days = db.reference('analytics/days').order_by_key().end_at(day_cutoff).get() or {}
    events = db.reference('analytics/events').order_by_child('timestamp').end_at(now_ms - 365*86400000).get() or {}
    paths = ['days/' + key for key in days] + ['events/' + key for key in events]
    for key, timestamp_key in [('latest', 'windowEnd'), ('lastLocation', 'timestamp')]:
        data = db.reference('analytics/' + key).get()
        if data and data.get(timestamp_key, 0) < now_ms - 365*86400000:
            paths.append(key)
    last = db.reference('analytics/lastMovementAt').get()
    if last and last < now_ms - 365*86400000:
        paths.append('lastMovementAt')
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True, help='Must match configured FIREBASE_PROJECT_ID')
    parser.add_argument('--execute', action='store_true', help='Apply deletions (default: list counts only)')
    args = parser.parse_args()
    if args.project != os.environ.get('FIREBASE_PROJECT_ID'):
        parser.error('Project does not match configured target')
    from app import firebase_database
    db = firebase_database()
    paths = expired_paths(db, int(time.time()*1000))
    print(f'{len(paths)} expired analytics records; execute={args.execute}')
    if args.execute:
        for i in range(0, len(paths), 200):
            db.reference('analytics').update({p: None for p in paths[i:i+200]})


if __name__ == '__main__':
    main()
