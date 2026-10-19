"""Deterministic activity estimates, independent of the fall classifier.
Only observed windows contribute time; gaps are unknown, never inactivity.
"""
import math
from datetime import datetime, timezone, timedelta

DAY_MS = 86_400_000
BUCKET_MS = 900_000
RETENTION_DAYS = 90


def activity_metrics(samples):
    duration = samples[-1]['t'] - samples[0]['t'] + 20
    magnitudes = [math.sqrt(sum(s[k] ** 2 for k in ('ax', 'ay', 'az'))) for s in samples]
    mean = sum(magnitudes) / len(magnitudes)
    rms = math.sqrt(sum((v - mean) ** 2 for v in magnitudes) / len(magnitudes))
    gyro = math.sqrt(sum(sum(s[k] ** 2 for k in ('gx', 'gy', 'gz')) for s in samples) / len(samples))
    # Transparent heuristics in g and deg/s, not another ML model.
    intensity = 'high' if rms >= .30 or gyro >= 90 else 'moderate' if rms >= .10 or gyro >= 30 else 'low' if rms >= .025 or gyro >= 5 else 'inactive'
    return {'durationMs': duration, 'intensity': intensity, 'accelerationVariationG': round(rms, 5)}


def split_days(start, end):
    while start < end:
        day = datetime.fromtimestamp(start / 1000, timezone.utc).strftime('%Y-%m-%d')
        boundary = (start // DAY_MS + 1) * DAY_MS
        stop = min(end, boundary)
        yield day, start, stop
        start = stop


def update_day(current, start, end, intensity, probability, window_id=None):
    """Firebase transaction callback: safe on retries, duplicate/overlap clipped.
    Late windows preceding the high-water mark are ignored, not double counted.
    """
    d = dict(current or {})
    seen = dict(d.get('recentWindows') or {})
    if window_id and window_id in seen:
        return d
    last = int(d.get('lastEnd', 0))
    start = max(start, last)
    if end <= start:
        return d
    if window_id:
        seen[window_id] = end
        d['recentWindows'] = dict(sorted(seen.items(), key=lambda item: item[1])[-64:])
    duration = end - start
    inactive = intensity == 'inactive'
    d['observedMs'] = d.get('observedMs', 0) + duration
    key = 'inactivityMs' if inactive else 'movementMs'
    d[key] = d.get(key, 0) + duration
    run = (d.get('currentInactivityMs', 0) if start == last else 0) + duration if inactive else 0
    d['currentInactivityMs'] = run
    d['longestInactivityMs'] = max(d.get('longestInactivityMs', 0), run)
    d['lastEnd'] = end
    if not inactive:
        d['lastMovementAt'] = end
    levels = dict(d.get('intensityMs') or {})
    levels[intensity] = levels.get(intensity, 0) + duration
    d['intensityMs'] = levels
    buckets = {k: dict(v) for k, v in (d.get('timeline') or {}).items()}
    while start < end:
        stop = min(end, (start // BUCKET_MS + 1) * BUCKET_MS)
        key = str((start % DAY_MS) // BUCKET_MS).zfill(2)
        b = buckets.setdefault(key, {})
        metric = 'inactivityMs' if inactive else 'movementMs'
        b[metric] = b.get(metric, 0) + stop - start
        b['probability'] = probability
        b['maxProbability'] = max(b.get('maxProbability', 0), probability)
        b['timestamp'] = stop
        start = stop
    d['timeline'] = buckets
    return d


def publish_analytics(db, result, metrics, start, end, location=None, window_id=None):
    # No mutations of legacy paths, fall flags or model threshold.
    committed_ends = []
    for day, a, b in split_days(start, end):
        committed = db.reference('analytics/days/' + day).transaction(
            lambda value, a=a, b=b: update_day(value, a, b, metrics['intensity'], result['probability'], window_id))
        if window_id and window_id in (committed.get('recentWindows') or {}):
            committed_ends.append(committed['recentWindows'][window_id])
    if window_id and not committed_ends:
        return  # late window, no newly observed activity
    if committed_ends:
        end = max(committed_ends)
        start = end - metrics['durationMs']
    latest = {**result, **metrics, 'timestamp': end, 'windowStart': start, 'windowEnd': end}
    if location:
        latest['location'] = location
    db.reference('analytics/latest').transaction(
        lambda old: latest if not old or old.get('windowEnd', 0) < end else old)
    if metrics['intensity'] != 'inactive':
        db.reference('analytics/lastMovementAt').transaction(lambda old: max(old or 0, end))
    if location:
        db.reference('analytics/lastLocation').transaction(
            lambda old: {**location, 'timestamp': end} if not old or old.get('timestamp', 0) < end else old)


def prune_analytics(db, now_ms):
    cutoff = datetime.fromtimestamp(now_ms / 1000, timezone.utc) - timedelta(days=RETENTION_DAYS)
    # Query only expired daily records, bounded deletion batches; never legacy paths.
    expired = db.reference('analytics/days').order_by_key().end_at(cutoff.strftime('%Y-%m-%d')).limit_to_first(100).get() or {}
    if expired:
        db.reference('analytics/days').update({key: None for key in expired})
