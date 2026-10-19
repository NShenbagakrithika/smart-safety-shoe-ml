import copy
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analytics import activity_metrics, update_day, split_days, publish_analytics, DAY_MS


def samples(amplitude=0, gyro=0):
    return [dict(ax=0, ay=0, az=1 + amplitude * math.sin(i * .4), gx=gyro, gy=0, gz=0, t=i*20) for i in range(150)]


def test_intensity_and_duration():
    for amplitude, gyro, expected in [(0, 0, 'inactive'), (.05, 0, 'low'), (.2, 0, 'moderate'), (.5, 0, 'high'), (0, 100, 'high')]:
        metric = activity_metrics(samples(amplitude, gyro))
        assert metric['intensity'] == expected
        assert metric['durationMs'] == 3000


def test_gaps_never_count_as_inactivity():
    day = update_day(None, 1000, 4000, 'inactive', .1)
    day = update_day(day, 4000, 7000, 'inactive', .1)
    assert day['longestInactivityMs'] == 6000
    day = update_day(day, 100000, 103000, 'inactive', .1)
    assert day['inactivityMs'] == 9000
    assert day['currentInactivityMs'] == 3000
    assert day['longestInactivityMs'] == 6000


def test_duplicate_overlap_and_late_requests():
    day = update_day(None, 1000, 4000, 'low', .1, 'a')
    assert update_day(day, 1000, 4000, 'low', .1, 'a') == day
    assert update_day(day, 5000, 8000, 'low', .1, 'a') == day
    day = update_day(day, 3000, 6000, 'high', .8, 'b')
    assert day['movementMs'] == 5000
    assert update_day(day, 1000, 2000, 'inactive', .1, 'c') == day
    assert sum(b.get('movementMs', 0) for b in day['timeline'].values()) == 5000


def test_midnight_and_bucket_boundaries():
    segments = list(split_days(DAY_MS-1000, DAY_MS+2000))
    assert len(segments) == 2
    assert sum(b-a for _, a, b in segments) == 3000
    day = update_day(None, 899000, 902000, 'low', .3)
    assert day['timeline']['00']['movementMs'] == 1000
    assert day['timeline']['01']['movementMs'] == 2000


class MemoryDB:
    def __init__(self): self.values = {}
    def reference(self, path):
        db = self
        class Ref:
            def transaction(self, callback):
                original = copy.deepcopy(db.values.get(path))
                # Emulate Firebase retry: callbacks must be pure.
                first = callback(copy.deepcopy(original))
                second = callback(copy.deepcopy(original))
                assert first == second
                db.values[path] = second
                return second
        return Ref()


def test_publish_retry_does_not_invent_recent_activity():
    db = MemoryDB()
    result = {'probability': .1, 'timestamp': 103000, 'fallDetected': False}
    metric = activity_metrics(samples(.1))
    publish_analytics(db, result, metric, 100000, 103000, window_id='same')
    publish_analytics(db, result, metric, 110000, 113000, window_id='same')
    assert db.values['analytics/latest']['windowEnd'] == 103000
    assert db.values['analytics/lastMovementAt'] == 103000
    assert db.values['analytics/days/1970-01-01']['movementMs'] == 3000
    assert not any(k.startswith('sensorData') for k in db.values)


def test_retention_targets_only_expired_analytics():
    from maintenance import expired_paths
    reads = []
    class RetentionDB:
        def reference(self, path):
            reads.append(path)
            class Ref:
                def order_by_key(self): return self
                def order_by_child(self, key): assert key == 'timestamp'; return self
                def end_at(self, value): return self
                def get(self):
                    return {'2020-01-01': {}} if path.endswith('/days') else {'expired': {}} if path.endswith('/events') else None
            return Ref()
    assert expired_paths(RetentionDB(), 1792368000000) == ['days/2020-01-01', 'events/expired']
    assert all(p.startswith('analytics/') for p in reads)
