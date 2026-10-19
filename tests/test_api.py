import os
import sys
from pathlib import Path
# Tests explicitly disable production access before importing the service.
for name in list(os.environ):
    if name.startswith('FIREBASE_'):
        os.environ.pop(name)
os.environ['ML_API_KEY'] = 'local-test-only'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
import app
from test_analytics import samples, MemoryDB

client = TestClient(app.app)
headers = {'X-API-Key': 'local-test-only'}


def test_health_and_auth():
    assert client.get('/health').json()['window'] == 150
    assert client.post('/predict', json={'samples': samples()}).status_code == 401


def test_real_saved_model_and_threshold_unchanged():
    response = client.post('/predict', json={'samples': samples(.2)}, headers=headers)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['threshold'] == app.model.threshold
    assert result['fallDetected'] == (result['probability'] >= result['threshold'])
    assert result['activity']['durationMs'] == 3000
    assert result['activity']['intensity'] == 'moderate'


def test_invalid_cadence_range_and_clock():
    for payload in [
        {'samples': samples()[:100]},
        {'samples': [{**s, 't': i*50} for i, s in enumerate(samples())]},
        {'samples': [{**s, 'az': 20} for s in samples()]},
        {'samples': samples(), 'capturedAt': 1000},
        {'samples': samples(), 'location': {'latitude': 99, 'longitude': 0}},
    ]:
        assert client.post('/predict', json=payload, headers=headers).status_code == 422


def test_firebase_publish_mocked_only(monkeypatch):
    from unittest.mock import Mock
    db = MemoryDB()
    # Provide only a fake database; no Firebase Admin initialization can occur.
    monkeypatch.setattr(app, '_firebase_db', db)
    monkeypatch.setattr(app, 'prune_analytics', Mock())
    monkeypatch.setenv('ENABLE_ANALYTICS', 'true')
    app.publish_prediction({'timestamp': 103000, 'probability': .1, 'fallDetected': False},
        {'durationMs': 3000, 'intensity': 'low'}, 100000, 103000,
        {'latitude': 13.0, 'longitude': 80.0}, 'window')
    assert db.values['mlPrediction']['probability'] == .1
    assert db.values['analytics/lastLocation']['latitude'] == 13
    assert all(not path.startswith('sensorData') for path in db.values)
