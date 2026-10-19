import os
import time
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List
from inference import FallPredictor
from analytics import activity_metrics, publish_analytics, prune_analytics
from pydantic import ConfigDict
import hmac
import hashlib
import json

app = FastAPI(title='Smart Safety Shoe ML Inference')
model = FallPredictor(str(Path(__file__).with_name('fall_rf_sisfall.joblib')))

class Sample(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    t: int = Field(ge=0)  # ESP32 millis()

class Location(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

class Window(BaseModel):
    # Optional Unix window end; legacy firmware remains compatible.
    capturedAt: int | None = Field(default=None, ge=1)
    location: Location | None = None
    samples: List[Sample] = Field(min_length=150, max_length=150)

@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'RandomForestClassifier', 'window': model.win, 'samplingHz': model.fs}

@app.post('/predict')
def predict(payload: Window, x_api_key: str | None = Header(default=None)):
    key = os.environ.get('ML_API_KEY')
    if not key or not x_api_key or not hmac.compare_digest(x_api_key.encode(), key.encode()):
        raise HTTPException(status_code=401, detail='Unauthorized')
    samples = [s.model_dump() for s in payload.samples]
    times = [s['t'] for s in samples]
    intervals = [b - a for a, b in zip(times, times[1:])]
    # Check sample cadence. ESP32 millis wraps every ~49 days; reset window then.
    if not all(10 <= dt <= 40 for dt in intervals):
        raise HTTPException(status_code=422, detail='Sampling interval outside 10–40 ms; cannot trust 50 Hz window')
    if not (2600 <= times[-1] - times[0] <= 3400):
        raise HTTPException(status_code=422, detail='Window duration is not approximately 3 seconds')
    try:
        result = model.predict(samples)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result['timestamp'] = int(time.time() * 1000)
    result['source'] = 'sisfall-rf'
    metrics = activity_metrics(samples)
    end = payload.capturedAt or result['timestamp']
    if abs(end - result['timestamp']) > 60_000:
        raise HTTPException(status_code=422, detail='capturedAt must be within 60 seconds of server time')
    start = end - metrics['durationMs']
    result['activity'] = metrics
    # Publish ML results separately; never overwrite sensorData/fallDetected.
    # Firebase integration is enabled only with valid Admin SDK credentials.
    if os.environ.get('FIREBASE_PROJECT_ID') and os.environ.get('FIREBASE_DATABASE_URL'):
        publish_prediction(result, metrics, start, end, payload.location.model_dump() if payload.location else None, hashlib.sha256(json.dumps(payload.model_dump(), sort_keys=True).encode()).hexdigest())
    return result

_firebase_db = None
_last_pruned_day = None

def firebase_database():
    global _firebase_db
    if _firebase_db is None:
        import firebase_admin
        from firebase_admin import credentials, db
        private_key = os.environ.get('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n')
        email = os.environ.get('FIREBASE_CLIENT_EMAIL', '')
        if not private_key or not email:
            raise HTTPException(status_code=503, detail='Firebase credentials not configured')
        cred = credentials.Certificate({'type': 'service_account', 'project_id': os.environ['FIREBASE_PROJECT_ID'], 'private_key': private_key, 'client_email': email, 'token_uri': 'https://oauth2.googleapis.com/token'})
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': os.environ['FIREBASE_DATABASE_URL']})
        _firebase_db = db
    return _firebase_db


def publish_prediction(result, metrics, start, end, location, window_id):
    db = firebase_database()
    try:
        # Separate node; existing ESP32 sensorData PUTs cannot erase ML results.
        db.reference('mlPrediction').transaction(lambda old: result if not old or old.get('timestamp', 0) < result['timestamp'] else old)
        if os.environ.get('ENABLE_ANALYTICS') == 'true':
            publish_analytics(db, result, metrics, start, end, location, window_id)
        # Run at most once per UTC day per worker; retries remain safe.
        global _last_pruned_day
        day = result['timestamp'] // 86_400_000
        if os.environ.get('ENABLE_ANALYTICS') == 'true' and day != _last_pruned_day:
            prune_analytics(db, result['timestamp'])
            _last_pruned_day = day
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Firebase write failed') from exc
