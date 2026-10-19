import os
import time
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import List
from inference import FallPredictor

app = FastAPI(title='Smart Safety Shoe ML Inference')
model = FallPredictor(str(Path(__file__).with_name('fall_rf_sisfall.joblib')))

class Sample(BaseModel):
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    t: int = Field(ge=0)  # ESP32 millis()

class Window(BaseModel):
    samples: List[Sample] = Field(min_length=150, max_length=150)

@app.get('/health')
def health():
    return {'status': 'ok', 'model': 'RandomForestClassifier', 'window': model.win, 'samplingHz': model.fs}

@app.post('/predict')
def predict(payload: Window, x_api_key: str | None = Header(default=None)):
    key = os.environ.get('ML_API_KEY')
    if not key or x_api_key != key:
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
    # Publish ML results separately; never overwrite sensorData/fallDetected.
    # Firebase integration is enabled only with valid Admin SDK credentials.
    if os.environ.get('FIREBASE_PROJECT_ID') and os.environ.get('FIREBASE_DATABASE_URL'):
        publish_prediction(result)
    return result

_firebase_db = None

def publish_prediction(result):
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
    try:
        # Separate node; existing ESP32 sensorData PUTs cannot erase ML results.
        _firebase_db.reference('mlPrediction').set(result)
    except Exception as exc:
        raise HTTPException(status_code=503, detail='Firebase write failed') from exc
