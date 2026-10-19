"""Exact feature definition imported from the provided SisFall training script."""
import joblib
import numpy as np
from train_fall_model import window_features

class FallPredictor:
    def __init__(self, path='fall_rf_sisfall.joblib'):
        saved = joblib.load(path)  # Load only trusted model files.
        self.model = saved['model']
        self.features = saved['features']
        self.fs = saved['fs']
        self.win = saved['win']
        self.threshold = saved['threshold']
        assert self.fs == 50 and self.win == 150
        assert len(self.features) == self.model.n_features_in_

    def predict(self, samples):
        if len(samples) != self.win:
            raise ValueError('Exactly 150 samples are required')
        x = np.asarray([[row['ax'], row['ay'], row['az'], row['gx'], row['gy'], row['gz']] for row in samples], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('All sensor values must be finite')
        # MPU6050 readings: accelerometer in g, gyroscope in deg/s.
        if np.max(np.abs(x[:, :3])) > 12 or np.max(np.abs(x[:, 3:])) > 650:
            raise ValueError('Sensor readings out of expected range')
        acc = np.linalg.norm(x[:, :3], axis=1).astype(np.float32)[None, :]
        gyro = np.linalg.norm(np.clip(x[:, 3:], -500, 500), axis=1).astype(np.float32)[None, :]
        features = window_features(acc, gyro).reindex(columns=self.features)
        prob = float(self.model.predict_proba(features)[0, list(self.model.classes_).index(1)])
        return {'fallDetected': bool(prob >= self.threshold), 'probability': round(prob, 6), 'threshold': self.threshold}
