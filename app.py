# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
from pathlib import Path
import numpy as np

app = FastAPI(title="ETA Delay Predictor", version="0.1.0")

# Train a toy model on first boot if file is missing (keeps it simple)
MODEL_PATH = Path("model.joblib")
if not MODEL_PATH.exists():
    from train import clf  # training saves model
    # (import side-effect from train.py writes model.joblib)

model = load(MODEL_PATH)

class Features(BaseModel):
    distance_km: float
    planned_stops: int
    traffic_score: float
    weather_score: float
    priority: int
    promised_hours: int

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(f: Features):
    try:
        X = np.array([[f.distance_km, f.planned_stops, f.traffic_score,
                       f.weather_score, f.priority, f.promised_hours]])
        prob_delay = float(model.predict_proba(X)[0][1])
        label = int(prob_delay >= 0.5)
        return {
            "delay_risk": "High" if label else "Low",
            "delay_prob": round(prob_delay, 4),
            "eta_hours": max(0.0, float(f.promised_hours) + (6 if label else 0)),
            "model_version": "v0.1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
