# app.py
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from joblib import load
import math

API_KEY = None  # set via env in Render if you want auth (optional)
try:
    import os
    API_KEY = os.getenv("API_KEY")
except Exception:
    pass

model = load("model.joblib")  # scikit model

app = FastAPI(title="ETA Delay Predictor", version="1.0.0")

class PredictIn(BaseModel):
    distance_km: float
    planned_stops: int
    traffic_score: float
    weather_score: float
    priority: int      # 0/1
    promised_hours: float

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict")
def predict(p: PredictIn, x_api_key: str | None = Header(default=None)):
    # Optional API-key check
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    features = [[
        float(p.distance_km),
        int(p.planned_stops),
        float(p.traffic_score),
        float(p.weather_score),
        int(p.priority),
        float(p.promised_hours),
    ]]

    # binary delay classification
    prob_delay = float(model.predict_proba(features)[0][1])
    is_delay = int(prob_delay >= 0.5)

    # toy ETA: promised ± adjustment from traffic/weather/priority
    adj = (p.traffic_score*4 + p.weather_score*3 + (1.5 if p.priority==1 else 0))
    eta_hours = max(1.0, float(p.promised_hours + (2 if is_delay else -1) + adj/2))

    return {
        "delay_risk": "High" if is_delay else "Low",
        "delay_prob": round(prob_delay, 4),
        "eta_hours": round(eta_hours, 2),
        "model_version": "rf-1.0.0"
    }
