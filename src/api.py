"""
api.py
------
FastAPI service exposing the battery fault classifier.

Run with:
    uvicorn src.api:app --reload

Endpoints:
    GET  /health           - liveness check
    POST /predict          - accepts a short window of recent readings,
                              returns the fault classification for the
                              most recent one.

We require a *window* of readings (not a single point) because the
feature engineering layer relies on rolling statistics and deltas —
a single reading alone can't tell you "rate of change". The API
contract documents this in the request schema so it's clear why the
minimum window size exists (see docs/SRS.md, FR-3).
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_feature_matrix
from src.model import FEATURE_COLUMNS, BatteryFaultClassifier
from src.validation import Reading as ValidationReading
from src.validation import validate_batch

MODEL_PATH = "model.joblib"
MIN_WINDOW_SIZE = 5

app = FastAPI(
    title="Battery Fault Detection API",
    description="Classifies Li-ion battery telemetry into normal or fault states.",
    version="1.0.0",
)

_classifier = BatteryFaultClassifier()


class TelemetryReading(BaseModel):
    voltage: float = Field(..., description="Cell voltage in volts")
    current: float = Field(..., description="Current in amps (negative = discharge)")
    temperature: float = Field(..., description="Temperature in degrees Celsius")
    timestamp: float | None = Field(default=None, description="Optional epoch seconds")


class PredictRequest(BaseModel):
    readings: list[TelemetryReading] = Field(
        ..., description=f"Chronological window of at least {MIN_WINDOW_SIZE} readings"
    )


class PredictResponse(BaseModel):
    fault_type: str
    rejected_readings: int
    used_window_size: int


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Loads a pre-trained model if present; otherwise leaves it untrained
    and /predict will return a clear error instead of a confusing crash."""
    if Path(MODEL_PATH).exists():
        _classifier.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _classifier._is_trained}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    if not _classifier._is_trained:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and save a model to model.joblib first "
                   "(see src/model.py __main__ block).",
        )

    if len(request.readings) < MIN_WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {MIN_WINDOW_SIZE} readings to compute rolling features, "
                   f"got {len(request.readings)}.",
        )

    validation_readings = [
        ValidationReading(voltage=r.voltage, current=r.current, temperature=r.temperature)
        for r in request.readings
    ]
    valid, rejected = validate_batch(validation_readings)

    if len(valid) < MIN_WINDOW_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"Only {len(valid)} valid readings after validation "
                   f"({len(rejected)} rejected); need at least {MIN_WINDOW_SIZE}.",
        )

    df = pd.DataFrame([{"voltage": r.voltage, "current": r.current, "temperature": r.temperature} for r in valid])
    featured = build_feature_matrix(df)
    latest_row = featured.iloc[[-1]][FEATURE_COLUMNS]

    prediction = _classifier.model.predict(latest_row)[0]

    return PredictResponse(
        fault_type=str(prediction),
        rejected_readings=len(rejected),
        used_window_size=len(valid),
    )
