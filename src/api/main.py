from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import mlflow

# -----------------------------
# CONFIG
# -----------------------------
TRACKING_URI = "http://44.192.74.248:5000"
MODEL_NAME = "movielens_top3"

# Registry versions (matches what you registered)
MODEL1_VERSION = "1"  # GLM
MODEL2_VERSION = "2"  # RandomForest
MODEL3_VERSION = "3"  # XGBoost

MODEL_LABELS = {"1": "GLM", "2": "RandomForest", "3": "XGBoost"}

# Expected feature columns (from your processed dataset)
GENRE_COLS = [
    "genre_Action", "genre_Adventure", "genre_Animation", "genre_Children's",
    "genre_Comedy", "genre_Crime", "genre_Documentary", "genre_Drama",
    "genre_Fantasy", "genre_Film-Noir", "genre_Horror", "genre_Musical",
    "genre_Mystery", "genre_Romance", "genre_Sci-Fi", "genre_Thriller",
    "genre_War", "genre_Western",
]
BASE_COLS = ["userId", "movieId", "gender", "age", "occupation"]
FEATURE_COLS = BASE_COLS + GENRE_COLS


# -----------------------------
# Pydantic Schemas
# -----------------------------
class Record(BaseModel):
    userId: int = Field(..., ge=1)
    movieId: int = Field(..., ge=1)
    gender: str = Field(..., description="M or F")
    age: int = Field(..., ge=0)
    occupation: int = Field(..., ge=0)

    # Optional genre one-hots (default 0 if omitted)
    genre_Action: Optional[int] = 0
    genre_Adventure: Optional[int] = 0
    genre_Animation: Optional[int] = 0
    genre_Children_s: Optional[int] = Field(0, alias="genre_Children's")
    genre_Comedy: Optional[int] = 0
    genre_Crime: Optional[int] = 0
    genre_Documentary: Optional[int] = 0
    genre_Drama: Optional[int] = 0
    genre_Fantasy: Optional[int] = 0
    genre_Film_Noir: Optional[int] = Field(0, alias="genre_Film-Noir")
    genre_Horror: Optional[int] = 0
    genre_Musical: Optional[int] = 0
    genre_Mystery: Optional[int] = 0
    genre_Romance: Optional[int] = 0
    genre_Sci_Fi: Optional[int] = Field(0, alias="genre_Sci-Fi")
    genre_Thriller: Optional[int] = 0
    genre_War: Optional[int] = 0
    genre_Western: Optional[int] = 0

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in {"M", "F"}:
            raise ValueError("gender must be 'M' or 'F'")
        return v

    class Config:
        populate_by_name = True  # allow aliases like "genre_Children's"


class PredictRequest(BaseModel):
    records: Union[Record, List[Record]]


class PredictResponse(BaseModel):
    model_name: str
    model_version: str
    model_label: str
    timestamp_utc: str
    n_records: int
    predictions: List[float]


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="MovieLens Rating Predictor (MLflow Registry)")

_models: Dict[str, Any] = {}


def normalize_to_df(payload: PredictRequest) -> pd.DataFrame:
    recs = payload.records if isinstance(payload.records, list) else [payload.records]
    rows = [r.model_dump(by_alias=True) for r in recs]
    df = pd.DataFrame(rows)

    # Ensure all expected columns exist
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0

    # Drop extras and enforce ordering
    df = df[FEATURE_COLS].copy()

    # Encode gender to numeric: F->0, M->1
    df["gender"] = df["gender"].map({"F": 0, "M": 1}).astype("int64")

    # Coerce all features to numeric
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="raise")

    return df


def do_predict(version: str, X: pd.DataFrame) -> List[float]:
    if version not in _models:
        raise HTTPException(status_code=500, detail=f"Model version {version} not loaded")
    try:
        preds = _models[version].predict(X)
        return [float(p) for p in list(preds)]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")


@app.on_event("startup")
def load_models() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)

    for v in [MODEL1_VERSION, MODEL2_VERSION, MODEL3_VERSION]:
        uri = f"models:/{MODEL_NAME}/{v}"
        _models[v] = mlflow.pyfunc.load_model(uri)

    print(f"[startup] loaded models: {MODEL_NAME} v1/v2/v3")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "tracking_uri": TRACKING_URI,
        "model_name": MODEL_NAME,
        "loaded_versions": sorted(list(_models.keys())),
    }


@app.post("/predict_model1", response_model=PredictResponse)
def predict_model1(req: PredictRequest) -> PredictResponse:
    X = normalize_to_df(req)
    preds = do_predict(MODEL1_VERSION, X)
    return PredictResponse(
        model_name=MODEL_NAME,
        model_version=MODEL1_VERSION,
        model_label=MODEL_LABELS[MODEL1_VERSION],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(preds),
        predictions=preds,
    )


@app.post("/predict_model2", response_model=PredictResponse)
def predict_model2(req: PredictRequest) -> PredictResponse:
    X = normalize_to_df(req)
    preds = do_predict(MODEL2_VERSION, X)
    return PredictResponse(
        model_name=MODEL_NAME,
        model_version=MODEL2_VERSION,
        model_label=MODEL_LABELS[MODEL2_VERSION],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(preds),
        predictions=preds,
    )


@app.post("/predict_model3", response_model=PredictResponse)
def predict_model3(req: PredictRequest) -> PredictResponse:
    X = normalize_to_df(req)
    preds = do_predict(MODEL3_VERSION, X)
    return PredictResponse(
        model_name=MODEL_NAME,
        model_version=MODEL3_VERSION,
        model_label=MODEL_LABELS[MODEL3_VERSION],
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        n_records=len(preds),
        predictions=preds,
    )
