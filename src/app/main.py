from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
import os
import threading
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
import mlflow
from mlflow.tracking import MlflowClient

# -----------------------------
# CONFIG
# -----------------------------
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "movielens_top3"

# Map internal labels (from register_model.py) to API keys
REQUIRED_LABELS = ["GLM", "RandomForest", "XGBoost"]

# Expected feature columns 
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
        populate_by_name = True


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

# Store loaded models: {"GLM": model_obj, "XGBoost": model_obj}
_models: Dict[str, Any] = {}
_versions: Dict[str, str] = {} # {"GLM": "1", ...}
_models_lock = threading.Lock()  # Thread-safe access to _models and _versions

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

@app.on_event("startup")
def load_models() -> None:
    print(f"[startup] Connecting to MLflow at {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Find latest version for each label
    # We query all versions of the registered model
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    
    # Organize by label
    # We want the LATEST version that has the tag 'model_label' == 'XGBoost', etc.
    # search_model_versions returns latest first usually, but let's sort to be sure.
    versions = sorted(versions, key=lambda x: int(x.version), reverse=True)

    loaded_labels = set()

    for v in versions:
        if not v.tags:
            continue
        label = v.tags.get("model_label")
        if label in REQUIRED_LABELS and label not in loaded_labels:
            print(f"[startup] Found {label} -> Version {v.version}")
            
            uri = f"models:/{MODEL_NAME}/{v.version}"
            try:
                model = mlflow.pyfunc.load_model(uri)
                with _models_lock:
                    _models[label] = model
                    _versions[label] = v.version
                loaded_labels.add(label)
                print(f"[startup] Loaded {label} successfully.")
            except Exception as e:
                print(f"[startup] Failed to load {label} (v{v.version}): {e}")

    if not loaded_labels:
        print("[startup] WARNING: No models loaded! Check MLflow Registry.")

@app.get("/health")
def health() -> Dict[str, Any]:
    with _models_lock:
        loaded_models = _versions.copy()
    return {
        "status": "ok",
        "tracking_uri": TRACKING_URI,
        "model_name": MODEL_NAME,
        "loaded_models": loaded_models,
    }

@app.post("/predict/{label}", response_model=PredictResponse)
def predict(label: str, req: PredictRequest) -> PredictResponse:
    with _models_lock:
        model_exists = label in _models
        if model_exists:
            model = _models[label]
            version = _versions[label]
    
    if not model_exists:
        raise HTTPException(status_code=404, detail=f"Model '{label}' not found or not loaded.")
    
    X = normalize_to_df(req)
    
    try:
        preds = model.predict(X)
        preds_list = [float(p) for p in list(preds)]
        
        return PredictResponse(
            model_name=MODEL_NAME,
            model_version=version,
            model_label=label,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            n_records=len(preds_list),
            predictions=preds_list,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
