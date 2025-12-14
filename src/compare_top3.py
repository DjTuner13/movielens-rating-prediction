import math
import tempfile
import pandas as pd
import joblib
import xgboost as xgb
import mlflow

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TRACKING_URI = "http://44.192.74.248:5000"
TEST_CSV = "/home/ubuntu/movielens-rating-prediction/data/processed/test.csv"
TARGET = "rating"

RUNS = [
    ("GLM", "eaa1a5db2a4a4447814208fb887d234f"),
    ("RandomForest", "7a098ccd6a7e486dafb3abe91c6c041c"),
    ("XGBoost", "b6d407bc19f246e8911ff6f818db0a8d"),
]

def RMSE(y, p): return math.sqrt(mean_squared_error(y, p))

def encode_gender_if_needed(df):
    if "gender" in df.columns and df["gender"].dtype == "object":
        df = df.copy()
        df["gender"] = df["gender"].map({"F": 0, "M": 1})
    return df

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    df = pd.read_csv(TEST_CSV)
    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    # Drop datetime column if present (models typically trained on numeric features)
    if "rating_datetime" in X.columns:
        X = X.drop(columns=["rating_datetime"])

    X = encode_gender_if_needed(X)

    rows = []

    for name, run_id in RUNS:
        with tempfile.TemporaryDirectory() as td:
            if name == "GLM":
                model_path = client.download_artifacts(run_id, "model.joblib", td)
                scaler_path = client.download_artifacts(run_id, "scaler.joblib", td)
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)

                # Use exactly scaler-fit columns if available
                if hasattr(scaler, "feature_names_in_"):
                    cols = list(scaler.feature_names_in_)
                    X_use = X[cols].copy()
                else:
                    X_use = X.copy()

                X_use = encode_gender_if_needed(X_use)
                X_use.loc[:, X_use.columns] = scaler.transform(X_use)

                pred = model.predict(X_use)

            elif name == "RandomForest":
                model_path = client.download_artifacts(run_id, "model.joblib", td)
                model = joblib.load(model_path)
                if hasattr(model, "feature_names_in_"):
                    X_use = X[list(model.feature_names_in_)].copy()
                else:
                    X_use = X.copy()
                X_use = encode_gender_if_needed(X_use)
                pred = model.predict(X_use)

            elif name == "XGBoost":
                model_path = client.download_artifacts(run_id, "model.json", td)
                booster = xgb.Booster()
                booster.load_model(model_path)
                feats = booster.feature_names
                X_use = X[feats].copy() if feats else X.copy()
                X_use = encode_gender_if_needed(X_use)
                pred = booster.predict(xgb.DMatrix(X_use))

            else:
                raise ValueError(name)

        rows.append({
            "Model": name,
            "RunID": run_id,
            "RMSE": RMSE(y, pred),
            "MAE": mean_absolute_error(y, pred),
            "R2": r2_score(y, pred),
        })

    out = pd.DataFrame(rows).sort_values(["RMSE", "MAE", "R2"], ascending=[True, True, False])
    print("\n=== COMPARISON TABLE (Test Set) ===")
    print(out.to_string(index=False))

    champ = out.iloc[0]
    print(f"\nCHAMPION = {champ['Model']} (lowest RMSE)")

    out.to_csv("/home/ubuntu/movielens-rating-prediction/comparison_table.csv", index=False)
    print("\nSaved: comparison_table.csv")

if __name__ == "__main__":
    main()
