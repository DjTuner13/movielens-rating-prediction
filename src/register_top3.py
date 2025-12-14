import tempfile
import mlflow
import joblib
import xgboost as xgb
import pandas as pd
from mlflow.tracking import MlflowClient

TRACKING_URI = "http://44.192.74.248:5000"
EXPERIMENT_NAME = "movielens_experiment"
MODEL_NAME = "movielens_top3"   # One model name, 3 versions

TEST_CSV = "/home/ubuntu/movielens-rating-prediction/data/processed/test.csv"
TARGET = "rating"

RUNS = [
    ("GLM", "eaa1a5db2a4a4447814208fb887d234f"),
    ("RandomForest", "7a098ccd6a7e486dafb3abe91c6c041c"),
    ("XGBoost", "b6d407bc19f246e8911ff6f818db0a8d"),
]

def encode_gender_if_needed(df):
    if "gender" in df.columns and df["gender"].dtype == "object":
        df = df.copy()
        df["gender"] = df["gender"].map({"F": 0, "M": 1})
    return df

class GLMPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.scaler = joblib.load(context.artifacts["scaler"])
        self.cols = list(getattr(self.scaler, "feature_names_in_", []))

    def predict(self, context, model_input):
        X = model_input.copy()
        if "rating_datetime" in X.columns:
            X = X.drop(columns=["rating_datetime"])
        X = encode_gender_if_needed(X)
        if self.cols:
            X = X[self.cols].copy()
        X.loc[:, X.columns] = self.scaler.transform(X)
        return self.model.predict(X)

class JoblibPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])
        self.cols = list(getattr(self.model, "feature_names_in_", []))

    def predict(self, context, model_input):
        X = model_input.copy()
        if "rating_datetime" in X.columns:
            X = X.drop(columns=["rating_datetime"])
        X = encode_gender_if_needed(X)
        if self.cols:
            X = X[self.cols].copy()
        return self.model.predict(X)

class XGBoostPyFunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.booster = xgb.Booster()
        self.booster.load_model(context.artifacts["model"])
        self.cols = self.booster.feature_names or []

    def predict(self, context, model_input):
        X = model_input.copy()
        if "rating_datetime" in X.columns:
            X = X.drop(columns=["rating_datetime"])
        X = encode_gender_if_needed(X)
        if self.cols:
            X = X[self.cols].copy()
        return self.booster.predict(xgb.DMatrix(X))

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # small input example for logging
    df = pd.read_csv(TEST_CSV).drop(columns=[TARGET])
    if "rating_datetime" in df.columns:
        df = df.drop(columns=["rating_datetime"])
    df = encode_gender_if_needed(df)
    input_example = df.head(5)

    with mlflow.start_run(run_name="register_top3_models") as run:
        for label, src_run in RUNS:
            with tempfile.TemporaryDirectory() as td:
                if label == "GLM":
                    m = client.download_artifacts(src_run, "model.joblib", td)
                    s = client.download_artifacts(src_run, "scaler.joblib", td)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{label.lower()}",
                        python_model=GLMPyFunc(),
                        artifacts={"model": m, "scaler": s},
                        input_example=input_example,
                    )
                    model_uri = f"runs:/{run.info.run_id}/model_{label.lower()}"

                elif label == "RandomForest":
                    m = client.download_artifacts(src_run, "model.joblib", td)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{label.lower()}",
                        python_model=JoblibPyFunc(),
                        artifacts={"model": m},
                        input_example=input_example,
                    )
                    model_uri = f"runs:/{run.info.run_id}/model_{label.lower()}"

                elif label == "XGBoost":
                    m = client.download_artifacts(src_run, "model.json", td)
                    mlflow.pyfunc.log_model(
                        artifact_path=f"model_{label.lower()}",
                        python_model=XGBoostPyFunc(),
                        artifacts={"model": m},
                        input_example=input_example,
                    )
                    model_uri = f"runs:/{run.info.run_id}/model_{label.lower()}"

                else:
                    raise ValueError(label)

            mv = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
            client.set_model_version_tag(MODEL_NAME, mv.version, "source_run_id", src_run)
            client.set_model_version_tag(MODEL_NAME, mv.version, "model_label", label)
            print(f"Registered {label} as {MODEL_NAME} v{mv.version}")

if __name__ == "__main__":
    main()
