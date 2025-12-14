import pandas as pd
import numpy as np
import os
import logging
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mlflow.set_experiment("movielens_experiment")

def load_data(path):
    """Load data and encode features for Random Forest."""
    df = pd.read_csv(path)
    
    feature_cols = ['userId', 'movieId', 'gender', 'age', 'occupation'] + [c for c in df.columns if c.startswith('genre_')]
    target_col = 'rating'
    
    for col in ['gender']:
        if df[col].dtype == 'object':
             df[col] = df[col].astype('category').cat.codes
             
    X = df[feature_cols]
    y = df[target_col]
    return X, y

def train_model(train_path, valid_path, n_estimators=50, max_depth=10, output_dir="models/drf"):
    logger.info("Initializing Random Forest (DRF) training...")
    
    with mlflow.start_run(run_name="RandomForest"):
        # Log Params
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "model_type": "DRF (Random Forest)"
        }
        mlflow.log_params(params)
        
        # 1. Load Data
        X_train, y_train = load_data(train_path)
        X_valid, y_valid = load_data(valid_path)
        
        logger.info(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape}")
        
        # 2. Train Model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # 3. Evaluate
        preds_valid = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
        logger.info(f"Validation RMSE: {rmse:.4f}")
        mlflow.log_metric("valid_rmse", rmse)
        
        # 4. Save
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.joblib")
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    if os.path.exists("data/processed/train.csv"):
        train_model(
            train_path="data/processed/train.csv",
            valid_path="data/processed/validate.csv"
        )
    else:
        logger.warning("Data not found.")
