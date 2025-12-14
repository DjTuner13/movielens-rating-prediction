import xgboost as xgb
import pandas as pd
import numpy as np
import os
import logging
import mlflow
from sklearn.metrics import mean_squared_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mlflow.set_experiment("movielens_experiment")

def load_data(path):
    """Load data and prepare for XGBoost (Label Encode Categoricals)."""
    df = pd.read_csv(path)
    
    # Features to use directly
    feature_cols = ['userId', 'movieId', 'gender', 'age', 'occupation'] + [c for c in df.columns if c.startswith('genre_')]
    target_col = 'rating'
    
    # We will map categorical string columns to codes
    for col in ['gender']:
        df[col] = df[col].astype('category').cat.codes
        
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y

def train_model(train_path, valid_path, n_estimators=100, max_depth=6, learning_rate=0.1, output_dir="models/xgboost"):
    logger.info("Initializing XGBoost training...")
    
    with mlflow.start_run(run_name="XGBoost"):
        # Log Params
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "model_type": "XGBoost"
        }
        mlflow.log_params(params)
        
        # 1. Load Data
        X_train, y_train = load_data(train_path)
        X_valid, y_valid = load_data(valid_path)
        
        logger.info(f"Train Shape: {X_train.shape}, Valid Shape: {X_valid.shape}")
        
        # 2. Train Model
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1,
            enable_categorical=True 
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            verbose=False
        )
        
        # 3. Evaluate
        preds_valid = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
        logger.info(f"Validation RMSE: {rmse:.4f}")
        mlflow.log_metric("valid_rmse", rmse)
        
        # 4. Save
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.json")
        model.get_booster().save_model(model_path)
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
