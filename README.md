# MovieLens Rating Prediction Project

## Project Overview
This project aims to build a machine learning pipeline to predict movie ratings (1-5 stars) using the MovieLens 1M dataset. The goal is to compare AutoML approaches against manually implemented algorithms.

## Objectives
1.  **Data Preparation**: Clean data, handle "Cold Start" scenarios, and implement a Time-Based Split.
2.  **AutoML Analysis**: Use **H2O AutoML** to identify top-performing model types.
3.  **Manual Implementation**: Build standalone training scripts for the top 3 algorithms identified by H2O.

## Installation
1.  **Install uv** (if not already installed):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
2.  **Sync Dependencies**:
    ```bash
    uv sync
    ```

## Setup & Data Pipeline

### 1. Prepare Raw Data
Place your raw dataset file (`merged_movielens.csv`) in the following directory:
`data/rawData/merged_movielens.csv`

### 2. Run the Pipeline
Execute the orchestration script to process the data:
```bash
uv run python scripts/run_pipeline.py
```

**What this does:**
1.  **Cleaning**: Handles missing values and correct data types.
2.  **Feature Engineering**: One-hot encodes genres and selects relevant features.
3.  **Splitting**: Performs a **Time-Based Split** to create `train.csv`, `validate.csv`, and `test.csv` in `data/processed/`.

## Model Training
We implemented 4 models based on H2O's leaderboard suggestions:

### A. AutoML (H2O)
```bash
# Model 1: H2O AutoML
uv run python -m src.models.train_h2o
```

### B. Manual Models (Scikit-Learn / XGBoost)
```bash
# Model 2: XGBoost (Gradient Boosting)
uv run python -m src.models.train_xgboost

# Model 3: Random Forest (DRF)
uv run python -m src.models.train_drf

# Model 4: ElasticNet (GLM)
uv run python -m src.models.train_glm
```

## Experiment Tracking
All runs are logged to **MLflow**. To view the dashboard:
```bash
uv run mlflow ui
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to compare RMSE metrics.
