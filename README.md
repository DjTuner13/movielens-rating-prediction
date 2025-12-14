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

## ☁️ AWS Cloud Workflow (Production)

We have fully automated the specific MLOps infrastructure using **Terraform**.

### 1. Deploy Infrastructure
Provision the **EC2 Server (t3.small)**, **S3 Buckets**, and **MLflow Service** with one command:
```bash
cd infra
# Uses terraform.tfvars for secrets (Neon DSN, Key Name)
terraform apply
```

### 2. Connect to the Server
Use the IP output from Terraform:
```bash
ssh -i ~/.ssh/mlflow-key.pem ubuntu@<PUBLIC_IP>
```

### 3. Run the Pipeline (Zero-Touch)
The server automatically installs Python, MLflow, and downloads the raw data from S3 on boot.
Just run the training:
```bash
# 1. Setup Environment (Once per session)
source $HOME/.local/bin/env
cd movielens-rating-prediction
uv sync

# 2. Run Data Pipeline (Downloads from S3 -> Process -> Uploads to S3)
uv run scripts/run_pipeline.py

# 3. Train Models (Logs to Neon & S3)
uv run python -m src.models.train_xgboost
```

### 4. Monitor Experiments
*   **MLflow UI**: [http://<PUBLIC_IP>:5000](http://<PUBLIC_IP>:5000)
    *   Tracks Parameters, Metrics (RMSE), and Artifacts (Models).
*   **Data Warehouse**: `s3://movielens-data-XXXX/processed/`
    *   Stores versioned datasets (`train.csv`, `test.csv`).

## Experiment Tracking (Local)
If running locally, you can still view the UI:
```bash
uv run mlflow ui
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000).
