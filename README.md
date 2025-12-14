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

## ☁️ AWS Production Workflow
This workflow assumes you have provisioned the infrastructure via Terraform.

### 1. Connect & Initial Setup (Day 2 Checklist)
When starting a **fresh EC2 instance**, follow these steps to restore service:

1.  **SSH into the Server** (Use the IP from `terraform output`):
    ```bash
    ssh -i ~/.ssh/mlflow-key.pem ubuntu@<PUBLIC_IP>
    ```
2.  **Sync Code**:
    ```bash
    cd movielens-rating-prediction
    # Ensure latest code (especially if you just replaced the instance)
    git pull
    # Sync dependencies
    /home/ubuntu/.local/bin/uv sync
    ```

### 2. Run MLOps Pipeline
Execute these scripts on the server to update models.

**A. Train Models (Logs to Neon DB & S3)**
```bash
# Example: Train XGBoost (Tag: XGBoost)
uv run python -m src.models.train_xgboost
```

**B. Compare & Select Champion**
Automatically finds the best model from `movielens_cloud` experiment.
```bash
uv run scripts/post_training/compare_models.py
```

**C. Register to Model Registry**
Tags the best models so the API can find them.
```bash
uv run scripts/post_training/register_model.py
```

### 3. Serve API (FastAPI)
The API starts **automatically** on boot (via `systemd`).

**Check Status & Logs:**
```bash
# Check if running
systemctl status fastapi

# View real-time logs
journalctl -u fastapi -f
```

*(Manual Start - only if needed):*
```bash
nohup uv run uvicorn src.app.main:app --host 0.0.0.0 --port 8000 > api.log 2>&1 &
```

### 4. Accessing Services
Once running, access the services via your instance's **Public IP**:

| Service | URL | Description |
| :--- | :--- | :--- |
| **MLflow UI** | `http://<Public-IP>:5000` | View Experiments, Metrics, and Model Registry. |
| **API Docs** | `http://<Public-IP>:8000/docs` | Interactive Swagger UI to test predictions. |

#### 🔍 How to use Swagger UI
1.  Go to the **API Docs** URL.
2.  Click **POST /predict/{label}** -> **Try it out**.
3.  **Important**: In the `label` field, type one of the model tags:
    *   `XGBoost` (Recommended)
    *   `GLM`
    *   `RandomForest`
4.  Click **Execute**.

