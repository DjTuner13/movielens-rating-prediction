# MovieLens Rating Prediction Project

## Project Overview
This project aims to build a machine learning pipeline to predict movie ratings (1-5 stars) using the MovieLens 1M dataset. The goal is to compare AutoML approaches against manually implemented algorithms.

## Objectives
1.  **Data Preparation**: Clean data, handle "Cold Start" scenarios, and implement a Time-Based Split.
2.  **AutoML Analysis**: Use **H2O AutoML** to identify top-performing model types.
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

