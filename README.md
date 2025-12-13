# MovieLens Rating Prediction Project

## Project Overview
This project aims to build a machine learning pipeline to predict movie ratings (1-5 stars) using the MovieLens 1M dataset. The goal is to compare AutoML approaches against manually implemented algorithms.

## Objectives
1.  **Data Preparation**: Clean data, handle "Cold Start" scenarios, and implement a Time-Based Split.
2.  **AutoML Analysis**: Use **H2O AutoML** to identify top-performing model types.
3.  **Manual Implementation**: Build standalone training scripts for the top 3 algorithms identified by H2O.
5.  **Experiment Tracking**: Use **MLflow** to log parameters, metrics, and models.

