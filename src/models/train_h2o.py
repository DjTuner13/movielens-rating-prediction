import h2o
from h2o.automl import H2OAutoML
import os
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class H2OWrapper:
    def __init__(self, max_runtime_secs=600, seed=42):
        """
        Initialize the H2O AutoML wrapper.
        """
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.aml = None
        self.train_frame = None
        self.valid_frame = None
        
    def start_h2o(self):
        """Initialize the H2O cluster."""
        h2o.init()
        logger.info("H2O initialized.")

    def load_data(self, train_path, valid_path):
        """
        Load data from CSV files into H2O Frames.
        """
        logger.info(f"Loading training data from {train_path}")
        self.train_frame = h2o.import_file(train_path)
        
        logger.info(f"Loading validation data from {valid_path}")
        self.valid_frame = h2o.import_file(valid_path)
        
        # Identify columns
        self.features = [c for c in self.train_frame.columns if c not in ['rating', 'rating_datetime']]
        self.target = 'rating'
        
        # Convert categoricals
        cat_cols = ['userId', 'movieId', 'gender', 'occupation', 'zip']
        for col in cat_cols:
            if col in self.features:
                self.train_frame[col] = self.train_frame[col].asfactor()
                self.valid_frame[col] = self.valid_frame[col].asfactor()

    def train(self):
        """Run the H2O AutoML training process."""
        if not self.train_frame:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        logger.info(f"Starting AutoML training for {self.max_runtime_secs} seconds...")
        
        self.aml = H2OAutoML(
            max_runtime_secs=self.max_runtime_secs,
            seed=self.seed,
            project_name="movielens_experiment",
            sort_metric="RMSE"
        )
        
        self.aml.train(
            x=self.features,
            y=self.target,
            training_frame=self.train_frame,
            validation_frame=self.valid_frame
        )
        logger.info("AutoML training complete.")
        
        # Log Metrics
        lb = self.aml.leaderboard
        lb_df = lb.as_data_frame()
        best_rmse = lb_df.iloc[0]['rmse']
        logger.info(f"Best Model RMSE: {best_rmse}")
        
        # Save artifacts
        self.save_best_model()

    def save_best_model(self, output_dir="models/h2o"):
        """Save the best model and leaderboard."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Leaderboard
        lb = self.aml.leaderboard
        lb_path = os.path.join(output_dir, "leaderboard.csv")
        h2o.export_file(lb, path=lb_path, force=True)
        
        # Save Model
        best_model = self.aml.leader
        model_path = h2o.save_model(model=best_model, path=output_dir, force=True)
        logger.info(f"Best model saved to {model_path}")
        
        return lb, model_path

if __name__ == "__main__":
    # Example usage
    wrapper = H2OWrapper(max_runtime_secs=60) 
    wrapper.start_h2o()
    
    if os.path.exists("data/processed/train.csv"):
        wrapper.load_data(
            train_path="data/processed/train.csv",
            valid_path="data/processed/validate.csv"
        )
        wrapper.train()
    else:
        logger.warning("Data not found.")
