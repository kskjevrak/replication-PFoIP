#!/usr/bin/env python
"""
xgb_pred.py - Class file for running XGBoost quantile predictions for a specified date range
Compatible with cluster execution

"""
import os
import sys
import torch
import logging
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta
from functools import partial
import json
import joblib
import yaml
from pathlib import Path

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


from src.models.xgboost_quantile import XGBoostQuantileModel
from src.data.loader import DataProcessor

class XGBoostDailyPredictor:
    """
    Class for making daily XGBoost quantile predictions to support market bidding
    """
    
    def __init__(self, zone='no1', run_id='1'):
        """
        Initialize the XGBoost predictor
        
        Parameters:
        -----------
        zone : str
            Market zone identifier (e.g., 'no1', 'no2', etc.)
        run_id : str
            Identifier for model run
        """
        self.zone = zone.lower()
        self.run_id = run_id
        
        # Set up directories and paths
        self.root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        self.config_path = self.root_dir / 'config' / 'default_config.yml'
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up paths for models, scalers, and results
        self.results_dir = self.root_dir / 'results'
        self.model_dir = self.results_dir / 'models' / f"{self.zone}_xgboost_{self.run_id}" / 'best_model'
        self.scaler_path = self.results_dir / 'scalers' / f"{self.zone}_xgboost_{self.run_id}"
        self.params_dir = self.results_dir / 'forecasts' / f"distparams_{self.zone}_xgboost_{self.run_id}"
        self.df_forecasts_dir = self.results_dir / 'forecasts' / f"df_forecasts_{self.zone}_xgboost_{self.run_id}"
        
        # Create output directories
        os.makedirs(self.params_dir, exist_ok=True)
        os.makedirs(self.df_forecasts_dir, exist_ok=True)
        
        # Set up logging
        self.logs_dir = self.results_dir / 'logs'
        os.makedirs(self.logs_dir, exist_ok=True)
        self._setup_logging()
        
        # Load best parameters and model
        self._load_best_parameters()
        self._load_model()
        
        # Set data parameters
        self.window_size = self.config.get('prediction', {}).get('window_size', 730)
        self.prediction_hours = 24
        
        # Initialize data processor
        self.data_processor = DataProcessor(zone=self.zone, config=self.config)
        
    def _setup_logging(self):
        """Configure logging"""
        log_file = self.logs_dir / f"xgboost_prediction_{self.zone}_{self.run_id}_{os.getpid()}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"xgboost_prediction_{self.zone}_{os.getpid()}")
        
    def _load_best_parameters(self):
        """Load best parameters from tuning"""
        best_params_file = self.model_dir.parent / 'best_params.yaml'
        
        if not best_params_file.exists():
            raise FileNotFoundError(f"Best parameters file not found: {best_params_file}")
            
        with open(best_params_file, 'r') as f:
            self.best_params = yaml.safe_load(f)
            
        self.logger.info(f"Loaded best parameters from {best_params_file}")
        
    def _load_model(self):
        """Load the trained XGBoost model"""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
            
        # Initialize XGBoost model
        self.model = XGBoostQuantileModel()
        self.model.load_model(str(self.model_dir))
        
        self.logger.info(f"Loaded XGBoost model from {self.model_dir}")
        
        # Load feature groups for consistency check
        import pickle
        preproc_dir = f'./src/data/preprocessing/{self.zone}_xgboost_{self.run_id}'
        with open(f'{preproc_dir}/feature_groups.pkl', 'rb') as f:
            self.feature_groups = pickle.load(f)
    
    def load_data(self, prediction_date):
        """Load the latest market data for prediction"""
        self.logger.info(f"Loading data for zone {self.zone}, date {prediction_date}...")
        
        # Load data file
        data_dir = self.root_dir / 'src' / 'data' / f"{self.zone}"
        data_file = data_dir / f"merged_dataset_{self.zone}.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file {data_file} not found")
            
        # Load data
        data = pd.read_parquet(data_file)
        data.index = pd.to_datetime(data.index, utc=True).tz_convert('Europe/Oslo')
        
        # Create features using DataProcessor
        data = self.data_processor._create_lagged_features(data)
        
        # Extract the training window
        start_date = prediction_date - timedelta(days=self.window_size)
        window_data = data[(data.index >= start_date) & (data.index < prediction_date + timedelta(days=1))]
        
        self.data = window_data
        self.logger.info(f"Loaded {len(self.data)} records")
        return True
        
    def prepare_features(self):
        """Prepare feature matrices following DDNN approach"""
        self.logger.info("Preparing features...")
        
        forecast_horizon = self.prediction_hours
        days = len(self.data) // forecast_horizon
        
        # Trim data to complete days
        self.data = self.data.iloc[:-(len(self.data) % forecast_horizon)] if len(self.data) % forecast_horizon != 0 else self.data
        
        # Separate premium (target) and features
        premium_data = self.data['premium'].values
        features = self.data.drop(columns=['premium'])
        
        # Prepare arrays
        Y = np.zeros((days, forecast_horizon))
        X = np.zeros((days, forecast_horizon * len(features.columns)))
        
        # Fill arrays
        for d in range(days):
            start_idx = d * forecast_horizon
            end_idx = start_idx + forecast_horizon
            Y[d, :] = premium_data[start_idx:end_idx]
            for i, feature in enumerate(features.columns):
                X[d, i * forecast_horizon:(i + 1) * forecast_horizon] = features.iloc[start_idx:end_idx, i].values
        
        # Store the last day for prediction
        self.X_forecast = X[-1:, :]
        self.Y_forecast = Y[-1:, :]  # For logging/comparison if needed
        
        self.logger.info(f"Feature matrix prepared: X_forecast shape {self.X_forecast.shape}")
        return True
        
    def apply_scaling_and_selection(self):
        """Apply scaling and feature selection"""
        self.logger.info("Applying scaling and feature selection...")
        
        # Load scalers
        scaler_X = {}
        for feature_file in self.scaler_path.glob('scaler_X_*.joblib'):
            feature_name = feature_file.stem.replace('scaler_X_', '')
            scaler_X[feature_name] = joblib.load(feature_file)
            
        scaler_Y = joblib.load(self.scaler_path / 'scaler_Y.joblib')
        self.scaler_Y = scaler_Y  # Store for inverse transform
        
        # Define cyclical features
        cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                            'month_sin', 'month_cos', 'is_weekend']
        
        # Scale features
        X_forecast_scaled = np.zeros_like(self.X_forecast)
        
        for feature_name, feature_info in self.feature_groups.items():
            if feature_name in scaler_X:
                start_idx = feature_info['start_idx']
                size = feature_info['size']
                
                if feature_name in cyclical_features or scaler_X[feature_name] is None:
                    X_forecast_scaled[:, start_idx:start_idx + size] = self.X_forecast[:, start_idx:start_idx + size]
                else:
                    X_forecast_scaled[:, start_idx:start_idx + size] = scaler_X[feature_name].transform(
                        self.X_forecast[:, start_idx:start_idx + size]
                    )
        
        # Apply feature selection based on best parameters
        colmask = np.zeros(X_forecast_scaled.shape[1], dtype=bool)
        
        for feature_name, feature_info in self.feature_groups.items():
            # Check if feature was selected during tuning
            if feature_name in self.best_params and self.best_params[feature_name]:
                start_idx = feature_info['start_idx']
                size = feature_info['size']
                colmask[start_idx:start_idx + size] = True
        
        # Apply mask
        self.X_forecast_selected = X_forecast_scaled[:, colmask]
        
        self.logger.info(f"After feature selection: {np.sum(colmask)} features selected")
        return True
        
    def generate_forecast(self, prediction_date):
        """Generate quantile forecast for next 24 hours"""
        self.logger.info("Generating XGBoost forecast...")
        
        # Get quantile predictions
        quantile_preds = self.model.predict_quantiles(self.X_forecast_selected)  # shape: (1, 24, n_quantiles)
        
        # Get distribution parameters (for metrics.py compatibility)
        params = self.model.get_distribution_params(self.X_forecast_selected)
        
        # Save distribution parameters
        params_path = self.params_dir / f"{prediction_date.strftime('%Y-%m-%d')}.json"
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        # Generate samples for statistics calculation
        samples = self.model.predict_samples(self.X_forecast_selected, n_samples=10000)  # shape: (10000, 1, 24)
        samples = samples[:, 0, :]  # Remove single input dimension: (10000, 24)
        
        # Apply inverse transform
        predictions = self.scaler_Y.inverse_transform(samples)
        
        # Calculate forecast statistics (same as DDNN)
        forecast_df = pd.DataFrame(index=pd.date_range(
            start=prediction_date,
            periods=24,
            freq='H'
        ))
        
        # Extract state information
        threshold = 0.5
        up_prob = np.mean(predictions > threshold, axis=0)
        down_prob = np.mean(predictions < -threshold, axis=0)
        normal_prob = 1.0 - up_prob - down_prob
        
        states = np.zeros(24, dtype=int)
        states[up_prob > down_prob] = 1  # Up regulation
        states[down_prob > up_prob] = 2  # Down regulation
        
        state_names = ['normal', 'up_regulation', 'down_regulation']
        forecast_df['state'] = [state_names[s] for s in states]
        forecast_df['normal_prob'] = normal_prob
        forecast_df['up_prob'] = up_prob
        forecast_df['down_prob'] = down_prob
        
        # Calculate statistics
        forecast_df['mean'] = np.mean(predictions, axis=0)
        forecast_df['median'] = np.median(predictions, axis=0)
        forecast_df['std'] = np.std(predictions, axis=0)
        forecast_df['p10'] = np.percentile(predictions, 10, axis=0)
        forecast_df['p25'] = np.percentile(predictions, 25, axis=0)
        forecast_df['p75'] = np.percentile(predictions, 75, axis=0)
        forecast_df['p90'] = np.percentile(predictions, 90, axis=0)
        
        # Save forecast dataframe
        forecast_df.to_csv(self.df_forecasts_dir / f"{prediction_date.strftime('%Y-%m-%d')}.csv")
        
        self.logger.info(f"Forecast generated for {prediction_date.strftime('%Y-%m-%d')}")
        return True
        
    def run(self, prediction_date):
        """Run the complete prediction pipeline for a specific date"""
        try:
            self.prediction_date = prediction_date
            
            # Load and prepare data
            self.load_data(prediction_date)
            self.prepare_features()
            self.apply_scaling_and_selection()
            
            # Generate forecast
            self.generate_forecast(prediction_date)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in prediction pipeline: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
