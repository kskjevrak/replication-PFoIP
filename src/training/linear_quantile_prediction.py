#!/usr/bin/env python
"""
linear_quantile_prediction.py - Daily prediction script for mFRR market prices using Linear Quantile Regression

This script loads the best Linear Quantile model parameters from a previous optimization,
loads the latest data, and predicts mFRR prices for the next 24 hours.
It's designed to be a linear benchmark for the DDNN models.

Usage:
    python linear_quantile_prediction.py [--zone ZONE] [--run_id RUN_ID] [--config CONFIG]
"""
import os
import yaml
import traceback
import logging
import filelock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path

from ..models.linear_quantile import LinearQuantileModel
from ..data.loader import DataProcessor

class LinearQuantilePredictor:
    """
    Class for making daily mFRR price predictions using Linear Quantile Regression
    """
    
    def __init__(self, zone='no1', run_id=None, config_path=None):
        """
        Initialize the predictor with market zone and model configuration
        
        Parameters:
        -----------
        zone : str
            Market zone identifier (e.g., 'no1', 'no2', etc.)
        run_id : str
            Identifier for model run (if None, use the latest)
        config_path : str
            Path to configuration file
        """
        self.zone = zone.lower()
        
        # Set up directories and paths
        self.root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        self.config_path = config_path or self.root_dir / 'config' / 'default_config.yml'
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Determine run_id if not provided
        if run_id is None:
            run_id = self._get_latest_run_id()
        self.run_id = run_id
        
        # Set up paths for models, scalers, and results
        self.results_dir = self.root_dir / 'results'
        self.model_dir = self.results_dir / 'models' / f"{self.zone}_lqa_{self.run_id}"
        self.scaler_path = self.results_dir / 'scalers' / f"{self.zone}_lqa_{self.run_id}"
        self.df_forecasts_dir = self.results_dir / 'forecasts' / f"df_forecasts_{self.zone}_lqa_{self.run_id}"
        self.params_dir = self.results_dir / 'forecasts' / f"distparams_{self.zone}_lqa_{self.run_id}"
        
        # Create directories if they don't exist
        for directory in [self.df_forecasts_dir, self.params_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        self.logs_dir = self.results_dir / 'logs'
        os.makedirs(self.logs_dir, exist_ok=True)
        self._setup_logging()
        
        # Load best parameters and model
        self._load_best_parameters()
        
        # Set data parameters
        self.window_size = self.config.get('prediction', {}).get('window_size', 730)  # 2 years default
        self.prediction_hours = self.config.get('prediction', {}).get('hours', 24)
        
        # Initialize data processor
        self.data_processor = DataProcessor(zone=self.zone, config=self.config)
        
    def _setup_logging(self):
        """Configure logging"""
        log_file = self.logs_dir / f"prediction_{self.zone}_lqa_{self.run_id}_{os.getpid()}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"prediction_{self.zone}_lqa_{os.getpid()}")
        
    def _get_latest_run_id(self):
        """Get the latest run ID from available model directories"""
        models_dir = self.results_dir / 'models'
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory {models_dir} not found")
            
        # Find all directories matching the pattern
        pattern = f"linear_quantile_{self.zone}_*"
        matching_dirs = list(models_dir.glob(pattern))
        
        if not matching_dirs:
            raise FileNotFoundError(f"No Linear Quantile model directories found for {self.zone}")
            
        # Sort by modification time (newest first)
        latest_dir = max(matching_dirs, key=os.path.getmtime)
        
        # Extract run_id from directory name
        run_id = latest_dir.name.split('_')[-1]
        return run_id
        
    def _load_best_parameters(self):
        """Load best parameters from previous optimization"""
        try:
            # Load from YAML file
            best_params_file = self.model_dir / 'best_params.yaml'
            if best_params_file.exists():
                with open(best_params_file, 'r') as f:
                    self.best_params = yaml.safe_load(f)
                self.logger.info(f"Loaded best parameters from {best_params_file}")
            else:
                raise FileNotFoundError(f"Best parameters file not found: {best_params_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to load parameters: {str(e)}")
            raise
    
    def load_data(self):
        """Load the latest market data for prediction"""
        self.logger.info(f"Loading data for zone {self.zone}...")
        
        # Determine the data file path based on configuration
        data_dir = self.root_dir / 'src' / 'data' / f"{self.zone}"
        data_file = data_dir / f"merged_dataset_{self.zone}.parquet"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file {data_file} not found")
            
        # Load data
        data = pd.read_parquet(data_file)
        data.index = pd.to_datetime(data.index, utc=True).tz_convert('Europe/Oslo')
        
        # Create features using DataProcessor
        data = self.data_processor._create_lagged_features(data)

        # Extract the training window plus one day for forecasting
        start_date = self.prediction_date - timedelta(days=self.window_size)
        window_data = data[(data.index >= start_date) & (data.index < self.prediction_date + timedelta(days=1))]
        
        # Check if we have enough data
        if len(window_data) < self.window_size * 24:
            self.logger.warning(f"Insufficient data: expected {self.window_size * 24} hourly records, got {len(window_data)}")

        self.data = window_data
        self.logger.info(f"Loaded {len(self.data)} records from {start_date} to {self.prediction_date}")
        return True
        
    def prepare_features(self):
        """Prepare feature matrices for model training and prediction"""
        try:
            self.logger.info("Preparing features...")
            
            # Get forecast horizon
            forecast_horizon = self.prediction_hours  # 24 hours by default
            
            # Determine total days based on window size
            days = len(self.data) // forecast_horizon
            if len(self.data) % forecast_horizon != 0:
                # Trim data to complete days
                self.data = self.data.iloc[:-(len(self.data) % forecast_horizon)]
            
            # The last day will be used for forecasting
            training_days = days - 1
            self.logger.info(f"Using {training_days} days for training, 1 day for forecasting")
            
            # Separate premium (target) and features
            premium_data = self.data['premium'].values
            features = self.data.drop(columns=['premium'])
            
            # Prepare arrays for daily premium values
            Y = np.zeros((days, forecast_horizon))
            X = np.zeros((days, forecast_horizon * len(features.columns)))
            
            # Create feature group register
            self.feature_groups = {}
            for i, feature in enumerate(features.columns):
                self.feature_groups[feature] = {'start_idx': i * forecast_horizon, 'size': forecast_horizon}
            
            # Set input size for later use
            self.input_size = X.shape[1]
            
            # Fill X and Y with values for each day
            for d in range(days):
                start_idx = d * forecast_horizon
                end_idx = start_idx + forecast_horizon
                if end_idx <= len(premium_data):
                    Y[d, :] = premium_data[start_idx:end_idx]
                for i, feature in enumerate(features.columns):
                    X[d, i * forecast_horizon:(i + 1) * forecast_horizon] = features.iloc[start_idx:end_idx, i].values
            
            # Explicitly separate training data from forecast data
            self.X_train = X[:training_days]
            self.Y_train = Y[:training_days]
            self.X_forecast = X[training_days:]
            
            # Store values for later use
            self.X = X
            self.Y = Y
            
            self.logger.info(f"Feature matrices prepared: X_train shape {self.X_train.shape}, Y_train shape {self.Y_train.shape}, X_forecast shape {self.X_forecast.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    def apply_scaling_and_selection(self):
        """Apply scaling and feature selection to the prepared data"""
        self.logger.info("Applying scaling and feature selection...")
    
        # Load scalers
        scaler_X = {}
        for feature_file in self.scaler_path.glob('scaler_X_*.joblib'):
            feature_name = feature_file.stem.replace('scaler_X_', '')
            scaler_X[feature_name] = joblib.load(feature_file)
            
        scaler_Y = joblib.load(self.scaler_path / 'scaler_Y.joblib')
        
        # Define cyclical features that need special handling
        cyclical_features = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                            'month_sin', 'month_cos', 'is_weekend']
        
        # Scale each feature group separately
        X_train_scaled = np.zeros_like(self.X_train)
        X_forecast_scaled = np.zeros_like(self.X_forecast)
        
        # Apply scaling to each feature group
        for feature_name, scaler in scaler_X.items():
            if feature_name in self.feature_groups:
                start_idx = self.feature_groups[feature_name]['start_idx']
                size = self.feature_groups[feature_name]['size']
                
                if feature_name in cyclical_features or scaler is None:
                    X_train_scaled[:, start_idx:start_idx + size] = self.X_train[:, start_idx:start_idx + size]
                    X_forecast_scaled[:, start_idx:start_idx + size] = self.X_forecast[:, start_idx:start_idx + size]
                else:
                    # Scale training data
                    X_train_scaled[:, start_idx:start_idx + size] = scaler.transform(
                        self.X_train[:, start_idx:start_idx + size]
                    )
                
                    # Scale forecast data
                    X_forecast_scaled[:, start_idx:start_idx + size] = scaler.transform(
                        self.X_forecast[:, start_idx:start_idx + size]
                    )
        
        # Scale target data
        Y_train_scaled = scaler_Y.transform(self.Y_train)
        
       # Reconstruct feature selection from best parameters
        colmask = np.zeros(self.X_train.shape[1], dtype=bool)

        for feature_name, feature_info in self.feature_groups.items():
            # Handle base and lagged features
            if feature_name in self.best_params and self.best_params[feature_name]:
                colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']] = True
                continue
                
            # Handle lagged features
            base_name = feature_name.split('_D-')[0] if '_D-' in feature_name else feature_name
            lagged_feature_found = False
            
            # Check for any lagged version of this feature
            for lag in [2, 3, 7]:
                lagged_name = f"{base_name}_D-{lag}"
                if lagged_name in self.best_params and self.best_params[lagged_name]:
                    colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']] = True
                    lagged_feature_found = True
                    break
        
        # Ensure at least one feature is selected
        if not np.any(colmask):
            default_feature = list(self.feature_groups.keys())[0]
            feature_info = self.feature_groups[default_feature]
            colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']] = True
            self.logger.warning(f"No features selected. Forcing selection of {default_feature}.")

        # Apply colmask to the scaled data
        self.X_train_selected = X_train_scaled[:, colmask]
        self.X_forecast_selected = X_forecast_scaled[:, colmask]
        self.Y_train_selected = Y_train_scaled
        
        self.selected_input_size = np.sum(colmask)
        self.logger.info(f"After feature selection: {self.selected_input_size} features selected")
        
        return True

    def train_model(self):
        """Train the Linear Quantile model for the current day"""
        try:
            self.logger.info("Training Linear Quantile model...")
            
            # Prepare training data
            X_train = self.X_train_selected
            Y_train = self.Y_train_selected
            
            # Prepare model parameters from best trial (exclude feature selection parameters)
            feature_group_keys = set(self.feature_groups.keys())
            model_params = {k: v for k, v in self.best_params.items() if k not in feature_group_keys}
            
            # Add quantiles
            model_params['quantiles'] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            
            self.logger.info(f"Model parameters: {model_params}")
            
            # Create and train model
            model = LinearQuantileModel(**model_params)
            model.fit(
                X_train=X_train,
                y_train=Y_train,
                feature_names=None  # We don't need feature names for prediction
            )
            
            self.model = model
            self.logger.info("Linear Quantile model training completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def generate_forecast(self):
        """Generate probabilistic forecast for next 24 hours"""
        self.logger.info("Generating forecast...")
        
        forecast_lock_file = os.path.join(os.path.dirname(self.df_forecasts_dir), "forecast_lock.lock")

        # Use the model to generate predictions
        prediction_features = self.X_forecast_selected
        
        # Generate quantile predictions
        quantile_predictions = self.model.predict_quantiles(prediction_features)  # shape: (1, 24, n_quantiles)
        
        # Generate samples for compatibility with evaluation
        samples = self.model.predict_samples(prediction_features, n_samples=10000)  # shape: (10000, 1, 24)
        samples = samples[:, 0, :]  # Remove middle dimension: (10000, 24)
        
        # Get distribution parameters in DDNN-compatible format
        params_dict = self.model.get_distribution_params(prediction_features)

        with filelock.FileLock(forecast_lock_file):
            # Convert samples back to original scale
            scaler_Y = joblib.load(self.scaler_path / 'scaler_Y.joblib')
            predictions_unscaled = scaler_Y.inverse_transform(samples)
            
            # Save the distribution parameters
            params_path = self.params_dir / f"{self.prediction_date.strftime('%Y-%m-%d')}.json"
            with open(params_path, 'w') as f:
                json.dump(params_dict, f)
            
            # Create a dataframe with forecast statistics
            forecast_df = pd.DataFrame(index=pd.date_range(
                start=self.prediction_date,
                periods=24,
                freq='H'
            ))

            # Extract state information from premium forecasts
            state_info = self.get_state_from_predictions(predictions_unscaled)

            # Add state information to forecast dataframe
            state_names = ['normal', 'up_regulation', 'down_regulation']
            forecast_df['state'] = [state_names[s] for s in state_info['states']]
            forecast_df['normal_prob'] = state_info['probabilities']['normal']
            forecast_df['up_prob'] = state_info['probabilities']['up'] 
            forecast_df['down_prob'] = state_info['probabilities']['down']

            # Calculate statistics
            forecast_df['mean'] = np.mean(predictions_unscaled, axis=0)
            forecast_df['median'] = np.median(predictions_unscaled, axis=0)
            forecast_df['std'] = np.std(predictions_unscaled, axis=0)
            forecast_df['p10'] = np.percentile(predictions_unscaled, 10, axis=0)
            forecast_df['p25'] = np.percentile(predictions_unscaled, 25, axis=0)
            forecast_df['p75'] = np.percentile(predictions_unscaled, 75, axis=0)
            forecast_df['p90'] = np.percentile(predictions_unscaled, 90, axis=0)
            
            # Save forecast dataframe
            forecast_df.to_csv(self.df_forecasts_dir / f"{self.prediction_date.strftime('%Y-%m-%d')}.csv")
            
        # Log forecast summary
        self.logger.info(f"Generated Linear Quantile forecast for {self.prediction_date.strftime('%Y-%m-%d')}:")
        for h in range(24):
            hour_str = f"Hour {h:02d}: Mean: {forecast_df['mean'].iloc[h]:.4f}, Std: {forecast_df['std'].iloc[h]:.4f}"
            if h % 6 == 0:  # Log every 6 hours to avoid spam
                self.logger.info(hour_str)
            
        # Store the forecast for return
        self.forecast = forecast_df
        
        return True
            
    def run(self):
        """Run the complete prediction pipeline"""
        try:
            self.logger.info("Starting Linear Quantile prediction pipeline...")
            
            # Load and prepare data
            self.load_data()
            self.prepare_features()
            self.apply_scaling_and_selection()
            
            # Train model and generate forecast
            self.train_model()
            self.generate_forecast()
            
            # Generate summary report
            self.generate_summary_report()
            
            self.logger.info("Linear Quantile prediction pipeline completed successfully")
            return self.forecast
            
        except Exception as e:
            self.logger.error(f"Error in prediction pipeline: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise

    def get_state_from_predictions(self, predictions, threshold=0.5):
        """Extract state information from premium predictions"""
        up_prob = np.mean(predictions > threshold, axis=0)
        down_prob = np.mean(predictions < -threshold, axis=0)
        normal_prob = 1.0 - up_prob - down_prob
        
        states = np.zeros(predictions.shape[1], dtype=int)
        states[up_prob > down_prob] = 1  # Up regulation
        states[down_prob > up_prob] = 2  # Down regulation
        
        return {
            'probabilities': {
                'normal': normal_prob,
                'up': up_prob,
                'down': down_prob
            },
            'states': states
        }

    def generate_summary_report(self):
        """Generate a summary report for market bidding"""
        if not hasattr(self, 'forecast'):
            self.logger.error("No forecast available, generate a forecast first")
            return None
            
        # Create a summary report
        summary = {
            'prediction_date': self.prediction_date.strftime('%Y-%m-%d'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'zone': self.zone,
            'model_type': 'linear_quantile',
            'run_id': self.run_id,
            'forecast': {
                'overall_mean': float(self.forecast['mean'].mean()),
                'overall_std': float(self.forecast['std'].mean()),
                'peak_hours': {
                    'hours': [h for h in range(24) if self.forecast['mean'].iloc[h] > self.forecast['mean'].mean()],
                    'mean': float(self.forecast.loc[self.forecast['mean'] > self.forecast['mean'].mean(), 'mean'].mean())
                },
                'volatile_hours': {
                    'hours': [h for h in range(24) if self.forecast['std'].iloc[h] > self.forecast['std'].mean()],
                    'std': float(self.forecast.loc[self.forecast['std'] > self.forecast['std'].mean(), 'std'].mean())
                }
            }
        }
        
        # Add market state information if available
        if 'state' in self.forecast.columns:
            states = self.forecast['state'].unique()
            summary['market_states'] = {
                state: {
                    'hours': [h for h in range(24) if self.forecast['state'].iloc[h] == state],
                    'count': int((self.forecast['state'] == state).sum()),
                    'mean': float(self.forecast.loc[self.forecast['state'] == state, 'mean'].mean())
                } for state in states
            }
            
        # Save the summary report
        summary_path = self.results_dir / 'summaries' / f"{self.zone}_lqa_{self.run_id}_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary