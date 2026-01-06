#!/usr/bin/env python
"""
prediction.py - Daily prediction script for mFRR market prices

This script loads the best model parameters from a previous optimization,
loads the latest data, and predicts mFRR prices for the next 24 hours.
It's designed to be run daily to assist bidders in the market.

Usage:
    python prediction.py [--zone ZONE] [--distribution DIST] [--run_id RUN_ID] [--config CONFIG]
"""
import os
import yaml
import traceback
import logging
import filelock
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime, timedelta
import joblib
import optuna
import json
from pathlib import Path

from ..models.neural_nets import create_probabilistic_model
from ..data.loader import DataProcessor

class DailyMarketPredictor:
    """
    Daily probabilistic predictor for electricity imbalance prices.

    This class loads a trained DDNN model and generates probabilistic forecasts
    for the next 24 hours based on historical market data. It's designed for
    operational use in electricity markets to support bidding decisions.

    The prediction pipeline:
    1. Loads best hyperparameters from tuning
    2. Loads and preprocesses recent historical data (window_size days)
    3. Applies saved scalers for feature normalization
    4. Generates distributional forecasts (mean, quantiles, full distribution)
    5. Saves forecasts and distribution parameters

    Attributes
    ----------
    zone : str
        Bidding zone identifier
    distribution : str
        Probability distribution type (JSU, Normal, skewt)
    run_id : str
        Model run identifier
    model_dir : Path
        Directory containing trained model parameters
    device : torch.device
        Computation device (cuda or cpu)
    window_size : int
        Days of historical data used for prediction (default: 730 days / 2 years)

    Example
    -------
    >>> predictor = DailyMarketPredictor(
    ...     zone='no1',
    ...     distribution='jsu',
    ...     run_id='replication_001'
    ... )
    >>> predictor.prediction_date = pd.Timestamp('2024-04-26', tz='Europe/Oslo')
    >>> forecast = predictor.run()
    >>> print(forecast[['mean', 'median', 'q05', 'q95']].head())

    Notes
    -----
    - Requires completed hyperparameter tuning (best_params.yaml must exist)
    - Prediction typically takes 2-5 minutes per day (CPU)
    - Forecasts saved to results/forecasts/df_forecasts_{zone}_{distribution}_{run_id}/
    - Distribution parameters saved to results/forecasts/distparams_{zone}_{distribution}_{run_id}/

    See Also
    --------
    OptunaHPTuner : For hyperparameter optimization
    create_probabilistic_model : For DDNN architecture details
    """

    def __init__(self, zone='no1', distribution='JSU', run_id=None, config_path=None):
        """
        Initialize the daily predictor.

        Parameters
        ----------
        zone : str, default='no1'
            Market zone identifier (no1, no2, no3, no4, no5)
        distribution : str, default='JSU'
            Probability distribution type:
            - 'JSU': Johnson's SU (recommended for heavy-tailed prices)
            - 'Normal': Gaussian distribution
            - 'skewt': Skewed Student's t distribution
        run_id : str, optional
            Identifier for the model run. If None, attempts to use most recent.
            Must match the run_id from hyperparameter tuning.
        config_path : str, optional
            Path to configuration file. If None, uses config/default_config.yml

        Raises
        ------
        FileNotFoundError
            If model directory or best_params.yaml not found
        ValueError
            If distribution type is not supported
        """
        self.zone = zone.lower()
        self.distribution = distribution
        
        # Initialize temperature parameter
        self.temperature = 1.0  # Default temperature value
        
        # Set up directories and paths
        self.root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
        print(self.root_dir)
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
        self.model_dir = self.results_dir / 'models' / f"{self.zone}_{self.distribution.lower()}_{self.run_id}"
        self.scaler_path = self.results_dir / 'scalers' / f"{self.zone}_{self.distribution.lower()}_{self.run_id}"
        self.df_forecasts_dir = self.results_dir / 'forecasts' / f"df_forecasts_{self.zone}_{self.distribution.lower()}_{self.run_id}"
        self.params_dir = self.results_dir / 'forecasts' / f"distparams_{self.zone}_{self.distribution.lower()}_{self.run_id}"
        
        # Create directories if they don't exist
        for directory in [self.df_forecasts_dir, self.params_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Set up logging
        self.logs_dir = self.results_dir / 'logs'
        os.makedirs(self.logs_dir, exist_ok=True)
        self._setup_logging()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.logger.info(f"Using device: {self.device}")
        
        # Load best parameters and model
        self._load_best_parameters()
        
        # Set data parameters
        self.window_size = self.config.get('prediction', {}).get('window_size', 730)  # 2 years default
        self.prediction_hours = self.config.get('prediction', {}).get('hours', 24)
        
        # Initialize data processor
        self.data_processor = DataProcessor(zone=self.zone, config=self.config)
        
    def _setup_logging(self):
        """Configure logging"""
        log_file = self.logs_dir / f"prediction_{self.zone}_{self.distribution}_{self.run_id}_{os.getpid()}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"prediction_{self.zone}_{os.getpid()}")
        
    def _get_latest_run_id(self):
        """Get the latest run ID from available model directories"""
        models_dir = self.results_dir / 'models'
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory {models_dir} not found")
            
        # Find all directories matching the pattern
        pattern = f"{self.zone}_{self.distribution.lower()}_*"
        matching_dirs = list(models_dir.glob(pattern))
        
        if not matching_dirs:
            raise FileNotFoundError(f"No model directories found for {self.zone} with {self.distribution} distribution")
            
        # Sort by modification time (newest first)
        latest_dir = max(matching_dirs, key=os.path.getmtime)
        
        # Extract run_id from directory name
        run_id = latest_dir.name.split('_')[-1]
        return run_id
        
    def _load_best_parameters(self):
        """Load best parameters from previous optimization or use defaults"""
        try:
            # First try to load from YAML file
            best_params_file = self.model_dir / 'best_params.yaml'
            if best_params_file.exists():
                with open(best_params_file, 'r') as f:
                    self.best_params = yaml.safe_load(f)
                self.logger.info(f"Loaded best parameters from {best_params_file}")
                return
                
            # If YAML not found, try to load from optuna study
            study_name = f'{self.zone}_{self.distribution.lower()}_{self.run_id}'
            db_dir = f'./results/models/{self.zone}_{self.distribution.lower()}_{self.run_id}'
            storage_name = f'sqlite:///{os.path.abspath(os.path.join(db_dir, f"{study_name}.db"))}'
            
            try:
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                self.best_params = study.best_params
                self.logger.info(f"Loaded best parameters from optuna study {study_name}")
            except Exception as e:
                self.logger.warning(f"Could not load optuna study: {str(e)}")
                self.logger.info("Using default parameters instead")
                
                # Define default parameters
                self.best_params = {
                    # Model architecture
                    'n_layers': 4,
                    'hidden_size_0': 106,
                    'hidden_size_1': 122,
                    'hidden_size_2': 123,
                    'hidden_size_3': 239,
                    'dropout_rate_0': 0.044454789764886496,
                    'dropout_rate_1': 0.0017282076087829362,
                    'dropout_rate_2': 0.21225144465465506,
                    'dropout_rate_3': 0.2770531782316719,
                    
                    # Training parameters
                    'lr': 0.004003504559614525,
                    'batch_size': 16,
                    'l2_lambda': 1.3057315136635778e-05,
                    'max_grad_norm': 7.439846458358759,
                    'patience': 74,

                    # NEW: Location-Scale parameters
                    'ls_decay_rate': 5.0,
                    'ls_reduction_factor': 0.5,

                    # ADD layer-specific parameters (assuming 4 layers based on your default):
                    'lambda_network_layer_0': 1e-4,
                    'lambda_bias_layer_0': 1e-5,
                    'lambda_network_layer_1': 1e-4,
                    'lambda_bias_layer_1': 1e-5,
                    'lambda_network_layer_2': 1e-4,
                    'lambda_bias_layer_2': 1e-5,
                    'lambda_network_layer_3': 1e-4,
                    'lambda_bias_layer_3': 1e-5,
                    
                    # Distribution-specific regularization (for all distributions)
                    'lambda_dist_weight_loc': 1e-6,
                    'lambda_dist_bias_loc': 1e-7,
                    'lambda_dist_weight_scale': 1e-6,
                    'lambda_dist_bias_scale': 1e-7,
                    
                    # JSU-specific parameters
                    'lambda_dist_weight_tailweight': 1e-6,
                    'lambda_dist_bias_tailweight': 1e-7,
                    'lambda_dist_weight_skewness': 1e-6,
                    'lambda_dist_bias_skewness': 1e-7,
                    
                    # SkewT-specific parameters  
                    'lambda_dist_weight_a': 1e-6,
                    'lambda_dist_bias_a': 1e-7,
                    'lambda_dist_weight_b': 1e-6,
                    'lambda_dist_bias_b': 1e-7,

                    # Feature selection parameters
                    'premium_D-2': True,
                    'premium_D-3': True,
                    'premium_D-7': True,
                    'mFRR_price_up_D-2': False,
                    'mFRR_price_up_D-3': True,
                    'mFRR_price_up_D-7': True,
                    'mFRR_price_down_D-2': False,
                    'mFRR_price_down_D-3': True,
                    'mFRR_price_down_D-7': False,
                    'mFRR_vol_up_D-2': True,
                    'mFRR_vol_up_D-3': True,
                    'mFRR_vol_up_D-7': True,
                    'mFRR_vol_down_D-2': False,
                    'mFRR_vol_down_D-3': True,
                    'mFRR_vol_down_D-7': True,
                    'spot_price': True,
                    'spot_price_D-1': True,
                    'spot_price_D-2': True,
                    'spot_price_D-7': True,
                    'hour_cos': True,
                    'hour_sin': False,
                    'day_of_week_cos': False,
                    'day_of_week_sin': False,
                    'month_cos': True,
                    'month_sin': False,
                    'is_weekend': False,
                    'consumption_forecast': True,
                    'weather_temperature_forecast': True,
                    'production_solar_total_forecast': True,
                    'production_wind_total_forecast': False,
                    'production_hydro_total_forecast': False,
                    'net_exchange_no1_no2': True,
                    'net_exchange_no1_no3': True,
                    'net_exchange_no1_no5': True,
                    'net_exchange_no1_se3': False
                }
                
                # Save default parameters to YAML for future use
                os.makedirs(os.path.dirname(best_params_file), exist_ok=True)
                with open(best_params_file, 'w') as f:
                    yaml.dump(self.best_params, f)
                self.logger.info(f"Saved default parameters to {best_params_file}")
                
        except Exception as e:
            self.logger.error(f"Failed to load parameters: {str(e)}")
            raise
    
    def _create_optimizer_with_param_groups(self, model, lr, lambda_params):
        """
        Create optimizer with layer-specific regularization following DDNN paper Eq(3)+(4)
        """
        # Get number of layers
        n_layers = self.best_params.get('n_layers', 4)
        
        # Organize parameters by layer and type
        layer_weights = {i: [] for i in range(n_layers)}
        layer_biases = {i: [] for i in range(n_layers)}
        distribution_param_groups = {}
        
        # Initialize distribution parameter groups
        if self.distribution.lower() == 'normal':
            param_names = ['loc', 'scale']
        elif self.distribution.lower() == 'jsu':
            param_names = ['loc', 'scale', 'tailweight', 'skewness']
        elif self.distribution.lower() == 'skewt':
            param_names = ['loc', 'scale', 'a', 'b']
        else:
            param_names = []
        
        for param_name in param_names:
            distribution_param_groups[param_name] = {'weights': [], 'biases': []}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Shared network parameters (hidden layers) - GROUP BY LAYER
            if name.startswith('shared_network'):
                # Extract layer number from parameter name
                parts = name.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    
                    # Map sequential layer indices to logical layer numbers
                    # shared_network.0 = layer 0, shared_network.4 = layer 1, etc.
                    # This assumes each layer has 4 components: Linear, BN, Activation, Dropout
                    logical_layer = layer_idx // 4
                    
                    if logical_layer < n_layers:  # Make sure we don't exceed expected layers
                        if name.endswith('.weight') and 'BatchNorm' not in name:
                            layer_weights[logical_layer].append(param)
                        elif name.endswith('.bias') and 'BatchNorm' not in name:
                            layer_biases[logical_layer].append(param)
            
            # Distribution parameter layers
            else:
                param_assigned = False
                for param_name in param_names:
                    if name == f'{param_name}_layer.weight':
                        distribution_param_groups[param_name]['weights'].append(param)
                        param_assigned = True
                        break
                    elif name == f'{param_name}_layer.bias':
                        distribution_param_groups[param_name]['biases'].append(param)
                        param_assigned = True
                        break
                
                if not param_assigned:
                    print(f"WARNING: Parameter {name} was not assigned to any group!")
        
        # Create parameter groups
        param_groups = []
        
        # Add layer-specific parameter groups
        for layer_idx in range(n_layers):
            if layer_weights[layer_idx]:
                param_groups.append({
                    'params': layer_weights[layer_idx],
                    'weight_decay': lambda_params.get(f'network_layer_{layer_idx}', 1e-5)
                })
            
            if layer_biases[layer_idx]:
                param_groups.append({
                    'params': layer_biases[layer_idx],
                    'weight_decay': lambda_params.get(f'bias_layer_{layer_idx}', 1e-6)
                })
        
        # Add distribution parameter groups
        for param_name in param_names:
            if distribution_param_groups[param_name]['weights']:
                param_groups.append({
                    'params': distribution_param_groups[param_name]['weights'],
                    'weight_decay': lambda_params.get(f'dist_weight_{param_name}', 1e-7)
                })
            
            if distribution_param_groups[param_name]['biases']:
                param_groups.append({
                    'params': distribution_param_groups[param_name]['biases'],
                    'weight_decay': lambda_params.get(f'dist_bias_{param_name}', 1e-8)
                })
        
        # Count parameters and validate regularization groups
        total_params = 0
        for group in param_groups:
            total_params += sum(p.numel() for p in group['params'])
        
        # Expected number of regularization groups:
        # - 2 groups per hidden layer (weights + biases) = 2 * n_layers
        # - 2 groups per distribution parameter (weights + biases) = 2 * len(param_names)
        expected_groups = 2 * n_layers + 2 * len(param_names)
        actual_groups = len(param_groups)
        
        status = "OK" if actual_groups == expected_groups else f"MISMATCH (expected {expected_groups})"
        self.logger.info(f"Layer-wise regularization {status}: {actual_groups} groups, {total_params:,} params")
        
        return torch.optim.Adam(param_groups, lr=lr)

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
        """Prepare feature matrices for model training and prediction using a similar approach to DataProcessor"""
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
            
            # Fill X and Y with values for each day to generate matrices
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
        #try:
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
        
        # Feature selection based on best parameters
        colmask = np.zeros(self.X_train.shape[1], dtype=bool)
        
        # Apply feature selection
        # In prediction.py, update apply_scaling_and_selection
        for feature_name, feature_info in self.feature_groups.items():
            # Handle base features
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
            
            # If neither base nor lagged versions are found, continue
            if not lagged_feature_found:
                continue
        
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
        
        #except Exception as e:
        #    self.logger.error(f"Error in scaling and feature selection: {str(e)}")
        #    return False

    def train_model(self):
        """Train the probabilistic forecast model for the current day"""
        try:
            self.logger.info("Training forecast model...")
            
            # Different model initialization per run_id
            base_seed = 42
            date_seed = int(self.prediction_date.strftime('%Y%m%d'))

            # Handle both numeric and string run_ids
            try:
                run_id_numeric = int(self.run_id)
            except (ValueError, TypeError):
                # Hash the run_id string to get a consistent numeric value
                run_id_numeric = abs(hash(str(self.run_id))) % 10000

            run_seed = base_seed + run_id_numeric * 1000 + date_seed

            # Set PyTorch seeds BEFORE model creation
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(run_seed)

            # Also update train/val split seed
            split_seed = date_seed + run_id_numeric * 10
            np.random.seed(split_seed)  # Different split per run_id

            # Prepare training and validation datasets
            X_train = self.X_train_selected
            Y_train = self.Y_train_selected
            
            # Create a validation set
            np.random.seed(42)  # For reproducibility
            perm = np.random.permutation(X_train.shape[0])
            val_split = 0.2
            train_idx = perm[:int((1 - val_split) * len(perm))]
            val_idx = perm[int((1 - val_split) * len(perm)):]
            
            # Create PyTorch datasets
            X_train_tensor = torch.FloatTensor(X_train[train_idx])
            Y_train_tensor = torch.FloatTensor(Y_train[train_idx])
            X_val_tensor = torch.FloatTensor(X_train[val_idx])
            Y_val_tensor = torch.FloatTensor(Y_train[val_idx])
            
            train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
            
            batch_size = self.best_params.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Set up model with hyperparameters from best run
            hidden_layers_config = []
            n_layers = self.best_params.get('n_layers', 4)

            for i in range(n_layers):
                hidden_layers_config.append({
                    'units': self.best_params.get(f'hidden_size_{i}', 128),
                    'activation': self.best_params.get(f'activation_{i}', 'relu'),
                    'batch_norm': True,
                    'dropout_rate': self.best_params.get(f'dropout_rate_{i}', 0.1)
                })
            
            # Get LS parameters from best_params
            ls_decay_rate = self.best_params.get('ls_decay_rate', 5.0)
            ls_reduction_factor = self.best_params.get('ls_reduction_factor', 0.5)

            # Convert selected_input_size to Python int
            input_size = int(self.selected_input_size)
            self.logger.info(f"Creating model with input size: {input_size}")
            
            # Create model
            model = create_probabilistic_model(
                distribution=self.distribution,
                input_size=input_size,
                hidden_layers_config=hidden_layers_config,
                output_size=24,
                ls_decay_rate=ls_decay_rate,
                ls_reduction_factor=ls_reduction_factor
            )

            # Log model architecture details
            self.logger.info(f"Created model with {n_layers} layers:")
            for i, config in enumerate(hidden_layers_config):
                self.logger.info(f"  Layer {i+1}: {config['units']} units, dropout={config['dropout_rate']}")
            
            model.to(self.device)
            
            # Training loop parameters
            patience = self.best_params.get('patience', 74)  # Updated from tuned hyperparameters
            lr = self.best_params.get('lr', 0.004)  # Use lr from parameters 
            #l2_lambda = self.best_params.get('l2_lambda', 1e-04)  # Use l2_lambda from parameters
            max_grad_norm = self.best_params.get('max_grad_norm', 1.0)
            #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            max_epochs = self.best_params.get('max_epochs', 1500)
            
            # Prepare layer-specific lambda parameters for optimizer
            lambda_params = {}
            n_layers = self.best_params.get('n_layers', 4)

            # Layer-specific regularization parameters
            for i in range(n_layers):
                lambda_params[f'network_layer_{i}'] = self.best_params.get(f'lambda_network_layer_{i}', 1e-4)
                lambda_params[f'bias_layer_{i}'] = self.best_params.get(f'lambda_bias_layer_{i}', 1e-5)
            
            # Add distribution-specific lambda parameters
            if self.distribution.lower() == 'normal':
                param_types = ['loc', 'scale']
            elif self.distribution.lower() == 'jsu':
                param_types = ['loc', 'scale', 'tailweight', 'skewness']
            elif self.distribution.lower() == 'skewt':
                param_types = ['loc', 'scale', 'a', 'b']
            else:
                param_types = []
            
            for param_type in param_types:
                lambda_params[f'dist_weight_{param_type}'] = self.best_params.get(f'lambda_dist_weight_{param_type}', 1e-6)
                lambda_params[f'dist_bias_{param_type}'] = self.best_params.get(f'lambda_dist_bias_{param_type}', 1e-7)
            
            # Create optimizer with parameter groups
            optimizer = self._create_optimizer_with_param_groups(model, lr, lambda_params)
                

            best_val_loss = float('inf')
            best_epoch = 0
            no_improvement = 0
            best_state_dict = None
            
            # Training loop
            for epoch in range(max_epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                
                for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    dist = model(X_batch)
                    
                    # Simple negative log-likelihood - regularization handled by optimizer
                    nll_loss = -dist.log_prob(y_batch).mean()
                    
                    # No manual L2 regularization needed - handled by weight_decay
                    loss = nll_loss
                    
                    # Backward and optimize
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    
                    optimizer.step()
                    train_loss += nll_loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation phase
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        
                        dist = model(X_batch)
                        loss = -dist.log_prob(y_batch).mean()
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                               
                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    no_improvement = 0
                    # Save the model
                    best_state_dict = model.state_dict().copy()
                else:
                    no_improvement += 1
                
                # Log progress
                if epoch % 10 == 0 or epoch == max_epochs - 1:
                    self.logger.info(f"Epoch {epoch}/{max_epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
                
                # Early stopping
                if no_improvement >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch} with val loss: {best_val_loss:.6f}")
                    break
            
            # Load the best model
            if best_state_dict is None:
                raise RuntimeError("Model training failed - no best state dict found")
                
            model.load_state_dict(best_state_dict)
            self.model = model
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
            
    def generate_forecast(self):
        """Generate probabilistic forecast for next 24 hours"""
        
        #try:
        self.logger.info("Generating forecast...")
        
        # Add to the beginning of generate_forecast method:
        forecast_lock_file = os.path.join(os.path.dirname(self.df_forecasts_dir), "forecast_lock.lock")

        # Use the model to generate predictions
        self.model.eval()
        
        prediction_features = self.X_forecast_selected
            
        Xf_tensor = torch.FloatTensor(prediction_features).to(self.device)
        
        # Generate samples
        with torch.no_grad():
            dist = self.model(Xf_tensor)
            samples = dist.sample((10000,))
            
        # Get distribution parameters
        if self.distribution.lower() == 'normal':
            params_dict = {
                'loc': dist.loc.cpu().numpy().flatten(),
                'scale': dist.scale.cpu().numpy().flatten()
            }

        elif self.distribution.lower() == 'jsu':
            params_dict = {
                'loc': dist.loc.cpu().numpy().flatten(),
                'scale': dist.scale.cpu().numpy().flatten(),
                'tailweight': dist.tailweight.cpu().numpy().flatten(),
                'skewness': dist.skewness.cpu().numpy().flatten()
            }

        elif self.distribution.lower() == 'skewt':
            params_dict = {
                'loc': dist.loc.cpu().numpy().flatten(),
                'scale': dist.scale.cpu().numpy().flatten(),
                'a': dist.a.cpu().numpy().flatten(),
                'b': dist.b.cpu().numpy().flatten()
            }

        with filelock.FileLock(forecast_lock_file):
                                                        
            # Convert samples back to original scale
            scaler_Y = joblib.load(self.scaler_path / 'scaler_Y.joblib')
            predictions = scaler_Y.inverse_transform(samples.cpu().numpy().reshape(-1, 24))
            
            
            # Save the distribution parameters
            params_path = self.params_dir / f"{self.prediction_date.strftime('%Y-%m-%d')}.json"
            with open(params_path, 'w') as f:
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params_dict.items()}, f)
            
            # Create a dataframe with forecast statistics
            forecast_df = pd.DataFrame(index=pd.date_range(
                start=self.prediction_date,
                periods=24,
                freq='h'
            ))

            # Extract state information from premium forecasts
            state_info = self.get_state_from_predictions(predictions)

             # Add state information to forecast dataframe
            state_names = ['normal', 'up_regulation', 'down_regulation']
            forecast_df['state'] = [state_names[s] for s in state_info['states']]
            forecast_df['normal_prob'] = state_info['probabilities']['normal']
            forecast_df['up_prob'] = state_info['probabilities']['up'] 
            forecast_df['down_prob'] = state_info['probabilities']['down']

            # Calculate statistics
            forecast_df['mean'] = np.mean(predictions, axis=0)
            forecast_df['median'] = np.median(predictions, axis=0)
            forecast_df['std'] = np.std(predictions, axis=0)
            forecast_df['p01'] = np.percentile(predictions, 1, axis=0)
            forecast_df['p05'] = np.percentile(predictions, 5, axis=0)
            forecast_df['p10'] = np.percentile(predictions, 10, axis=0)
            forecast_df['p25'] = np.percentile(predictions, 25, axis=0)
            forecast_df['p75'] = np.percentile(predictions, 75, axis=0)
            forecast_df['p90'] = np.percentile(predictions, 90, axis=0)
            forecast_df['p95'] = np.percentile(predictions, 95, axis=0)
            forecast_df['p98'] = np.percentile(predictions, 99, axis=0)
            
            # Save forecast dataframe
            forecast_df.to_csv(self.df_forecasts_dir / f"{self.prediction_date.strftime('%Y-%m-%d')}.csv")
            
        # Log forecast summary
        self.logger.info(f"Generated forecast for {self.prediction_date.strftime('%Y-%m-%d')}:")
        hourly_summary = []
        for h in range(24):
            hour_str = f"Hour {h:02d}: "
            hour_str += f"Mean: {forecast_df['mean'].iloc[h]:.2f}, "
            hour_str += f"Median: {forecast_df['median'].iloc[h]:.2f}, "
            hour_str += f"Std: {forecast_df['std'].iloc[h]:.2f}"
            hourly_summary.append(hour_str)
        
        for i, summary in enumerate(hourly_summary):
            self.logger.info(summary)
            
        # Store the forecast for return
        self.forecast = forecast_df
        
        return True
            
        #except Exception as e:
        #    self.logger.error(f"Error generating forecast: {str(e)}")
        #    return False
            
    def run(self):
        """Run the complete prediction pipeline"""
        try:
            self.logger.info("Starting prediction pipeline...")
            
            # Load and prepare data
            self.load_data()
            self.prepare_features()
            self.apply_scaling_and_selection()
            
            # Train model and generate forecast
            self.train_model()
            self.generate_forecast()
            
            # Generate summary report
            self.generate_summary_report()
            
            self.logger.info("Prediction pipeline completed successfully")
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
            'distribution': self.distribution,
            'run_id': self.run_id,
            'forecast': {
                'overall_mean': float(self.forecast['mean'].mean()),
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
            
        # Add bidding recommendations
        summary['bidding_strategy'] = {
            'high_value_hours': [h for h in range(24) if (
                self.forecast['mean'].iloc[h] > self.forecast['mean'].quantile(0.75)
            )],
            'high_risk_hours': [h for h in range(24) if (
                self.forecast['std'].iloc[h] > self.forecast['std'].quantile(0.75)
            )]
        }
        
        # Save the summary report
        summary_path = self.results_dir / 'summaries' / f"{self.zone}_{self.distribution.lower()}_{self.run_id}_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary
        

def main():
    """Main function to run daily prediction"""
    '''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Daily mFRR price prediction script")
    parser.add_argument("--zone", default="no1", help="Market zone (e.g., no1, no2)")
    parser.add_argument("--distribution", default="JSU", choices=["JSU", "Normal", "StudentT", "st"], help="Probability distribution")
    parser.add_argument("--run_id", default=None, help="Specific run ID (uses latest if not specified)")
    parser.add_argument("--config", default=None, help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Create and run the predictor
    predictor = DailyMarketPredictor(
        zone=args.zone,
        distribution=args.distribution,
        run_id=args.run_id,
        config_path=args.config
    )
    '''
    predictor = DailyMarketPredictor(
        zone="no1",
        distribution="JSU",
        run_id="1"
    )

    forecast = predictor.run()
    
    if forecast is not None:
        # Generate a summary report for bidding
        summary = predictor.generate_summary_report()
        print(f"Prediction complete. Summary saved to {predictor.results_dir}/summaries/")
    else:
        print("Prediction failed. Check the logs for more information.")
    
if __name__ == "__main__":
    main()