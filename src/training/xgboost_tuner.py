import os
import yaml
import logging
import optuna
import numpy as np
import pandas as pd
import joblib
import pickle
from sklearn.metrics import mean_absolute_error

from ..models.xgboost_quantile import XGBoostQuantileModel
from ..data.loader import DataProcessor

class XGBoostTuner:
    """Hyperparameter tuner using Optuna for XGBoost quantile regression"""
    
    def __init__(self, config_path, zone, run_id):
        """Initialize tuner with configuration"""
        self.config_path = config_path
        self.zone = zone
        self.run_id = run_id
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up directories
        self.results_dir = f'./results/models/{self.zone}_xgboost_{run_id}'
        self.scaler_path = f'./results/scalers/{self.zone}_xgboost_{run_id}'
        self.logs_dir = './results/logs'
        self.preproc_dir = f'./src/data/preprocessing/{self.zone}_xgboost_{self.run_id}'
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.scaler_path, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.preproc_dir, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Initialize variables
        self.data_loaded = False
        self.feature_groups = {}
        
        # Define quantiles for XGBoost
        self.quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.logs_dir}/{self.zone}_xgboost_{self.run_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{self.zone}_xgboost_{self.run_id}")
        
    def _load_data(self):
        """Load data for modeling"""
        self.logger.info("Loading and preparing data...")
        
        try:
            data_processor = DataProcessor(
                zone=self.zone,
                config=self.config
            )
            
            self.train_loader_dataset, self.val_loader_dataset, self.feature_groups, self.scaler_X, self.scaler_Y = data_processor.prepare_torch_datasets()
            
            # Validate feature groups structure
            if not self.feature_groups:
                self.logger.error("Feature groups dictionary is empty! Check DataProcessor.")
                raise ValueError("Empty feature groups dictionary")
                
            self.X_train, self.Y_train = self.train_loader_dataset.dataset.tensors
            self.X_val, self.Y_val = self.val_loader_dataset.dataset.tensors
            
            # Convert PyTorch tensors to numpy arrays for XGBoost
            self.X_train = self.X_train.numpy()
            self.Y_train = self.Y_train.numpy()
            self.X_val = self.X_val.numpy()
            self.Y_val = self.Y_val.numpy()
            
            self.logger.info(f"Data loaded successfully with sizes: X_train: {self.X_train.shape}, Y_train {self.Y_train.shape}, X_val {self.X_val.shape}, Y_val {self.Y_val.shape}")
            
            # Save scalers
            for feature_name, scaler in self.scaler_X.items():
                joblib.dump(scaler, f"{self.scaler_path}/scaler_X_{feature_name}.joblib")
            joblib.dump(self.scaler_Y, f"{self.scaler_path}/scaler_Y.joblib")
            
            self.data_loaded = True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        if not self.data_loaded:
            self._load_data()
        
        # XGBoost hyperparameters based on Nordic market literature
        params = {
            'quantiles': self.quantiles,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'n_estimators': trial.suggest_int('n_estimators', 25, 150), #Cut training shorter
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 5, 20),
            'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 50), #Cut training shorter
            'random_state': 42
        }
        
        # Feature selection using same approach as DDNN
        colmask = np.zeros(self.X_train.shape[1], dtype=bool)
        
        # Perform feature selection for each feature group
        for feature_name, feature_info in self.feature_groups.items():
            # Suggest whether to include this feature group
            if trial.suggest_categorical(feature_name, [True, False]):
                start_idx = feature_info['start_idx']
                size = feature_info['size']
                colmask[start_idx:start_idx + size] = True

        # Calculate filtered input size
        filtered_input_size = int(np.sum(colmask))
        
        # Ensure at least one feature is selected
        if filtered_input_size == 0:
            # Force selection of at least one feature group
            default_feature = list(self.feature_groups.keys())[0]
            feature_info = self.feature_groups[default_feature]
            colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']] = True
            filtered_input_size = feature_info['size']
            self.logger.warning(f"No features selected. Forcing selection of {default_feature}.")

        # Apply feature selection
        X_train_selected = self.X_train[:, colmask]
        X_val_selected = self.X_val[:, colmask]
        
        # Create feature names for XGBoost
        feature_names = []
        for feature_name, feature_info in self.feature_groups.items():
            if np.any(colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']]):
                for i in range(feature_info['size']):
                    feature_names.append(f"{feature_name}_{i}")
        
        # Create and train XGBoost model
        try:
            model = XGBoostQuantileModel(**params)
            
            # Train model
            model.fit(
                X_train=X_train_selected,
                y_train=self.Y_train,
                X_val=X_val_selected,
                y_val=self.Y_val,
                feature_names=feature_names
            )
            
            # Evaluate model using quantile loss (pinball loss)
            val_predictions = model.predict_quantiles(X_val_selected)
            
            # Calculate average pinball loss across all quantiles and hours
            total_loss = 0.0
            n_samples, n_hours, n_quantiles = val_predictions.shape
            
            for q_idx, quantile in enumerate(self.quantiles):
                for hour in range(n_hours):
                    y_true = self.Y_val[:, hour]
                    y_pred = val_predictions[:, hour, q_idx]
                    
                    # Pinball loss calculation
                    error = y_true - y_pred
                    loss = np.maximum(quantile * error, (quantile - 1) * error)
                    total_loss += np.mean(loss)
            
            # Average loss across all quantiles and hours
            avg_loss = total_loss / (n_quantiles * n_hours)
            
            self.logger.info(f"Trial {trial.number}: Average pinball loss = {avg_loss:.6f}")
            
            return avg_loss
            
        except Exception as e:
            self.logger.error(f"Error in trial {trial.number}: {str(e)}")
            return float('inf')  # Return worst possible score

    def run_optimization(self, n_trials=100, n_jobs=1):
        """Run the optimization process"""
        # Preprocess and save data before parallelization
        self.logger.info("Preprocessing data before optimization...")
        
        # Create a data processor
        data_processor = DataProcessor(
            zone=self.zone,
            config=self.config
        )
        
        # Load and process the data
        train_loader, val_loader, feature_groups, scaler_X, scaler_Y = data_processor.prepare_torch_datasets()
        
        # Check if lagged features are present
        has_lagged_features = any('D-' in feature for feature in feature_groups.keys())
        self.logger.info(f"Lagged features {'are' if has_lagged_features else 'are not'} present in the dataset")
        
        # Extract feature groups
        self.feature_groups = feature_groups

        # Extract tensors from dataloaders
        X_train, Y_train = train_loader.dataset.tensors
        X_val, Y_val = val_loader.dataset.tensors
        
        # Convert to numpy and save
        X_train_np = X_train.numpy()
        Y_train_np = Y_train.numpy()
        X_val_np = X_val.numpy()
        Y_val_np = Y_val.numpy()
        
        np.save(f'{self.preproc_dir}/X_train.npy', X_train_np)
        np.save(f'{self.preproc_dir}/Y_train.npy', Y_train_np)
        np.save(f'{self.preproc_dir}/X_val.npy', X_val_np)
        np.save(f'{self.preproc_dir}/Y_val.npy', Y_val_np)
        
        # Save scalers
        for feature_name, scaler in scaler_X.items():
            joblib.dump(scaler, f"{self.scaler_path}/scaler_X_{feature_name}.joblib")
        joblib.dump(scaler_Y, f"{self.scaler_path}/scaler_Y.joblib")
        
        # Save feature groups
        with open(f'{self.preproc_dir}/feature_groups.pkl', 'wb') as f:
            pickle.dump(feature_groups, f)
        
        self.logger.info(f"Data preprocessed and saved to {self.preproc_dir}")
        
        # Set up storage
        study_name = f'xgboost_{self.zone}_{self.run_id}'
        db_dir = f'./results/models/{self.zone}_xgboost_{self.run_id}'
        os.makedirs(db_dir, exist_ok=True)
        storage_path = os.path.abspath(os.path.join(db_dir, f"{study_name}.db"))
        storage_name = f'sqlite:///{storage_path}'
        
        # Check if the study already exists and delete it if it does
        if os.path.exists(storage_path):
            self.logger.info(f"Found existing study at {storage_path}. Deleting to start fresh.")
            try:
                os.remove(storage_path)
                self.logger.info("Database file removed.")
            except Exception as e:
                self.logger.warning(f"Error while trying to delete existing study: {str(e)}")
                self.logger.warning("Will attempt to continue by creating a new study.")
        
        self.logger.info(f"Starting XGBoost optimization with {n_trials} trials")
        
        try:
            # Use different pruner and sampler settings for XGBoost
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                interval_steps=10
            )
            
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=15,
                n_ei_candidates=24,
                seed=42
            )

            study = optuna.create_study(
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=False,
                direction='minimize',  # Minimize pinball loss
                pruner=pruner,
                sampler=sampler
            )
            
            study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
                        
            # Log the best parameters
            self.logger.info(f"Best pinball loss: {best_value}")
            self.logger.info(f"Best params: {best_params}")
            
            # Train final model with best parameters
            self.logger.info("Training final model with best parameters...")
            
            # Load preprocessed data
            X_train = np.load(f'{self.preproc_dir}/X_train.npy')
            Y_train = np.load(f'{self.preproc_dir}/Y_train.npy')
            X_val = np.load(f'{self.preproc_dir}/X_val.npy')
            Y_val = np.load(f'{self.preproc_dir}/Y_val.npy')
            
            # Apply feature selection from best trial
            colmask = np.zeros(X_train.shape[1], dtype=bool)
            for feature_name, feature_info in feature_groups.items():
                if feature_name in best_params and best_params[feature_name]:
                    start_idx = feature_info['start_idx']
                    size = feature_info['size']
                    colmask[start_idx:start_idx + size] = True
            
            # Ensure at least one feature is selected
            if not np.any(colmask):
                default_feature = list(feature_groups.keys())[0]
                feature_info = feature_groups[default_feature]
                colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']] = True
            
            X_train_selected = X_train[:, colmask]
            X_val_selected = X_val[:, colmask]
            
            # Create feature names
            feature_names = []
            for feature_name, feature_info in feature_groups.items():
                if np.any(colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']]):
                    for i in range(feature_info['size']):
                        feature_names.append(f"{feature_name}_{i}")
            
            # Prepare final model parameters
            final_params = {k: v for k, v in best_params.items() if k not in feature_groups}
            final_params['quantiles'] = self.quantiles
            
            # Train final model
            final_model = XGBoostQuantileModel(**final_params)
            final_model.fit(
                X_train=X_train_selected,
                y_train=Y_train,
                X_val=X_val_selected,
                y_val=Y_val,
                feature_names=feature_names
            )
            
            # Save the final model (following DDNN pattern: temp → final)
            temp_model_dir = f"{self.results_dir}/best_model_temp"
            final_model.save_model(temp_model_dir)
            
            # Save best parameters
            with open(f'{self.results_dir}/best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
            
            # Rename temp model to final (matching DDNN pattern)
            final_model_dir = f"{self.results_dir}/best_model"
            os.rename(temp_model_dir, final_model_dir)
            
            self.logger.info(f"Final model and parameters saved to {self.results_dir}")
                
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise

def create_xgboost_tuner(config_path, zone, run_id):
    """Factory function to create XGBoost tuner"""
    return XGBoostTuner(config_path, zone, run_id)