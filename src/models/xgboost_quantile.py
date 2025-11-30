import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import logging

class XGBoostQuantileModel:
    """
    XGBoost Quantile Regression model for mFRR premium forecasting.
    
    This class provides a probabilistic forecasting capability using XGBoost's 
    quantile regression objective, designed to be a benchmark for DDNN models.
    Uses 24 separate models (one per hour) for proper hour-specific predictions.
    """
    
    def __init__(self, 
                 quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                 n_estimators: int = 100,
                 learning_rate: float = 0.05,
                 max_depth: int = 6,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.9,
                 reg_alpha: float = 1.0,
                 reg_lambda: float = 1.0,
                 min_child_weight: int = 10,
                 early_stopping_rounds: int = 50,
                 random_state: int = 42):
        """
        Initialize XGBoost Quantile Regression model.
        
        Parameters:
        -----------
        quantiles : List[float]
            Quantile levels to predict (e.g., [0.05, 0.5, 0.95])
        n_estimators : int
            Number of boosting rounds
        learning_rate : float
            Learning rate (eta) for boosting
        max_depth : int
            Maximum tree depth
        subsample : float
            Fraction of samples used for each tree
        colsample_bytree : float
            Fraction of features used for each tree
        reg_alpha : float
            L1 regularization term
        reg_lambda : float
            L2 regularization term
        min_child_weight : int
            Minimum sum of instance weight needed in a child
        early_stopping_rounds : int
            Stop training if no improvement for this many rounds
        random_state : int
            Random seed for reproducibility
        """
        self.quantiles = np.array(quantiles)
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        
        # XGBoost parameters following Nordic market best practices
        self.params = {
            'objective': 'reg:quantileerror',
            'tree_method': 'hist',
            'quantile_alpha': self.quantiles,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'min_child_weight': min_child_weight,
            'random_state': random_state,
            'eval_metric': 'quantile',
            'verbosity': 0  # Reduce output for cleaner logs
        }
        
        self.models = {}  # Will store one model per hour (0-23)
        self.feature_names = None
        self.is_fitted = False
        
        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> None:
        """
        Fit XGBoost quantile regression model.
        
        Parameters:
        -----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Training features
        y_train : np.ndarray, shape (n_samples, 24)
            Training targets (24 hours)
        X_val : np.ndarray, optional
            Validation features
        y_val : np.ndarray, optional
            Validation targets
        feature_names : List[str], optional
            Names of input features
        """
        self.logger.info(f"Training XGBoost quantile model with {len(self.quantiles)} quantiles")
        self.logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        self.feature_names = feature_names
        n_samples, n_hours = y_train.shape
        
        # Train separate model for each hour to capture hour-specific patterns
        for hour in range(n_hours):
            self.logger.info(f"Training model for hour {hour}")
            
            # Extract hour-specific targets
            y_hour_train = y_train[:, hour]
            
            # Create XGBoost datasets
            dtrain = xgb.QuantileDMatrix(X_train, y_hour_train)
            
            evals = [(dtrain, 'train')]
            evals_result = {}
            
            # Add validation set if provided
            if X_val is not None and y_val is not None:
                y_hour_val = y_val[:, hour]
                dval = xgb.QuantileDMatrix(X_val, y_hour_val, ref=dtrain)
                evals.append((dval, 'val'))
            
            # Train model for this hour
            model = xgb.train(
                params=self.params,
                dtrain=dtrain,
                num_boost_round=self.n_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                evals=evals,
                evals_result=evals_result,
                verbose_eval=False  # Suppress training output
            )
            
            self.models[hour] = model
            
            # Log training summary with both train and val loss
            if 'val' in evals_result:
                final_train_score = evals_result['train']['quantile'][-1]
                final_val_score = evals_result['val']['quantile'][-1]
                self.logger.info(f"Hour {hour}: Train loss = {final_train_score:.4f}, Val loss = {final_val_score:.4f}")
            else:
                final_train_score = evals_result['train']['quantile'][-1]
                self.logger.info(f"Hour {hour}: Train loss = {final_train_score:.4f}")
        
        self.is_fitted = True
        self.logger.info("XGBoost training completed for all hours")
    
    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        """
        Predict quantiles for input features.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features
            
        Returns:
        --------
        np.ndarray, shape (n_samples, n_hours, n_quantiles)
            Quantile predictions for each hour
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        n_samples = X.shape[0]
        n_hours = len(self.models)
        n_quantiles = len(self.quantiles)
        
        predictions = np.zeros((n_samples, n_hours, n_quantiles))
        
        for hour in range(n_hours):
            # Get quantile predictions for this hour
            hour_preds = self.models[hour].inplace_predict(X)
            predictions[:, hour, :] = hour_preds
        
        return predictions
    
    def predict_samples(self, X: np.ndarray, n_samples: int = 10000) -> np.ndarray:
        """
        Generate samples from quantile predictions using linear interpolation.
        
        Parameters:
        -----------
        X : np.ndarray, shape (n_inputs, n_features)
            Input features
        n_samples : int
            Number of samples to generate
            
        Returns:
        --------
        np.ndarray, shape (n_samples, n_inputs, 24)
            Generated samples for each input and hour
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get quantile predictions
        quantile_preds = self.predict_quantiles(X)  # shape: (n_inputs, 24, n_quantiles)
        n_inputs, n_hours, n_quantiles = quantile_preds.shape
        
        # Generate samples using linear interpolation between quantiles
        samples = np.zeros((n_samples, n_inputs, n_hours))
        
        for input_idx in range(n_inputs):
            for hour in range(n_hours):
                # Get quantiles for this input and hour
                quantiles_values = quantile_preds[input_idx, hour, :]
                
                # Generate random quantile levels
                random_quantiles = np.random.uniform(0, 1, n_samples)
                
                # Interpolate to get corresponding values
                # Handle edge cases for extrapolation
                samples[:, input_idx, hour] = np.interp(
                    random_quantiles, 
                    self.quantiles, 
                    quantiles_values
                )
        
        return samples
    
    def get_distribution_params(self, X: np.ndarray) -> Dict:
        """
        Get distribution parameters compatible with DDNN output format.
        
        Parameters:
        -----------
        X : np.ndarray
            Input features (typically single prediction day)
            
        Returns:
        --------
        Dict
            Distribution parameters in DDNN-compatible format
        """
        quantile_preds = self.predict_quantiles(X)  # shape: (n_inputs, 24, n_quantiles)
        
        # For daily prediction (like DDNN), we expect single input
        if quantile_preds.shape[0] != 1:
            raise ValueError(f"Expected single input for daily prediction, got {quantile_preds.shape[0]} inputs")
        
        # Extract predictions for the single day: shape (24, n_quantiles)
        daily_quantiles = quantile_preds[0]  # Remove input dimension
        
        # Simple storage format for easy evaluation integration
        params = {
            'quantiles': self.quantiles.tolist(),
            'values': daily_quantiles.tolist(),  # shape: (24, n_quantiles)
            'model_type': 'xgboost'
        }
        
        return params
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Parameters:
        -----------
        save_path : str
            Directory path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save each hour's model
        for hour, model in self.models.items():
            model_file = os.path.join(save_path, f"xgb_hour_{hour}.json")
            model.save_model(model_file)
        
        # Save model metadata
        def convert_numpy_to_list(obj):
            """Recursively convert numpy arrays to lists for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_to_list(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            else:
                return obj

        metadata = {
            'quantiles': self.quantiles.tolist(),
            'params': convert_numpy_to_list(self.params),
            'feature_names': convert_numpy_to_list(self.feature_names),
            'n_hours': len(self.models)
        }

        metadata_file = os.path.join(save_path, "model_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str) -> None:
        """
        Load a trained model.
        
        Parameters:
        -----------
        load_path : str
            Directory path to load the model from
        """
        # Load metadata
        metadata_file = os.path.join(load_path, "model_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        self.quantiles = np.array(metadata['quantiles'])
        self.params.update(metadata['params'])
        self.feature_names = metadata['feature_names']
        n_hours = metadata['n_hours']
        
        # Load each hour's model
        self.models = {}
        for hour in range(n_hours):
            model_file = os.path.join(load_path, f"xgb_hour_{hour}.json")
            model = xgb.Booster()
            model.load_model(model_file)
            self.models[hour] = model
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {load_path}")

def create_xgboost_quantile_model(quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                                  **kwargs) -> XGBoostQuantileModel:
    """
    Factory function to create XGBoost quantile model (following DDNN pattern).
    
    Parameters:
    -----------
    quantiles : List[float]
        Quantile levels to predict
    **kwargs
        Additional parameters for XGBoostQuantileModel
        
    Returns:
    --------
    XGBoostQuantileModel
        Configured XGBoost quantile regression model
    """
    return XGBoostQuantileModel(quantiles=quantiles, **kwargs)