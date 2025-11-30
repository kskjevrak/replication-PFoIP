import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
import joblib
import json
import os
from typing import Dict, List, Optional, Tuple, Union
import logging

class LinearQuantileModel:
    """
    Linear Quantile Regression model for mFRR premium forecasting.
    
    This class provides a simple linear benchmark for the DDNN models,
    using sklearn's QuantileRegressor with separate models for each quantile and hour.
    Designed to be directly comparable to the probabilistic DDNN outputs.
    """
    
    def __init__(self, 
            quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            alpha: float = 1.0,
            solver: str = 'highs',
            fit_intercept: bool = True):  # REMOVE max_iter parameter
        """
        Initialize Linear Quantile Regression model.
        
        Parameters:
        -----------
        quantiles : List[float]
            Quantile levels to predict (e.g., [0.05, 0.5, 0.95])
        alpha : float
            Regularization strength (L1 penalty)
        solver : str
            Solver to use ('highs', 'highs-ds', 'highs-ipm')
        fit_intercept : bool
            Whether to calculate the intercept
        """
        self.quantiles = np.array(quantiles)
        self.alpha = alpha
        self.solver = solver
        self.fit_intercept = fit_intercept
        
        self.models = {}
        self.feature_names = None
        self.is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> None:
        """
        Fit Linear Quantile Regression model.
        
        Parameters:
        -----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Training features
        y_train : np.ndarray, shape (n_samples, 24)
            Training targets (24 hours)
        X_val : np.ndarray, optional
            Validation features (not used in linear model but kept for API consistency)
        y_val : np.ndarray, optional
            Validation targets (not used in linear model but kept for API consistency)
        feature_names : List[str], optional
            Names of input features
        """
        self.logger.info(f"Training Linear Quantile model with {len(self.quantiles)} quantiles")
        self.logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
        
        self.feature_names = feature_names
        n_samples, n_hours = y_train.shape
        
        # Train separate model for each hour and quantile
        for hour in range(n_hours):
            self.logger.info(f"Training models for hour {hour}")
            
            # Extract hour-specific targets
            y_hour_train = y_train[:, hour]
            y_hour_val = y_val[:, hour] if y_val is not None else None
            
            hour_train_losses = []
            hour_val_losses = []

            for quantile in self.quantiles:
                model = QuantileRegressor(
                    quantile=quantile,
                    alpha=self.alpha,
                    solver=self.solver,
                    fit_intercept=self.fit_intercept
                )
                
                # Fit the model
                model.fit(X_train, y_hour_train)
                            
                # Calculate training loss (pinball loss)
                y_pred_train = model.predict(X_train)
                train_error = y_hour_train - y_pred_train
                train_loss = np.mean(np.maximum(quantile * train_error, (quantile - 1) * train_error))
                hour_train_losses.append(train_loss)
                
                # Calculate validation loss if validation data provided
                if X_val is not None and y_hour_val is not None:
                    y_pred_val = model.predict(X_val)
                    val_error = y_hour_val - y_pred_val
                    val_loss = np.mean(np.maximum(quantile * val_error, (quantile - 1) * val_error))
                    hour_val_losses.append(val_loss)

                # Store model with (hour, quantile) key
                self.models[(hour, quantile)] = model
            
             # Log average losses across all quantiles for this hour
            avg_train_loss = np.mean(hour_train_losses)
            if hour_val_losses:
                avg_val_loss = np.mean(hour_val_losses)
                self.logger.info(f"Hour {hour}: Train loss = {avg_train_loss:.6f}, Val loss = {avg_val_loss:.6f}")
            else:
                self.logger.info(f"Hour {hour}: Train loss = {avg_train_loss:.6f}")

            #self.logger.info(f"Hour {hour}: Trained {len(self.quantiles)} quantile models")
        
        self.is_fitted = True
        self.logger.info("Linear Quantile training completed for all hours and quantiles")
    
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
        n_hours = 24  # Fixed for mFRR application
        n_quantiles = len(self.quantiles)
        
        predictions = np.zeros((n_samples, n_hours, n_quantiles))
        
        for hour in range(n_hours):
            for q_idx, quantile in enumerate(self.quantiles):
                model = self.models[(hour, quantile)]
                hour_quantile_pred = model.predict(X)
                predictions[:, hour, q_idx] = hour_quantile_pred
        
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
                samples[:, input_idx, hour] = np.interp(
                    random_quantiles, 
                    self.quantiles, 
                    quantiles_values
                )
        
        return samples
    
    def get_distribution_params(self, X: np.ndarray) -> Dict:
        """
        Get distribution parameters compatible with DDNN evaluation format.
        
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
        
        # Format compatible with evaluation metrics
        params = {
            'quantiles': self.quantiles.tolist(),
            'values': daily_quantiles.tolist(),  # shape: (24, n_quantiles)
            'model_type': 'linear_quantile'
        }
        
        return params
    
    def get_feature_importance(self) -> Dict:
        """
        Get average feature importance across all models.
        
        Returns:
        --------
        Dict
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        n_features = len(self.feature_names) if self.feature_names else len(next(iter(self.models.values())).coef_)
        
        # Average coefficients across all models (absolute values for importance)
        avg_coef = np.zeros(n_features)
        n_models = 0
        
        for (hour, quantile), model in self.models.items():
            avg_coef += np.abs(model.coef_)
            n_models += 1
        
        avg_coef /= n_models
        
        if self.feature_names:
            return dict(zip(self.feature_names, avg_coef))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(avg_coef)}
    
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
        
        # Save each (hour, quantile) model
        for (hour, quantile), model in self.models.items():
            model_file = os.path.join(save_path, f"linear_quantile_h{hour}_q{quantile:.3f}.joblib")
            joblib.dump(model, model_file)
        
        # Save model metadata
        metadata = {
            'quantiles': self.quantiles.tolist(),
            'alpha': float(self.alpha),
            'solver': self.solver,
            'fit_intercept': self.fit_intercept,
            'feature_names': self.feature_names,
            'n_hours': 24
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
        self.alpha = metadata['alpha']
        self.solver = metadata['solver']
        self.fit_intercept = metadata['fit_intercept']
        self.feature_names = metadata['feature_names']
        
        # Load each (hour, quantile) model
        self.models = {}
        for hour in range(24):
            for quantile in self.quantiles:
                model_file = os.path.join(load_path, f"linear_quantile_h{hour}_q{quantile:.3f}.joblib")
                model = joblib.load(model_file)
                self.models[(hour, quantile)] = model
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {load_path}")

def create_linear_quantile_model(quantiles: List[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
                                **kwargs) -> LinearQuantileModel:
    """
    Factory function to create Linear Quantile model (following DDNN pattern).
    
    Parameters:
    -----------
    quantiles : List[float]
        Quantile levels to predict
    **kwargs
        Additional parameters for LinearQuantileModel
        
    Returns:
    --------
    LinearQuantileModel
        Configured Linear Quantile Regression model
    """
    return LinearQuantileModel(quantiles=quantiles, **kwargs)