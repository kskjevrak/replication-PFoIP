"""
Data loading and preprocessing module for electricity market data.

This module provides the DataProcessor class for loading, preprocessing, and
preparing electricity market imbalance price data for deep learning models.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import logging


class DataProcessor:
    """
    Data processor for electricity market imbalance price forecasting.

    Handles:
    - Loading parquet data files
    - Creating lagged features (D-2, D-3, D-7)
    - Creating cyclical time features (hour, day of week, month)
    - Feature scaling
    - Train/validation splits
    - PyTorch DataLoader creation

    Parameters
    ----------
    zone : str
        Bidding zone identifier (e.g., 'no1', 'no2', etc.)
    config : dict
        Configuration dictionary containing data and training parameters
        Expected keys:
        - data.start_date: Training start date
        - data.end_date: Training end date
        - data.validation_start: Validation split date
        - training.batch_size: Batch size for DataLoaders

    Example
    -------
    >>> processor = DataProcessor(zone='no1', config=config)
    >>> train_loader, val_loader, feature_groups, scaler_X, scaler_Y = processor.prepare_torch_datasets()
    """

    def __init__(self, zone, config):
        self.zone = zone
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract config parameters
        self.start_date = config['data']['start_date']
        self.end_date = config['data']['end_date']
        self.validation_start = config['data']['validation_start']
        self.batch_size = config['training']['batch_size']

        # Data paths
        self.data_path = Path(f'src/data/{zone}/merged_dataset_{zone}.parquet')

        # Initialize storage
        self.df = None
        self.feature_groups = {}
        self.scaler_X = {}
        self.scaler_Y = None

    def load_data(self):
        """Load data from parquet file"""
        self.logger.info(f"Loading data from {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load parquet file
        self.df = pd.read_parquet(self.data_path)

        # Ensure datetime index
        if not isinstance(self.df.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

        # Filter date range
        self.df = self.df[self.start_date:self.end_date]

        self.logger.info(f"Loaded {len(self.df)} records from {self.df.index.min()} to {self.df.index.max()}")

        return self.df

    def _create_lagged_features(self, data):
        """
        Create lagged features for time series prediction.

        Creates D-2, D-3, and D-7 lags for each feature.
        D-N means N days ago (24*N hours ago).

        Parameters
        ----------
        data : pd.DataFrame
            Input dataframe with datetime index

        Returns
        -------
        pd.DataFrame
            Dataframe with original and lagged features
        """
        df_lagged = data.copy()

        # Define lag periods (in hours)
        lag_periods = {
            'D-2': 24 * 2,  # 2 days = 48 hours
            'D-3': 24 * 3,  # 3 days = 72 hours
            'D-7': 24 * 7,  # 7 days = 168 hours
        }

        # Get feature columns (exclude target 'premium' if present)
        feature_cols = [col for col in data.columns if col != 'premium']

        # Create lagged features
        for lag_name, lag_hours in lag_periods.items():
            for col in feature_cols:
                lagged_col_name = f"{col}_{lag_name}"
                df_lagged[lagged_col_name] = data[col].shift(lag_hours)

        # Drop rows with NaN values created by lagging
        max_lag = max(lag_periods.values())
        df_lagged = df_lagged.iloc[max_lag:]

        return df_lagged

    def _create_time_features(self, data):
        """
        Create cyclical time features from datetime index.

        Creates sin/cos encodings for:
        - Hour of day (0-23)
        - Day of week (0-6)
        - Month (1-12)

        Also creates binary feature:
        - is_weekend (0/1)

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with DatetimeIndex

        Returns
        -------
        pd.DataFrame
            Dataframe with added time features
        """
        df_time = data.copy()

        # Extract time components
        hours = df_time.index.hour.values
        days_of_week = df_time.index.dayofweek.values
        months = df_time.index.month.values

        # Hour of day (0-23) - cyclical encoding
        df_time['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df_time['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # Day of week (0-6) - cyclical encoding
        df_time['day_of_week_sin'] = np.sin(2 * np.pi * days_of_week / 7)
        df_time['day_of_week_cos'] = np.cos(2 * np.pi * days_of_week / 7)

        # Month (1-12) - cyclical encoding
        df_time['month_sin'] = np.sin(2 * np.pi * (months - 1) / 12)
        df_time['month_cos'] = np.cos(2 * np.pi * (months - 1) / 12)

        # Weekend indicator
        df_time['is_weekend'] = (days_of_week >= 5).astype(float)

        return df_time

    def _create_feature_groups(self, feature_names):
        """
        Create feature group mapping for feature selection.

        Groups features by their base name (e.g., all mFRR_price_up variants)
        to enable group-based feature selection during hyperparameter tuning.

        Parameters
        ----------
        feature_names : list of str
            List of feature column names

        Returns
        -------
        dict
            Dictionary mapping feature group names to {'start_idx', 'size'}
        """
        feature_groups = {}
        current_idx = 0

        # Group features by base name
        # Pattern: base_name or base_name_D-N
        base_feature_dict = {}

        for feature_name in feature_names:
            # Extract base name (remove lag suffix if present)
            if '_D-' in feature_name:
                base_name = feature_name.rsplit('_D-', 1)[0]
            else:
                base_name = feature_name

            if base_name not in base_feature_dict:
                base_feature_dict[base_name] = []
            base_feature_dict[base_name].append(feature_name)

        # Create feature group mapping with indices
        for base_name, features in base_feature_dict.items():
            n_features = len(features)
            feature_groups[base_name] = {
                'start_idx': current_idx,
                'size': n_features,
                'features': features
            }
            current_idx += n_features

        return feature_groups

    def prepare_torch_datasets(self):
        """
        Prepare PyTorch DataLoaders for training and validation.

        Complete pipeline:
        1. Load data
        2. Create lagged features
        3. Create time features
        4. Reshape into 24-hour windows (days x 24*features)
        5. Split train/validation
        6. Scale features and target
        7. Create feature groups
        8. Convert to PyTorch tensors
        9. Create DataLoaders

        Returns
        -------
        tuple
            (train_loader, val_loader, feature_groups, scaler_X, scaler_Y)
            - train_loader: DataLoader for training data
            - val_loader: DataLoader for validation data
            - feature_groups: dict mapping feature names to indices
            - scaler_X: dict of StandardScaler objects for features
            - scaler_Y: StandardScaler for target variable
        """
        # Load data
        if self.df is None:
            self.load_data()

        # Create lagged features
        self.logger.info("Creating lagged features...")
        df_lagged = self._create_lagged_features(self.df)

        # Create time features
        self.logger.info("Creating time features...")
        df_features = self._create_time_features(df_lagged)

        if 'premium' not in df_features.columns:
            raise ValueError("Target variable 'premium' not found in data")

        # Reshape into 24-hour windows (similar to prediction.py)
        forecast_horizon = 24
        premium_data = df_features['premium'].values
        feature_df = df_features.drop(columns=['premium'])

        # Calculate number of complete days
        num_hours = len(df_features)
        num_days = num_hours // forecast_horizon

        if num_hours % forecast_horizon != 0:
            # Trim to complete days
            trim_length = num_days * forecast_horizon
            premium_data = premium_data[:trim_length]
            feature_df = feature_df.iloc[:trim_length]
            df_features = df_features.iloc[:trim_length]

        # Create rolling window matrices
        # Y: (num_days, 24) - each row is 24 hours of premium
        # X: (num_days, num_features * 24) - each row is 24 hours of all features flattened
        Y = np.zeros((num_days, forecast_horizon))
        X = np.zeros((num_days, forecast_horizon * len(feature_df.columns)))

        # Fill X and Y
        for d in range(num_days):
            start_idx = d * forecast_horizon
            end_idx = start_idx + forecast_horizon

            # Target values for this day
            Y[d, :] = premium_data[start_idx:end_idx]

            # Feature values for this day (flattened across all features)
            for i, feature_name in enumerate(feature_df.columns):
                X[d, i * forecast_horizon:(i + 1) * forecast_horizon] = feature_df.iloc[start_idx:end_idx, i].values

        # Create feature groups (each feature now occupies 24 consecutive indices)
        feature_names = feature_df.columns.tolist()
        self.feature_groups = {}
        for i, feature_name in enumerate(feature_names):
            self.feature_groups[feature_name] = {
                'start_idx': i * forecast_horizon,
                'size': forecast_horizon
            }

        self.logger.info(f"Created {len(self.feature_groups)} feature groups")

        # Split into train/validation based on date
        # Find which day corresponds to validation_start
        validation_start_dt = pd.Timestamp(self.validation_start, tz=df_features.index.tz)
        train_days = 0
        for d in range(num_days):
            day_start_idx = d * forecast_horizon
            day_timestamp = df_features.index[day_start_idx]
            if day_timestamp >= validation_start_dt:
                train_days = d
                break
        else:
            # If validation date not found, use 80/20 split
            train_days = int(num_days * 0.8)

        X_train = X[:train_days]
        Y_train = Y[:train_days]
        X_val = X[train_days:]
        Y_val = Y[train_days:]

        self.logger.info(f"Train size: {len(X_train)} days, Validation size: {len(X_val)} days")

        # Scale features - each feature group (24 hours) scaled together
        self.logger.info("Scaling features...")
        X_train_scaled = np.zeros_like(X_train)
        X_val_scaled = np.zeros_like(X_val)

        for feature_name in feature_names:
            start_idx = self.feature_groups[feature_name]['start_idx']
            size = self.feature_groups[feature_name]['size']

            scaler = StandardScaler()
            # Fit on training data (all 24 hours together)
            X_train_scaled[:, start_idx:start_idx + size] = scaler.fit_transform(
                X_train[:, start_idx:start_idx + size]
            )
            # Transform validation data
            X_val_scaled[:, start_idx:start_idx + size] = scaler.transform(
                X_val[:, start_idx:start_idx + size]
            )
            self.scaler_X[feature_name] = scaler

        # Scale target - each day's 24 hours scaled together
        self.scaler_Y = StandardScaler()
        Y_train_scaled = self.scaler_Y.fit_transform(Y_train)
        Y_val_scaled = self.scaler_Y.transform(Y_val)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        Y_train_tensor = torch.FloatTensor(Y_train_scaled)
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        Y_val_tensor = torch.FloatTensor(Y_val_scaled)

        # Create TensorDatasets
        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.logger.info("Data preparation complete")

        return train_loader, val_loader, self.feature_groups, self.scaler_X, self.scaler_Y
