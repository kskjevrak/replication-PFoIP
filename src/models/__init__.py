"""
Model architectures and probability distributions for imbalance price forecasting.

This module contains the core model implementations for probabilistic forecasting
of Norwegian electricity imbalance prices.

Modules
-------
neural_nets.py
    Deep Distributional Neural Network (DDNN) architectures for parametric
    probabilistic forecasting. Includes multi-output networks that predict
    distribution parameters for 24-hour ahead forecasts.

distributions.py
    Probability distribution implementations including:
    - Normal (Gaussian) distribution
    - Johnson's SU (JSU) distribution
    - Student's t distribution
    Each with methods for PDF, CDF, quantile functions, and sampling.

layers.py
    Custom PyTorch layers used in neural network architectures:
    - Feature selection layers
    - Distribution parameter output layers
    - Custom activation functions

linear_quantile.py
    Linear Quantile Regression (LQR) models for non-parametric
    quantile estimation at specified probability levels.

xgboost_quantile.py
    XGBoost-based quantile regression models using gradient boosting
    trees for capturing non-linear patterns in quantile forecasting.

Key Classes
-----------
DDNN (neural_nets.py)
    Main deep distributional neural network class

NormalDistribution, JSUDistribution, StudentTDistribution (distributions.py)
    Probability distribution implementations

LinearQuantileRegressor (linear_quantile.py)
    Linear quantile regression model

XGBoostQuantileRegressor (xgboost_quantile.py)
    XGBoost quantile regression model

Usage Example
-------------
>>> from src.models.neural_nets import DDNN
>>> from src.models.distributions import JSUDistribution
>>>
>>> model = DDNN(input_size=240, hidden_layers=[256, 128], distribution='jsu')
>>> # Train model...
>>> predictions = model(features)
"""
