"""
Probabilistic Forecasting of Imbalance Prices (PFoIP)

This package implements probabilistic forecasting models for Norwegian electricity
imbalance prices across five bidding zones (NO1-NO5).

Main Components
---------------
- models: Neural network architectures and probability distributions
- training: Model training, hyperparameter tuning, and prediction pipelines
- evaluation: Performance metrics and visualization tools
- utils: Utility functions for ensemble methods and data processing
- data: Data loading and synthetic data generation

Models Implemented
------------------
1. Deep Distributional Neural Networks (DDNN)
   - Parametric probabilistic forecasting
   - Supports Normal, Johnson's SU, and Student's t distributions

2. Linear Quantile Regression (LQR)
   - Non-parametric quantile estimation
   - Fast training and inference

3. XGBoost Quantile Regression
   - Gradient boosting for quantile forecasting
   - Handles non-linear patterns

Usage Example
-------------
>>> from src.training.prediction import Predictor
>>> predictor = Predictor(zone='no1', distribution='jsu', run_id='test_001')
>>> predictor.predict_for_date('2024-04-26')

For detailed usage, see README.md and REPLICATION.md
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@domain.com"
