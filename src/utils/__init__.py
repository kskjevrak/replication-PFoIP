"""
Utility functions and helper modules for forecasting workflows.

This module contains utility functions for ensemble methods, data processing,
and other supporting functionality used across the forecasting pipeline.

Modules
-------
ensemble.py
    Ensemble methods for combining multiple forecasts:
    - Simple averaging
    - Weighted averaging
    - Quantile averaging
    - Distribution mixing
    Functions for model combination and ensemble prediction generation.

Key Functions
-------------
create_ensemble (ensemble.py)
    Combine multiple model predictions into ensemble forecast

weighted_quantile_average (ensemble.py)
    Weighted averaging of quantile forecasts

mix_distributions (ensemble.py)
    Combine probability distributions from multiple models

Usage Example
-------------
>>> from src.utils.ensemble import create_ensemble
>>>
>>> # Combine forecasts from multiple models
>>> ensemble_forecast = create_ensemble(
...     forecasts=[model1_forecast, model2_forecast, model3_forecast],
...     weights=[0.5, 0.3, 0.2],
...     method='weighted_average'
... )
"""
