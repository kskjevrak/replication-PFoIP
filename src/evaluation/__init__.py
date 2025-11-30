"""
Evaluation metrics and visualization tools for probabilistic forecasts.

This module provides comprehensive evaluation capabilities for assessing
the quality of probabilistic forecasts including scoring rules, calibration
tests, and visualization utilities.

Modules
-------
metrics.py
    Probabilistic forecast evaluation metrics including:
    - Continuous Ranked Probability Score (CRPS)
    - Quantile Score (pinball loss)
    - Interval Score
    - Coverage metrics
    - Winkler Score
    - Energy Score
    Functions for loading actual values and calculating metrics across
    date ranges.

visualization.py
    Plotting utilities for forecast visualization:
    - Probabilistic fan charts
    - Prediction intervals
    - Quantile plots
    - Actual vs predicted comparisons
    - Residual analysis
    - Distribution plots

heatmapping.py
    Heatmap generation for analyzing forecast performance:
    - Temporal performance patterns
    - Hour-of-day analysis
    - Day-of-week patterns
    - Seasonal variations
    - Error distribution heatmaps

Key Functions
-------------
calculate_crps (metrics.py)
    Calculate Continuous Ranked Probability Score

calculate_quantile_score (metrics.py)
    Calculate quantile score (pinball loss)

calculate_interval_score (metrics.py)
    Calculate interval score for prediction intervals

calculate_all_metrics (metrics.py)
    Comprehensive metric calculation for all dates

plot_probabilistic_forecast (visualization.py)
    Create fan chart visualization

create_performance_heatmap (heatmapping.py)
    Generate heatmap of forecast errors

Usage Example
-------------
>>> from src.evaluation.metrics import calculate_all_metrics
>>> from src.evaluation.visualization import plot_probabilistic_forecast
>>>
>>> # Calculate metrics
>>> metrics = calculate_all_metrics(
...     distribution='jsu',
...     nr='test_001',
...     zone='no1',
...     date_range=('2024-04-26', '2024-05-26')
... )
>>>
>>> # Visualize forecast
>>> plot_probabilistic_forecast(
...     predictions=forecasts,
...     actuals=actual_values,
...     date='2024-04-26',
...     zone='no1'
... )
"""
