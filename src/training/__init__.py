"""
Training pipelines, hyperparameter tuning, and prediction workflows.

This module contains all training-related functionality including hyperparameter
optimization using Optuna, model training, and prediction generation.

Modules
-------
optuna_tuner.py
    Hyperparameter tuning for Deep Distributional Neural Networks (DDNN)
    using Optuna. Includes cross-validation and feature selection optimization.

linear_quantile_tuner.py
    Hyperparameter tuning for Linear Quantile Regression (LQR) models.

xgboost_tuner.py
    Hyperparameter tuning for XGBoost quantile regression models.

prediction.py
    DDNN prediction pipeline for generating probabilistic forecasts.
    Loads trained models, prepares features, and generates distribution
    predictions for specified dates.

linear_quantile_prediction.py
    LQR prediction pipeline for quantile forecasting.

xgb_pred.py
    XGBoost prediction pipeline for quantile forecasting.

rolling_pred.py
    Rolling prediction workflow for generating forecasts across
    multiple dates in sequence.

Key Classes
-----------
OptunaHPTuner (optuna_tuner.py)
    Hyperparameter tuner for DDNN using Optuna optimization

LinearQuantileTuner (linear_quantile_tuner.py)
    Hyperparameter tuner for LQR models

XGBoostTuner (xgboost_tuner.py)
    Hyperparameter tuner for XGBoost models

Predictor (prediction.py)
    Main prediction class for DDNN forecasting

LinearQuantilePredictor (linear_quantile_prediction.py)
    Prediction class for LQR forecasting

XGBoostPredictor (xgb_pred.py)
    Prediction class for XGBoost forecasting

Workflow
--------
1. Tune hyperparameters:
   >>> from src.training.optuna_tuner import OptunaHPTuner
   >>> tuner = OptunaHPTuner(config_path='config/default_config.yml',
   ...                       zone='no1', distribution='jsu', run_id='test')
   >>> best_params = tuner.run_optimization(n_trials=128)

2. Make predictions:
   >>> from src.training.prediction import Predictor
   >>> predictor = Predictor(zone='no1', distribution='jsu', run_id='test')
   >>> predictor.predict_for_date('2024-04-26')

3. Results saved to:
   results/forecasts/{zone}_{distribution}_{run_id}/
   results/models/{zone}_{distribution}_{run_id}/
"""
