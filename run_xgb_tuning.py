#!/usr/bin/env python
"""
run_xgboost_tuning.py - Script to run XGBoost hyperparameter tuning with command-line arguments
Compatible with cluster execution

Usage:
    python run_xgboost_tuning.py [zone] [run_id]
"""
import os
import sys
import logging
from datetime import datetime

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.xgboost_tuner import XGBoostTuner

# Default parameters
ZONES = ['no1', 'no2', 'no3', 'no4', 'no5']
CONFIG_PATH = 'config/default_config.yml'

def main():
    # Parse command line arguments
    zone = 'no1'
    run_id = '1'

    if len(sys.argv) > 1:
        zone = sys.argv[1].lower()
    if len(sys.argv) > 2:
        run_id = sys.argv[2]
    
    # Validate inputs
    if zone not in ZONES:
        sys.exit(f'Error: Zone must be one of {ZONES}')

    # Display configuration and prompt for confirmation
    print("=" * 60)
    print("XGBoost Quantile Regression Hyperparameter Tuning")
    print("=" * 60)
    print(f'Zone: {zone.upper()}')
    print(f'Run ID: {run_id}')
    print(f'Model type: XGBoost Quantile Regression')
    print(f'Config: {CONFIG_PATH}')
    print(f'Output directory: ./results/models/xgboost_{zone}_{run_id}/')
    print("=" * 60)
    
    inp = input('Press enter to continue or any key to exit...')
    if inp != '':
        sys.exit('Tuning process interrupted - exiting...')
    
    # Set environment variables for better performance
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    print(f"\nStarting XGBoost hyperparameter tuning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Logs will be saved to: ./results/logs/xgboost_{zone}_{run_id}.log")
    
    try:
        # Initialize tuner
        tuner = XGBoostTuner(
            config_path=CONFIG_PATH,
            zone=zone,
            run_id=run_id
        )
        
        # Run optimization with reasonable number of trials for XGBoost
        print("Starting optimization process...")
        best_params = tuner.run_optimization(
            n_trials=100,  # XGBoost typically needs fewer trials than neural networks
            n_jobs=1       # Set to 1 for cluster compatibility, can be increased locally
        )
        
        print(f"Complete! Best parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
        '''  
        print(f"\nResults saved to: ./results/models/xgboost_{zone}_{run_id}/")
        print(f"  - best_params.yaml: Hyperparameter configuration")
        print(f"  - model files: Trained XGBoost models (one per hour)")
        print(f"  - feature_mask.npy: Selected features")
        
        print(f"\nScalers saved to: ./results/scalers/xgboost_{zone}_{run_id}/")
        print(f"  - scaler_X_*.joblib: Feature scalers")
        print(f"  - scaler_Y.joblib: Target scaler")
        
        print(f"\nNext steps:")
        print(f"  1. Run predictions: python run_xgboost_pred.py {zone} {run_id}")
        print(f"  2. Evaluate results: python -m src.evaluation.metrics --distributions xgboost")
        '''
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    print()
    main()