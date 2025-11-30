#!/usr/bin/env python
"""
run_tuning.py - Script to run hyperparameter tuning with command-line arguments
Compatible with cluster execution

Usage:
    python scripts/run_tuning.py [zone] [distribution] [run_id]
"""
import os
import sys
import logging
from datetime import datetime

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print()
from src.training.optuna_tuner import OptunaHPTuner

# Default parameters
ZONES = ['no1', 'no2', 'no3', 'no4', 'no5']
DISTRIBUTIONS = ['Normal', 'JSU', 'normal', 'jsu', 'skewt']
CONFIG_PATH = 'config/default_config.yml'

def main():
    # Parse command line arguments
    zone = 'no1'
    distribution = 'jsu'
    run_id = 'DEBUGtest1'

    if len(sys.argv) > 1:
        zone = sys.argv[1].lower()
    if len(sys.argv) > 2:
        distribution = sys.argv[2].lower()
    if len(sys.argv) > 3:
        run_id = sys.argv[3]
    
    # Validate inputs
    if zone not in ZONES:
        sys.exit(f'Error: Zone must be one of {ZONES}')
    if distribution not in DISTRIBUTIONS:
        sys.exit(f'Error: Distribution must be one of {DISTRIBUTIONS}')

    # Display configuration and prompt for confirmation
    print(f'Zone: {zone}, distribution: {distribution}, run_id: {run_id}')
    inp = input('Press enter to continue or any key to exit...')
    if inp != '':
        sys.exit('Tuning process interrupted - exiting...')
    
    os.environ["OMP_NUM_THREADS"] = "1"

    # Initialize tuner
    tuner = OptunaHPTuner(
        config_path=CONFIG_PATH,
        zone=zone,
        distribution=distribution,
        run_id=run_id
    )
    
    # Run optimization
    best_params = tuner.run_optimization(n_trials=128)
    print(f"Optimization complete! Best parameters saved to results/models/{zone}_{distribution.lower()}_{run_id}/")

if __name__ == "__main__":
    main()