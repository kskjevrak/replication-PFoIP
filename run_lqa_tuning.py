#!/usr/bin/env python
"""
run_linear_quantile_tuning.py - Script to run Linear Quantile Regression hyperparameter tuning
Compatible with cluster execution

Usage:
    # Single zone and run
    python run_linear_quantile_tuning.py no1 1
    
    # All zones, single run
    python run_linear_quantile_tuning.py --all-zones 1
    
    # Single zone, all runs  
    python run_linear_quantile_tuning.py no1 --all-runs
    
    # All zones, all runs
    python run_linear_quantile_tuning.py --all
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.linear_quantile_tuner import LinearQuantileTuner

# Default parameters
ZONES = ['no1', 'no2', 'no3', 'no4', 'no5']
RUNS = ['1', '2', '3', '4']
CONFIG_PATH = 'config/default_config.yml'

def run_single_tuning(zone, run_id, silent=False):
    """Run tuning for a single zone and run combination"""
    
    if not silent:
        print(f"Zone: {zone.upper()}, Run ID: {run_id}")
        print(f"Output: ./results/models/linear_quantile_{zone}_{run_id}/")
    
    try:
        # Initialize tuner
        tuner = LinearQuantileTuner(
            config_path=CONFIG_PATH,
            zone=zone,
            run_id=run_id
        )
        
        # Run optimization
        best_params = tuner.run_optimization(
            n_trials=25,
            n_jobs=1
        )
        
        if not silent:
            print(f"Completed: {zone.upper()} run {run_id}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"Interrupted: {zone} run {run_id}")
        return False
    except Exception as e:
        print(f"Error in {zone} run {run_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Linear Quantile Regression Hyperparameter Tuning')
    
    # Positional arguments (for backward compatibility)
    parser.add_argument('zone', nargs='?', help='Market zone (no1, no2, no3, no4, no5)')
    parser.add_argument('run_id', nargs='?', help='Run identifier (1, 2, 3, 4)')
    
    # Batch mode flags
    parser.add_argument('--all', action='store_true', help='Run all zones and all runs')
    parser.add_argument('--all-zones', metavar='RUN_ID', help='Run all zones for specified run ID')
    parser.add_argument('--all-runs', action='store_true', help='Run all runs for specified zone')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    
    args = parser.parse_args()
    
    # Set environment variables for better performance
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    # Determine execution mode and build job list
    jobs = []
    
    if args.all:
        jobs = [(zone, 1) for zone in ZONES]
        print(f"Running Linear Quantile tuning: {len(ZONES)} zones x 1 runs = {len(jobs)} total jobs")
        run_id = "1"
        
    else:
        # Single zone, single run
        zone = args.zone or 'no1'
        run_id = args.run_id or '1'
        
        if zone not in ZONES:
            parser.error(f"Zone must be one of {ZONES}")
        if run_id not in RUNS:
            parser.error(f"Run ID must be one of {RUNS}")
            
        jobs = [(zone, run_id)]
    
    # Execute jobs
    successful_jobs = 0
    failed_jobs = 0
    
    for i, (zone, run_id) in enumerate(jobs, 1):
        if len(jobs) > 1:
            print(f"Job {i}/{len(jobs)}: {zone.upper()} run {run_id}")
        
        success = run_single_tuning(zone, run_id, silent=(len(jobs) > 1))
        
        if success:
            successful_jobs += 1
        else:
            failed_jobs += 1
    
    # Final summary for batch jobs
    if len(jobs) > 1:
        print(f"Completed: {successful_jobs}/{len(jobs)} successful")
        if failed_jobs > 0:
            print(f"Failed: {failed_jobs} jobs")

if __name__ == "__main__":
    main()