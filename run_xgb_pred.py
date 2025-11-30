#!/usr/bin/env python
"""
run_xgb_pred.py - Script to run XGBoost quantile predictions for a specified date range
Compatible with cluster execution

Usage:
    python run_xgb_pred.py [zone] [run_id] [--workers WORKERS]
"""
import os
import sys
import torch
import logging
import argparse
import pandas as pd
import numpy as np
import multiprocessing as mp
from datetime import datetime, timedelta
from functools import partial
import json
import joblib
import yaml
from pathlib import Path

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.training.xgb_pred import XGBoostDailyPredictor


def process_day(prediction_date, zone, run_id, logger):
    """Process predictions for a single day"""
    try:
        logger.info(f"Starting XGBoost prediction for {prediction_date.strftime('%Y-%m-%d')}")
        
        # Create predictor
        predictor = XGBoostDailyPredictor(zone=zone, run_id=run_id)
        
        # Run prediction
        success = predictor.run(prediction_date)
        
        if success:
            logger.info(f"XGBoost prediction for {prediction_date.strftime('%Y-%m-%d')} completed successfully")
            return (prediction_date.strftime('%Y-%m-%d'), True)
        else:
            logger.error(f"XGBoost prediction for {prediction_date.strftime('%Y-%m-%d')} failed")
            return (prediction_date.strftime('%Y-%m-%d'), False)
            
    except Exception as e:
        logger.error(f"Error predicting for {prediction_date.strftime('%Y-%m-%d')}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return (prediction_date.strftime('%Y-%m-%d'), False)

def main():
    """Main function to run XGBoost predictions for a date range"""
    parser = argparse.ArgumentParser(description="XGBoost quantile prediction script")
    parser.add_argument("zone", nargs="?", default="no1", 
                        help="Market zone (e.g., no1, no2)")
    parser.add_argument("run_id", nargs="?", default="1", 
                        help="Run ID")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")
    
    args = parser.parse_args()

    # Set up logging
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"xgboost_prediction_{args.zone}_{args.run_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("xgboost_prediction")
    logger.info(f"Starting XGBoost predictions for zone: {args.zone}, run_id: {args.run_id}")
    
    # Define date range
    start_date = pd.Timestamp("2024-04-26", tz="Europe/Oslo")
    end_date = pd.Timestamp("2025-04-25", tz="Europe/Oslo")
    
    logger.info(f"Prediction date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Generate list of dates
    current_date = start_date
    date_list = []
    
    while current_date <= end_date:
        date_list.append(current_date)
        current_date += timedelta(days=1)
    
    logger.info(f"Total days to predict: {len(date_list)}")
    
    # Create partial function with fixed arguments
    worker_func = partial(process_day, zone=args.zone, run_id=args.run_id, logger=logger)
    
    # Process with multiprocessing
    num_workers = min(args.workers, len(date_list))
    logger.info(f"Using {num_workers} worker processes")
    
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(worker_func, date_list)
    
    # Process results
    successful_days = [date for date, success in results if success]
    failed_days = [date for date, success in results if not success]
    
    # Log summary
    logger.info(f"XGBoost prediction run complete. Summary:")
    logger.info(f"Total days: {len(date_list)}")
    logger.info(f"Successful predictions: {len(successful_days)}")
    logger.info(f"Failed predictions: {len(failed_days)}")
    
    if failed_days:
        logger.info(f"Failed dates: {', '.join(failed_days)}")

if __name__ == "__main__":
    # Set thread limits
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    
    # Set multiprocessing start method
    mp.set_start_method('spawn')
    
    main()