#!/usr/bin/env python
"""
run_pred.py - Script to run mFRR price predictions for a specified date range
Compatible with cluster execution

Usage:
    python scripts/run_pred.py [zone] [distribution] [run_id] [--config CONFIG_PATH]
"""
import os
import sys
import torch
import logging
import argparse
import pandas as pd
import multiprocessing as mp
from datetime import datetime, timedelta
from functools import partial


# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.training.prediction import DailyMarketPredictor


# Run predictions for each day
def process_day(prediction_date, zone, distribution, run_id, logger):
    try:
        logger.info(f"Starting prediction for {prediction_date.strftime('%Y-%m-%d')}")
        
        # Create predictor for this specific day
        predictor = DailyMarketPredictor(
            zone=zone,
            distribution=distribution,
            run_id=run_id
        )
        
        # Override the automatic date detection to use our specific date
        predictor.prediction_date = prediction_date
        
        # Run prediction
        forecast = predictor.run()
        
        if forecast is not None:
            # Generate summary report
            summary = predictor.generate_summary_report()
            logger.info(f"Prediction for {prediction_date.strftime('%Y-%m-%d')} completed successfully")
            return (prediction_date.strftime('%Y-%m-%d'), True)
        else:
            logger.error(f"Prediction for {prediction_date.strftime('%Y-%m-%d')} failed")
            return (prediction_date.strftime('%Y-%m-%d'), False)
    except Exception as e:
        logger.error(f"Error predicting for {prediction_date.strftime('%Y-%m-%d')}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return (prediction_date.strftime('%Y-%m-%d'), False)

def main():
    """Main function to run mFRR price predictions for a date range"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="mFRR price prediction script for date range")
    parser.add_argument("zone", nargs="?", default="no1", 
                        help="Market zone (e.g., no1, no2)")
    parser.add_argument("distribution", nargs="?", default="JSU", choices=["JSU", "Normal", "skewt"], 
                        help="Probability distribution")
    parser.add_argument("run_id", nargs="?", default="1", 
                        help="Specific run ID (uses latest if not specified)")
    parser.add_argument("--workers", type=int, default=1,
                    help="Number of worker processes (default: 8)")
    
    args = parser.parse_args()

    # Set up logging
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"prediction_{args.zone}_{args.distribution.lower()}_{args.run_id}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("prediction")
    logger.info(f"Starting predictions for zone: {args.zone}, distribution: {args.distribution}, run_id: {args.run_id}")
    
    # Parse date range
    start_date = pd.Timestamp("2024-04-26", tz="Europe/Oslo")
    end_date = pd.Timestamp("2025-04-25", tz="Europe/Oslo")  # Full circle days later
    #end_date = pd.Timestamp("2024-05-10", tz="Europe/Oslo")  # Full circle days later
    
    logger.info(f"Prediction date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Import prediction module
    try:
     
        # Generate list of dates to predict
        current_date = start_date
        date_list = []
        
        while current_date <= end_date:
            current_date = current_date.tz_convert("Europe/Oslo")
            date_list.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"Total days to predict: {len(date_list)}")
        
        # Track results
        successful_days = []
        failed_days = []
        
        # Create a partial function with fixed arguments
        worker_func = partial(process_day, zone=args.zone, distribution=args.distribution, run_id=args.run_id, logger=logger)

        # Determine number of processes to use
        num_workers = min(args.workers, len(date_list))
        logger.info(f"Using {num_workers} worker processes")

        # Start multiprocessing pool and execute
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(worker_func, date_list)

        # Process results
        successful_days = [date for date, success in results if success]
        failed_days = [date for date, success in results if not success]

        # Log final summary
        logger.info(f"Prediction run complete. Summary:")
        logger.info(f"Total days: {len(date_list)}")
        logger.info(f"Successful predictions: {len(successful_days)}")
        logger.info(f"Failed predictions: {len(failed_days)}")
        
        if failed_days:
            logger.info(f"Failed dates: {', '.join(failed_days)}")
        
        # Save results to file
        results_summary = {
            'total_days': len(date_list),
            'successful_days': successful_days,
            'failed_days': failed_days,
            'parameters': {
                'zone': args.zone,
                'distribution': args.distribution,
                'run_id': args.run_id
            }
        }
        
        import json
        #summary_path = os.path.join("results", f"prediction_summary_{args.zone}_{args.distribution}_{args.run_id}.json")
        #with open(summary_path, 'w') as f:
        #    json.dump(results_summary, f, indent=2)
        
        #logger.info(f"Results summary saved to {summary_path}")
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you have installed the package in development mode using 'pip install -e .'")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during prediction process: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":    
    # Set OpenMP thread count for better CPU utilization
    # Set thread limits for numeric libraries
    os.environ["OMP_NUM_THREADS"] = "1"  # OpenMP
    os.environ["MKL_NUM_THREADS"] = "1"  # Intel MKL
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # NumExpr
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # OpenBLAS

    torch.set_num_threads(1)
    # Set multiprocessing start method
    mp.set_start_method('spawn')

    main()