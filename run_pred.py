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
    parser = argparse.ArgumentParser(
        description="mFRR price prediction script for date range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using positional arguments
  python run_pred.py no1 jsu test --start-date 2024-04-26 --end-date 2024-04-26

  # Using named arguments
  python run_pred.py --zone no1 --distribution jsu --run-id test --start-date 2024-04-26 --end-date 2024-04-26

  # Multiple workers for faster processing
  python run_pred.py no1 jsu test --start-date 2024-04-26 --end-date 2024-05-01 --workers 4
        """
    )

    parser.add_argument("zone", nargs="?",
                        help="Market zone (e.g., no1, no2, no3, no4, no5)")
    parser.add_argument("distribution", nargs="?",
                        help="Probability distribution (jsu, normal, skewt)")
    parser.add_argument("run_id", nargs="?",
                        help="Run identifier matching the trained model")

    parser.add_argument("--zone", "-z", dest="zone_named", type=str,
                        help="Market zone (named argument)")
    parser.add_argument("--distribution", "-d", dest="distribution_named", type=str,
                        help="Probability distribution (named argument)")
    parser.add_argument("--run-id", "-r", dest="run_id_named", type=str,
                        help="Run identifier (named argument)")

    parser.add_argument("--start-date", type=str, required=True,
                        help="Start date for predictions (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="End date for predictions (YYYY-MM-DD)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes (default: 1)")

    args = parser.parse_args()

    # Handle positional vs named arguments
    zone = args.zone_named or args.zone
    distribution = args.distribution_named or args.distribution
    run_id = args.run_id_named or args.run_id

    # Validate required arguments
    if not zone:
        parser.error("zone is required")
    if not distribution:
        parser.error("distribution is required")
    if not run_id:
        parser.error("run_id is required")

    # Normalize distribution name
    distribution_map = {
        'normal': 'Normal',
        'jsu': 'JSU',
        'skewt': 'skewt',
        'studentt': 'skewt',
        't': 'skewt'
    }
    distribution_lower = distribution.lower()
    if distribution_lower not in distribution_map:
        parser.error(f"distribution must be one of {list(distribution_map.keys())}, got: {distribution}")

    distribution = distribution_map[distribution_lower]

    # Set up logging
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"prediction_{zone}_{distribution.lower()}_{run_id}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger("prediction")
    logger.info(f"Starting predictions for zone: {zone}, distribution: {distribution}, run_id: {run_id}")

    # Parse date range
    try:
        start_date = pd.Timestamp(args.start_date, tz="Europe/Oslo")
        end_date = pd.Timestamp(args.end_date, tz="Europe/Oslo")
    except Exception as e:
        parser.error(f"Invalid date format. Use YYYY-MM-DD. Error: {e}")

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
        worker_func = partial(process_day, zone=zone, distribution=distribution, run_id=run_id, logger=logger)

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
                'zone': zone,
                'distribution': distribution,
                'run_id': run_id
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