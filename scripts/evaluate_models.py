#!/usr/bin/env python
"""
evaluate_models.py - Generate Table 2: Aggregate forecasting performance metrics

This script evaluates all models across all Norwegian bidding zones and computes:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- CRPS (Continuous Ranked Probability Score)
- Pinball Loss at 10% quantile
- Winkler Score for 90% prediction intervals

Output: CSV file with zone-averaged metrics, color-coded from best to worst

Usage:
    python scripts/evaluate_models.py --zones no1 no2 no3 no4 no5 --output outputs/table2_performance.csv
    python scripts/evaluate_models.py --zones no2 --output outputs/table2_no2.csv

Author: Knut Skjevrak
Last updated: February 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import json
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import (
    calculate_crps,
    calculate_mae,
    calculate_rmse,
    calculate_pinball_loss,
    calculate_winkler_score,
    load_samples
)

def load_forecast_data(zone, distribution, run_id, test_period):
    """
    Load forecast predictions and actual values for evaluation

    Parameters:
    -----------
    zone : str
        Bidding zone (no1, no2, no3, no4, no5)
    distribution : str
        Distribution type (jsu, normal, skewt, lqr, xgb)
    run_id : str
        Model run identifier
    test_period : tuple
        (start_date, end_date) for evaluation

    Returns:
    --------
    dict: {date: {'predictions': samples, 'actual': values}}
    """
    results_dir = Path('results/forecasts')

    # Determine forecast directory based on model type
    if distribution in ['jsu', 'normal', 'skewt']:
        forecast_dir = results_dir / f'df_forecasts_{zone}_{distribution}_{run_id}'
        param_dir = results_dir / f'distparams_{zone}_{distribution}_{run_id}'
    else:
        forecast_dir = results_dir / f'{distribution}_{zone}_{run_id}'
        param_dir = None

    data = {}

    # Get all forecast files in date range
    start_date, end_date = pd.to_datetime(test_period[0]), pd.to_datetime(test_period[1])
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')

        # Load forecast file
        forecast_file = forecast_dir / f'{date_str}_forecast.csv'
        if not forecast_file.exists():
            continue

        try:
            df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)

            # Extract actual values
            if 'realized' in df.columns:
                actual = df['realized'].values
            else:
                continue  # Skip if no actual values

            # Generate prediction samples
            if distribution in ['jsu', 'normal', 'skewt']:
                # Load from distribution parameters
                param_file = param_dir / f'{date_str}.json'
                if param_file.exists():
                    with open(param_file, 'r') as f:
                        params = json.load(f)

                    # Generate samples (1000 samples per hour)
                    n_samples = 1000
                    if distribution == 'normal':
                        loc = np.array(params['loc'])
                        scale = np.array(params['scale'])
                        samples = np.random.normal(loc, scale, size=(n_samples, 24))
                    elif distribution == 'jsu':
                        # For JSU, use quantiles if parameters are unstable
                        samples = np.column_stack([
                            df[f'q{q:02d}'].values for q in range(1, 100)
                        ]).T
                    elif distribution == 'skewt':
                        # Use approximation via quantiles
                        samples = np.column_stack([
                            df[f'q{q:02d}'].values for q in range(1, 100)
                        ]).T
                else:
                    continue
            else:
                # For LQR and XGBoost, use quantiles
                quantile_cols = [c for c in df.columns if c.startswith('q')]
                if quantile_cols:
                    samples = df[quantile_cols].values.T
                else:
                    continue

            data[date_str] = {
                'predictions': samples,
                'actual': actual
            }

        except Exception as e:
            print(f"Error loading {date_str} for {zone}/{distribution}: {e}")
            continue

    return data

def evaluate_model(zone, distribution, run_id, test_period):
    """
    Evaluate a single model configuration

    Returns:
    --------
    dict: Evaluation metrics
    """
    data = load_forecast_data(zone, distribution, run_id, test_period)

    if not data:
        print(f"No data found for {zone}/{distribution}/{run_id}")
        return None

    # Aggregate metrics across all dates
    all_crps = []
    all_mae = []
    all_rmse = []
    all_pinball = []
    all_winkler = []

    for date, values in data.items():
        predictions = values['predictions']
        actual = values['actual']

        if len(actual) != 24:
            continue

        # Calculate metrics
        crps = calculate_crps(predictions, actual)
        mae = calculate_mae(predictions, actual)
        rmse = calculate_rmse(predictions, actual)
        pinball = calculate_pinball_loss(predictions, actual, quantiles=[0.1])
        winkler = calculate_winkler_score(predictions, actual, alpha=0.9)

        all_crps.append(np.mean(crps))
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_pinball.append(pinball['q10'])
        all_winkler.append(winkler)

    if not all_crps:
        return None

    return {
        'MAE': np.mean(all_mae),
        'RMSE': np.mean(all_rmse),
        'CRPS': np.mean(all_crps),
        'Pinball_10': np.mean(all_pinball),
        'Winkler': np.mean(all_winkler)
    }

def create_performance_table(zones, test_period, output_file):
    """
    Create Table 2: Aggregate forecasting performance

    Parameters:
    -----------
    zones : list
        List of bidding zones to evaluate
    test_period : tuple
        (start_date, end_date)
    output_file : str
        Path to output CSV file
    """
    # Define models to evaluate
    models = [
        ('Naive', 'naive', 'baseline'),
        ('Exponential Smoothing', 'exp_smooth', 'baseline'),
        ('Linear QR', 'lqr', 'replication_lqr'),
        ('XGBoost QR', 'xgb', 'replication_xgb'),
        ('DDNN-Normal', 'normal', 'replication_normal'),
        ('DDNN-JSU', 'jsu', 'replication_jsu'),
        ('DDNN-Skewed-t', 'skewt', 'replication_skewt')
    ]

    results = []

    print(f"Evaluating models across {len(zones)} zones...")
    print(f"Test period: {test_period[0]} to {test_period[1]}")
    print("")

    for model_name, distribution, run_id_template in tqdm(models, desc="Models"):
        zone_metrics = []

        for zone in zones:
            # Construct run_id for this zone
            if run_id_template == 'baseline':
                run_id = run_id_template
            else:
                run_id = f'{run_id_template}_{zone}'

            metrics = evaluate_model(zone, distribution, run_id, test_period)

            if metrics:
                zone_metrics.append(metrics)

        if zone_metrics:
            # Average across zones
            avg_metrics = {
                'Model': model_name,
                'MAE': np.mean([m['MAE'] for m in zone_metrics]),
                'RMSE': np.mean([m['RMSE'] for m in zone_metrics]),
                'CRPS': np.mean([m['CRPS'] for m in zone_metrics]),
                'Pinball_10': np.mean([m['Pinball_10'] for m in zone_metrics]),
                'Winkler': np.mean([m['Winkler'] for m in zone_metrics])
            }
            results.append(avg_metrics)

            print(f"{model_name:20s} | MAE: {avg_metrics['MAE']:6.2f} | "
                  f"RMSE: {avg_metrics['RMSE']:6.2f} | CRPS: {avg_metrics['CRPS']:6.2f}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Add ranking (1 = best, 7 = worst for each metric)
    for metric in ['MAE', 'RMSE', 'CRPS', 'Pinball_10', 'Winkler']:
        df[f'{metric}_Rank'] = df[metric].rank(method='min')

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.3f')

    print(f"\nTable 2 saved to: {output_file}")
    print(f"\nPerformance Summary:")
    print(df[['Model', 'MAE', 'RMSE', 'CRPS', 'Pinball_10', 'Winkler']].to_string(index=False))

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Generate Table 2: Aggregate forecasting performance metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--zones',
        nargs='+',
        default=['no1', 'no2', 'no3', 'no4', 'no5'],
        help='Bidding zones to evaluate (default: all zones)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-04-26',
        help='Start date for evaluation period (default: 2024-04-26)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-04-25',
        help='End date for evaluation period (default: 2025-04-25)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs/table2_performance.csv',
        help='Output file path (default: outputs/table2_performance.csv)'
    )

    args = parser.parse_args()

    # Run evaluation
    test_period = (args.start_date, args.end_date)
    create_performance_table(args.zones, test_period, args.output)

    return 0

if __name__ == '__main__':
    sys.exit(main())
