#!/usr/bin/env python
"""
plot_regional.py - Generate Figure 2: Zone-wise CRPS comparison across NO1-NO5

This script creates a line plot showing CRPS (Continuous Ranked Probability Score)
variation across Norwegian bidding zones for selected models:
- Linear QR (Linear Quantile Regression)
- DDNN-Normal
- DDNN-JSU

The figure demonstrates:
- Regional heterogeneity in forecast performance
- 31% performance gap between best (NO3) and worst (NO2) zones
- Consistent model rankings across regions

Usage:
    python scripts/plot_regional.py --zones no1 no2 no3 no4 no5 --output outputs/figure2_regional.pdf
    python scripts/plot_regional.py --zones no1 no2 --output outputs/figure2_partial.pdf

Author: Knut Skjevrak
Last updated: February 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.metrics import calculate_crps

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.3)

def load_forecasts_for_crps(zone, distribution, run_id, period):
    """
    Load forecast data and calculate CRPS

    Returns:
    --------
    float: Average CRPS across evaluation period
    """
    results_dir = Path('results/forecasts')

    # Determine forecast directory
    if distribution in ['jsu', 'normal', 'skewt']:
        forecast_dir = results_dir / f'df_forecasts_{zone}_{distribution}_{run_id}'
        param_dir = results_dir / f'distparams_{zone}_{distribution}_{run_id}'
    else:
        forecast_dir = results_dir / f'{distribution}_{zone}_{run_id}'
        param_dir = None

    all_crps = []

    start_date = pd.to_datetime(period[0])
    end_date = pd.to_datetime(period[1])
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        forecast_file = forecast_dir / f'{date_str}_forecast.csv'

        if not forecast_file.exists():
            continue

        try:
            df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)

            # Get actual values
            if 'realized' not in df.columns:
                continue

            actual = df['realized'].values

            if len(actual) != 24:
                continue

            # Generate prediction samples
            if distribution in ['jsu', 'normal', 'skewt']:
                # Use quantiles as approximation
                quantile_cols = [c for c in df.columns if c.startswith('q')]
                if quantile_cols:
                    samples = df[quantile_cols].values.T
                else:
                    continue
            else:
                # For LQR and XGBoost
                quantile_cols = [c for c in df.columns if c.startswith('q')]
                if quantile_cols:
                    samples = df[quantile_cols].values.T
                else:
                    continue

            # Calculate CRPS for this day
            crps_values = calculate_crps(samples, actual)
            all_crps.extend(crps_values)

        except Exception as e:
            continue

    if not all_crps:
        return None

    return np.mean(all_crps)

def calculate_regional_crps(zones, models, period):
    """
    Calculate CRPS for each model across all zones

    Parameters:
    -----------
    zones : list
        List of bidding zones
    models : list
        List of (model_name, distribution, run_id_template) tuples
    period : tuple
        (start_date, end_date)

    Returns:
    --------
    pd.DataFrame: CRPS values with zones as rows and models as columns
    """
    results = {zone: {} for zone in zones}

    print(f"Calculating regional CRPS across {len(zones)} zones...")
    print(f"Evaluation period: {period[0]} to {period[1]}")
    print("")

    for model_name, distribution, run_id_template in tqdm(models, desc="Models"):
        for zone in zones:
            # Construct run_id
            if run_id_template == 'baseline':
                run_id = run_id_template
            else:
                run_id = f'{run_id_template}_{zone}'

            # Calculate CRPS
            crps = load_forecasts_for_crps(zone, distribution, run_id, period)

            if crps is not None:
                results[zone][model_name] = crps
                print(f"  {model_name:20s} | Zone: {zone.upper()} | CRPS: {crps:6.2f}")
            else:
                results[zone][model_name] = np.nan
                print(f"  {model_name:20s} | Zone: {zone.upper()} | No data")

    # Convert to DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'Zone'

    return df

def create_regional_figure(zones, period, output_file):
    """
    Create Figure 2: Regional CRPS comparison

    Parameters:
    -----------
    zones : list
        List of bidding zones
    period : tuple
        (start_date, end_date)
    output_file : str
        Path to output PDF file
    """
    # Define models to compare
    models = [
        ('Linear QR', 'lqr', 'replication_lqr'),
        ('DDNN-Normal', 'normal', 'replication_normal'),
        ('DDNN-JSU', 'jsu', 'replication_jsu')
    ]

    # Calculate regional CRPS
    df_crps = calculate_regional_crps(zones, models, period)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot CRPS for each model
    zone_labels = [z.upper() for z in zones]
    x_positions = np.arange(len(zones))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']  # Circle, Square, Triangle

    for idx, (model_name, _, _) in enumerate(models):
        if model_name in df_crps.columns:
            values = df_crps[model_name].values

            ax.plot(x_positions, values,
                   marker=markers[idx],
                   color=colors[idx],
                   linewidth=2.5,
                   markersize=10,
                   label=model_name)

    # Formatting
    ax.set_xlabel('Norwegian Bidding Zone', fontsize=14, fontweight='bold')
    ax.set_ylabel('CRPS (EUR/MWh)', fontsize=14, fontweight='bold')
    ax.set_title('Figure 2: Regional Forecasting Performance', fontsize=16, fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(zone_labels, fontsize=12)

    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add performance gap annotation
    if 'DDNN-Normal' in df_crps.columns:
        best_zone = df_crps['DDNN-Normal'].idxmin()
        worst_zone = df_crps['DDNN-Normal'].idxmax()
        best_crps = df_crps.loc[best_zone, 'DDNN-Normal']
        worst_crps = df_crps.loc[worst_zone, 'DDNN-Normal']
        gap_pct = ((worst_crps - best_crps) / best_crps) * 100

        ax.annotate(f'Performance gap:\n{gap_pct:.0f}%',
                   xy=(0.05, 0.95),
                   xycoords='axes fraction',
                   fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   verticalalignment='top')

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure 2 saved to: {output_file}")

    plt.close()

    # Print summary table
    print("\nRegional CRPS Summary:")
    print(df_crps.to_string(float_format='%.2f'))
    print("")

    # Calculate and print statistics
    print("Performance Statistics:")
    for model in models:
        model_name = model[0]
        if model_name in df_crps.columns:
            values = df_crps[model_name].dropna()
            print(f"  {model_name:20s} | Mean: {values.mean():6.2f} | "
                  f"Min: {values.min():6.2f} ({values.idxmin().upper()}) | "
                  f"Max: {values.max():6.2f} ({values.idxmax().upper()})")

def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 2: Regional CRPS comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--zones',
        nargs='+',
        default=['no1', 'no2', 'no3', 'no4', 'no5'],
        help='Bidding zones to analyze (default: all zones)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-04-26',
        help='Start date for evaluation (default: 2024-04-26)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-04-25',
        help='End date for evaluation (default: 2025-04-25)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs/figure2_regional.pdf',
        help='Output file path (default: outputs/figure2_regional.pdf)'
    )

    args = parser.parse_args()

    # Create figure
    period = (args.start_date, args.end_date)
    create_regional_figure(args.zones, period, args.output)

    return 0

if __name__ == '__main__':
    sys.exit(main())
