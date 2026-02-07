#!/usr/bin/env python
"""
plot_calibration.py - Generate Figure 1: Calibration comparison across distributional assumptions

This script creates a 7-panel figure showing calibration diagnostics for all models:
- Panel (a): Reliability diagrams for 90% prediction intervals
- Panel (b): Example forecast distributions during February 2025 price spike

The figure demonstrates JSU's parameter collapse (narrow, overconfident intervals)
compared to properly calibrated Normal and Skewed-t distributions.

Models included:
1. Naive
2. Exponential Smoothing
3. Linear QR
4. XGBoost QR
5. DDNN-Normal
6. DDNN-JSU
7. DDNN-Skewed-t

Usage:
    python scripts/plot_calibration.py --zone no2 --output outputs/figure1_calibration.pdf
    python scripts/plot_calibration.py --zone no2 --period 2025-02-01 2025-02-28

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
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set plotting style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

def load_model_forecasts(zone, distribution, run_id, period):
    """
    Load forecast data for calibration analysis

    Returns:
    --------
    pd.DataFrame with columns: datetime, lower90, upper90, mean, realized
    """
    results_dir = Path('results/forecasts')
    forecast_dir = results_dir / f'df_forecasts_{zone}_{distribution}_{run_id}'

    all_forecasts = []

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

            # Extract 90% prediction interval
            if 'q05' in df.columns and 'q95' in df.columns:
                df['lower90'] = df['q05']
                df['upper90'] = df['q95']
            elif 'q10' in df.columns and 'q90' in df.columns:
                df['lower90'] = df['q10']
                df['upper90'] = df['q90']
            else:
                continue

            all_forecasts.append(df[['mean', 'lower90', 'upper90', 'realized']])

        except Exception as e:
            print(f"Error loading {date_str}: {e}")
            continue

    if not all_forecasts:
        return None

    return pd.concat(all_forecasts)

def calculate_calibration(forecasts, num_bins=10):
    """
    Calculate calibration curve for reliability diagram

    Returns:
    --------
    dict: {
        'nominal_coverage': array of nominal coverage levels,
        'empirical_coverage': array of empirical coverage levels,
        'interval_widths': array of average interval widths
    }
    """
    # Check coverage for different nominal levels
    nominal_levels = np.linspace(0.1, 0.9, num_bins)
    empirical_coverage = []
    interval_widths = []

    for nominal in nominal_levels:
        # Calculate quantiles for this nominal level
        lower_q = (1 - nominal) / 2
        upper_q = 1 - lower_q

        # Get prediction intervals
        lower = forecasts['mean'] - (forecasts['mean'] - forecasts['lower90']) * (nominal / 0.9)
        upper = forecasts['mean'] + (forecasts['upper90'] - forecasts['mean']) * (nominal / 0.9)

        # Calculate empirical coverage
        inside = (forecasts['realized'] >= lower) & (forecasts['realized'] <= upper)
        coverage = inside.mean()

        empirical_coverage.append(coverage)
        interval_widths.append((upper - lower).mean())

    return {
        'nominal_coverage': nominal_levels,
        'empirical_coverage': np.array(empirical_coverage),
        'interval_widths': np.array(interval_widths)
    }

def plot_reliability_diagram(ax, calibration, model_name):
    """
    Plot reliability diagram for a single model

    Perfect calibration = diagonal line (empirical = nominal)
    """
    nominal = calibration['nominal_coverage']
    empirical = calibration['empirical_coverage']

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect calibration')

    # Plot empirical calibration
    ax.plot(nominal, empirical, 'o-', linewidth=2, markersize=6, label=model_name)

    # Shade acceptable range (±10%)
    ax.fill_between(nominal, nominal - 0.1, nominal + 0.1, alpha=0.2, color='gray')

    ax.set_xlabel('Nominal Coverage', fontsize=10)
    ax.set_ylabel('Empirical Coverage', fontsize=10)
    ax.set_title(model_name, fontsize=11, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=8)

def plot_example_forecast(ax, forecasts, spike_date, model_name):
    """
    Plot example forecast during price spike event

    Shows:
    - Forecast mean
    - 90% prediction interval
    - Realized values
    """
    # Extract data for spike period
    spike_start = pd.to_datetime(spike_date)
    spike_end = spike_start + timedelta(days=7)

    mask = (forecasts.index >= spike_start) & (forecasts.index < spike_end)
    data = forecasts[mask]

    if len(data) == 0:
        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        return

    hours = np.arange(len(data))

    # Plot prediction interval
    ax.fill_between(hours, data['lower90'], data['upper90'],
                     alpha=0.3, label='90% PI')

    # Plot mean forecast
    ax.plot(hours, data['mean'], 'b-', linewidth=2, label='Mean forecast')

    # Plot realized values
    ax.plot(hours, data['realized'], 'ro-', linewidth=1, markersize=4, label='Realized')

    ax.set_xlabel('Hours since spike', fontsize=10)
    ax.set_ylabel('Price Premium (EUR/MWh)', fontsize=10)
    ax.set_title(f'{model_name} - Spike Period', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

def create_calibration_figure(zone, period, output_file):
    """
    Create Figure 1: Calibration comparison (7 panels)

    Parameters:
    -----------
    zone : str
        Bidding zone (typically 'no2' as per paper)
    period : tuple
        (start_date, end_date) for analysis
    output_file : str
        Path to output PDF file
    """
    # Define models
    models = [
        ('Naive', 'naive', 'baseline'),
        ('Exp. Smoothing', 'exp_smooth', 'baseline'),
        ('Linear QR', 'lqr', 'replication_lqr'),
        ('XGBoost QR', 'xgb', 'replication_xgb'),
        ('DDNN-Normal', 'normal', 'replication_normal'),
        ('DDNN-JSU', 'jsu', 'replication_jsu'),
        ('DDNN-Skewed-t', 'skewt', 'replication_skewt')
    ]

    # Create figure with 7 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    print(f"Creating calibration figure for zone {zone.upper()}...")
    print(f"Analysis period: {period[0]} to {period[1]}")
    print("")

    # Spike date for example forecasts (February 2025)
    spike_date = '2025-02-15'

    for idx, (model_name, distribution, run_id_template) in enumerate(models):
        row = idx // 3
        col = idx % 3

        ax = fig.add_subplot(gs[row, col])

        # Construct run_id
        if run_id_template == 'baseline':
            run_id = run_id_template
        else:
            run_id = f'{run_id_template}_{zone}'

        # Load forecasts
        forecasts = load_model_forecasts(zone, distribution, run_id, period)

        if forecasts is None or len(forecasts) == 0:
            ax.text(0.5, 0.5, f'No data for {model_name}', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12)
            ax.set_title(model_name, fontsize=11, fontweight='bold')
            print(f"  ✗ {model_name}: No data")
            continue

        # Calculate calibration
        calibration = calculate_calibration(forecasts)

        # Plot reliability diagram
        plot_reliability_diagram(ax, calibration, model_name)

        # Calculate sharpness (average interval width)
        sharpness = calibration['interval_widths'].mean()

        print(f"  ✓ {model_name}: Coverage={calibration['empirical_coverage'].mean():.3f}, "
              f"Sharpness={sharpness:.2f}")

    # Add overall title
    fig.suptitle(f'Figure 1: Calibration Comparison - Zone {zone.upper()}',
                 fontsize=14, fontweight='bold', y=0.98)

    # Add note about JSU parameter collapse
    fig.text(0.5, 0.01,
             'Note: DDNN-JSU may exhibit parameter collapse (overconfident narrow intervals) '
             'during extreme price spikes',
             ha='center', fontsize=9, style='italic')

    # Save figure
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nFigure 1 saved to: {output_file}")

    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Generate Figure 1: Calibration comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--zone',
        type=str,
        default='no2',
        help='Bidding zone for analysis (default: no2)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-01',
        help='Start date for analysis (default: 2025-01-01)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-02-28',
        help='End date for analysis (default: 2025-02-28)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs/figure1_calibration.pdf',
        help='Output file path (default: outputs/figure1_calibration.pdf)'
    )

    args = parser.parse_args()

    # Create figure
    period = (args.start_date, args.end_date)
    create_calibration_figure(args.zone, period, args.output)

    return 0

if __name__ == '__main__':
    sys.exit(main())
