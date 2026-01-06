#!/usr/bin/env python
"""
synthetic_data.py - Generate synthetic electricity imbalance price data

This script generates synthetic data that approximates the statistical properties
of the real Norwegian electricity market data for testing and demonstration purposes.

IMPORTANT: This is synthetic data for code testing only. Results will differ from
the paper as this does not capture the true market dynamics. See DATA_AVAILABILITY.md.

The generated data includes:
- Hourly time series with proper timezone handling (Europe/Oslo)
- Target variable: premium (imbalance price)
- Features: mFRR up/down prices and volumes

Statistical properties attempt to match thesis Appendix B (Tables B1-B5):
- Mean premium: ~85-90 EUR/MWh (zone-dependent)
- Std premium: ~90-100 EUR/MWh (heavy-tailed)
- Maximum spikes: >1200 EUR/MWh
- Temporal autocorrelation and regime-switching behavior

Usage:
    python src/data/synthetic_data.py --zone no1 --start-date 2019-08-25 --end-date 2024-04-26
    python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2024-04-26
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


def generate_synthetic_electricity_data(
    start_date: str,
    end_date: str,
    zone: str = 'no1',
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic electricity imbalance price data.

    This function creates time series data with statistical properties approximating
    those described in the thesis (Appendix B). The data exhibits:
    - Heavy-tailed price distributions
    - Temporal autocorrelation
    - Daily, weekly, and seasonal patterns
    - Regime-switching behavior (calm periods + extreme spikes)

    LIMITATIONS:
    - Simplified market dynamics (no strategic bidding, supply shocks, etc.)
    - Approximate correlations between variables
    - Does not capture non-stationarity and structural breaks in real data

    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
    zone : str
        Bidding zone identifier (no1, no2, no3, no4, no5)
        Different zones have slightly different statistical properties
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic data with hourly frequency and required columns:
        - premium: Price premium (target variable)
        - mFRR_price_up: Upward regulation price
        - mFRR_price_down: Downward regulation price
        - mFRR_vol_up: Upward activation volume
        - mFRR_vol_down: Downward activation volume
    """
    np.random.seed(seed + hash(zone) % 1000)  # Zone-specific seed

    # Create hourly datetime index with Oslo timezone
    start = pd.Timestamp(start_date, tz='Europe/Oslo')
    end = pd.Timestamp(end_date, tz='Europe/Oslo')

    # Generate hourly range
    date_range = pd.date_range(start=start, end=end, freq='h', tz='Europe/Oslo')
    n_hours = len(date_range)

    # Extract time features for pattern generation (convert to numpy arrays)
    hours = date_range.hour.values
    days_of_week = date_range.dayofweek.values
    months = date_range.month.values

    # Generate base patterns with daily and seasonal cycles
    # Hour-of-day pattern (peak during business hours)
    hour_pattern = 10 * np.sin(2 * np.pi * hours / 24 - np.pi/2) + 10

    # Day-of-week pattern (lower on weekends)
    weekday_pattern = -5 * ((days_of_week >= 5).astype(float))

    # Seasonal pattern (higher in winter)
    seasonal_pattern = 15 * np.sin(2 * np.pi * (months - 1) / 12 + np.pi)

    # Add zone-specific offset
    zone_offsets = {'no1': 0, 'no2': 5, 'no3': -3, 'no4': 2, 'no5': -5}
    zone_offset = zone_offsets.get(zone, 0)

    # Generate mFRR_price_up (positive, with spikes)
    mFRR_price_up_base = 50 + zone_offset + hour_pattern + weekday_pattern + seasonal_pattern
    mFRR_price_up_noise = np.random.gamma(2, 10, n_hours)  # Right-skewed noise
    mFRR_price_up_spikes = np.random.exponential(20, n_hours) * (np.random.random(n_hours) < 0.05)
    mFRR_price_up = np.maximum(0, mFRR_price_up_base + mFRR_price_up_noise + mFRR_price_up_spikes)

    # Generate mFRR_price_down (can be negative)
    mFRR_price_down_base = -30 + zone_offset + 0.5 * hour_pattern + weekday_pattern + seasonal_pattern
    mFRR_price_down_noise = np.random.normal(0, 15, n_hours)
    mFRR_price_down = mFRR_price_down_base + mFRR_price_down_noise

    # Generate volumes (always positive, correlated with prices)
    mFRR_vol_up_base = 200 + 50 * np.sin(2 * np.pi * hours / 24)
    mFRR_vol_up_noise = np.random.exponential(100, n_hours)
    mFRR_vol_up = np.maximum(0, mFRR_vol_up_base + mFRR_vol_up_noise)

    mFRR_vol_down_base = 150 + 40 * np.sin(2 * np.pi * hours / 24 + np.pi)
    mFRR_vol_down_noise = np.random.exponential(80, n_hours)
    mFRR_vol_down = np.maximum(0, mFRR_vol_down_base + mFRR_vol_down_noise)

    # Generate premium (target) - function of mFRR prices with noise
    # Premium is typically the difference between up and down regulation
    premium_base = 0.6 * (mFRR_price_up - mFRR_price_down)
    premium_noise = np.random.normal(0, 10, n_hours)
    premium_spikes = np.random.normal(0, 50, n_hours) * (np.random.random(n_hours) < 0.02)
    premium = premium_base + premium_noise + premium_spikes

    # Add some autocorrelation to make it more realistic
    for i in range(1, n_hours):
        premium[i] = 0.7 * premium[i] + 0.3 * premium[i-1]
        mFRR_price_up[i] = 0.6 * mFRR_price_up[i] + 0.4 * mFRR_price_up[i-1]
        mFRR_price_down[i] = 0.6 * mFRR_price_down[i] + 0.4 * mFRR_price_down[i-1]

    # Create DataFrame
    data = pd.DataFrame({
        'premium': premium,
        'mFRR_price_up': mFRR_price_up,
        'mFRR_price_down': mFRR_price_down,
        'mFRR_vol_up': mFRR_vol_up,
        'mFRR_vol_down': mFRR_vol_down,
    }, index=date_range)

    return data


def save_synthetic_data(
    data: pd.DataFrame,
    zone: str,
    output_dir: Path = None
):
    """
    Save synthetic data to parquet file in the expected location.

    Parameters
    ----------
    data : pd.DataFrame
        Synthetic data to save
    zone : str
        Bidding zone identifier
    output_dir : Path, optional
        Output directory. If None, uses src/data/{zone}/
    """
    if output_dir is None:
        # Default to src/data/{zone}/
        script_dir = Path(__file__).parent
        output_dir = script_dir / zone
    else:
        output_dir = Path(output_dir)

    # Create directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as parquet
    output_file = output_dir / f"merged_dataset_{zone}.parquet"
    data.to_parquet(output_file, engine='pyarrow', compression='snappy')

    print(f"Synthetic data saved to: {output_file}")
    print(f"  Shape: {data.shape}")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    print(f"  Columns: {list(data.columns)}")
    print(f"\nSummary statistics:")
    print(data.describe())


def main():
    """Main function to generate synthetic data from command line."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic electricity imbalance price data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate data for a single zone
  python src/data/synthetic_data.py --zone no1 --start-date 2019-08-25 --end-date 2024-04-26

  # Generate data for all zones
  python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2024-04-26

  # Specify custom output directory
  python src/data/synthetic_data.py --zone no1 --output-dir ./custom_data
        """
    )

    parser.add_argument(
        '--zone',
        type=str,
        default='no1',
        choices=['no1', 'no2', 'no3', 'no4', 'no5'],
        help='Bidding zone to generate data for (default: no1)'
    )

    parser.add_argument(
        '--all-zones',
        action='store_true',
        help='Generate data for all zones (no1-no5)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2019-08-25',
        help='Start date in YYYY-MM-DD format (default: 2019-08-25)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-04-26',
        help='End date in YYYY-MM-DD format (default: 2024-04-26)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: src/data/{zone}/)'
    )

    args = parser.parse_args()

    # Validate dates
    try:
        pd.Timestamp(args.start_date)
        pd.Timestamp(args.end_date)
    except Exception as e:
        print(f"Error: Invalid date format. Use YYYY-MM-DD format.")
        print(f"Details: {e}")
        return 1

    # Generate data for specified zone(s)
    if args.all_zones:
        zones = ['no1', 'no2', 'no3', 'no4', 'no5']
        print(f"Generating synthetic data for all zones: {zones}")
        print(f"Date range: {args.start_date} to {args.end_date}\n")
    else:
        zones = [args.zone]
        print(f"Generating synthetic data for zone: {args.zone}")
        print(f"Date range: {args.start_date} to {args.end_date}\n")

    for zone in zones:
        print(f"\n{'='*60}")
        print(f"Processing zone: {zone.upper()}")
        print(f"{'='*60}")

        # Generate data
        data = generate_synthetic_electricity_data(
            start_date=args.start_date,
            end_date=args.end_date,
            zone=zone,
            seed=args.seed
        )

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / zone
        else:
            output_dir = None  # Will use default

        # Save data
        save_synthetic_data(data, zone, output_dir)

    print(f"\n{'='*60}")
    print("Synthetic data generation complete!")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
