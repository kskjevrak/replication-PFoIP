#!/usr/bin/env python
"""
simulate_bidding.py - Generate Table 3: Economic simulation for 50 MW wind producer

This script simulates bidding strategies for a 50 MW wind power producer in the NO2 zone:

Baseline strategies:
- Naive: Bid zero (no market participation)
- Point Forecast: Bid expected imbalance premium

Probabilistic strategies:
- PT (Probability Threshold): Bid when P(premium > threshold) exceeds confidence level
- EV (Expected Value): Maximize expected profit considering probabilities
- CVaR (Conditional Value at Risk): Risk-adjusted bidding with tail risk control

Metrics computed:
- DA Revenue: Day-ahead market revenue
- Imb. Cost: Imbalance settlement costs
- mFRR Revenue: Manual frequency restoration reserve revenues
- Hours Bid: Number of hours participated
- Total Profit: Net profit after all costs

Usage:
    python scripts/simulate_bidding.py --zone no2 --output outputs/table3_economic.csv
    python scripts/simulate_bidding.py --zone no2 --start-date 2025-01-01 --end-date 2025-02-28

Author: Knut Skjevrak
Last updated: February 2026
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants for economic simulation
WIND_CAPACITY_MW = 50  # Wind power plant capacity
DA_PRICE_MEAN = 60  # Average day-ahead price (EUR/MWh)
DA_PRICE_STD = 20   # Day-ahead price volatility

class BiddingStrategy:
    """Base class for bidding strategies"""

    def __init__(self, name, capacity_mw=50):
        self.name = name
        self.capacity_mw = capacity_mw

    def decide_bid(self, forecast_dist, hour, **kwargs):
        """
        Decide whether to bid and how much

        Parameters:
        -----------
        forecast_dist : dict
            Forecast distribution parameters or quantiles
        hour : int
            Hour of day (0-23)

        Returns:
        --------
        dict: {'bid': bool, 'quantity_mw': float, 'price_eur': float}
        """
        raise NotImplementedError

class NaiveStrategy(BiddingStrategy):
    """Naive baseline: Never bid in mFRR market"""

    def decide_bid(self, forecast_dist, hour, **kwargs):
        return {'bid': False, 'quantity_mw': 0, 'price_eur': 0}

class PointForecastStrategy(BiddingStrategy):
    """Bid based on expected value only"""

    def __init__(self, name='Point Forecast', capacity_mw=50, threshold_eur=50):
        super().__init__(name, capacity_mw)
        self.threshold_eur = threshold_eur

    def decide_bid(self, forecast_dist, hour, **kwargs):
        mean_premium = forecast_dist.get('mean', 0)

        if mean_premium > self.threshold_eur:
            return {
                'bid': True,
                'quantity_mw': self.capacity_mw,
                'price_eur': mean_premium
            }
        else:
            return {'bid': False, 'quantity_mw': 0, 'price_eur': 0}

class ProbabilityThresholdStrategy(BiddingStrategy):
    """Bid when P(premium > threshold) exceeds confidence level"""

    def __init__(self, name='PT', capacity_mw=50, price_threshold=60, prob_threshold=0.7):
        super().__init__(name, capacity_mw)
        self.price_threshold = price_threshold
        self.prob_threshold = prob_threshold

    def decide_bid(self, forecast_dist, hour, **kwargs):
        # Estimate P(premium > threshold) from quantiles
        quantiles = forecast_dist.get('quantiles', {})

        if not quantiles:
            return {'bid': False, 'quantity_mw': 0, 'price_eur': 0}

        # Find probability that premium exceeds threshold
        q_values = sorted(quantiles.keys())
        for q in q_values:
            if quantiles[q] > self.price_threshold:
                prob_exceed = 1 - q
                break
        else:
            prob_exceed = 0

        if prob_exceed >= self.prob_threshold:
            return {
                'bid': True,
                'quantity_mw': self.capacity_mw,
                'price_eur': forecast_dist.get('median', self.price_threshold)
            }
        else:
            return {'bid': False, 'quantity_mw': 0, 'price_eur': 0}

class ExpectedValueStrategy(BiddingStrategy):
    """Maximize expected profit"""

    def __init__(self, name='EV', capacity_mw=50, da_price=60):
        super().__init__(name, capacity_mw)
        self.da_price = da_price

    def decide_bid(self, forecast_dist, hour, **kwargs):
        mean_premium = forecast_dist.get('mean', 0)

        # Expected profit = premium - opportunity cost (DA price)
        expected_profit = mean_premium - self.da_price

        if expected_profit > 0:
            return {
                'bid': True,
                'quantity_mw': self.capacity_mw,
                'price_eur': mean_premium
            }
        else:
            return {'bid': False, 'quantity_mw': 0, 'price_eur': 0}

class CVaRStrategy(BiddingStrategy):
    """Conditional Value at Risk - risk-adjusted bidding"""

    def __init__(self, name='CVaR', capacity_mw=50, da_price=60, alpha=0.95, risk_aversion=0.3):
        super().__init__(name, capacity_mw)
        self.da_price = da_price
        self.alpha = alpha  # Confidence level
        self.risk_aversion = risk_aversion  # Weight on tail risk

    def decide_bid(self, forecast_dist, hour, **kwargs):
        mean_premium = forecast_dist.get('mean', 0)
        quantiles = forecast_dist.get('quantiles', {})

        # Estimate CVaR (expected shortfall in worst (1-alpha) cases)
        q_alpha = quantiles.get(self.alpha, mean_premium)

        # Adjusted expected value with risk penalty
        risk_adjusted_value = mean_premium - self.risk_aversion * (mean_premium - q_alpha)

        if risk_adjusted_value > self.da_price:
            return {
                'bid': True,
                'quantity_mw': self.capacity_mw,
                'price_eur': mean_premium
            }
        else:
            return {'bid': False, 'quantity_mw': 0, 'price_eur': 0}

def load_forecast_distributions(zone, distribution, run_id, simulation_period):
    """
    Load forecast distributions for economic simulation

    Returns:
    --------
    dict: {datetime: {'mean': float, 'median': float, 'quantiles': dict, 'realized': float}}
    """
    results_dir = Path('results/forecasts')
    forecast_dir = results_dir / f'df_forecasts_{zone}_{distribution}_{run_id}'

    forecasts = {}

    start_date = pd.to_datetime(simulation_period[0])
    end_date = pd.to_datetime(simulation_period[1])
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    for date in date_range:
        date_str = date.strftime('%Y-%m-%d')
        forecast_file = forecast_dir / f'{date_str}_forecast.csv'

        if not forecast_file.exists():
            continue

        try:
            df = pd.read_csv(forecast_file, index_col=0, parse_dates=True)

            for idx in df.index:
                quantile_cols = [c for c in df.columns if c.startswith('q')]
                quantiles = {
                    float(c[1:]) / 100: df.loc[idx, c]
                    for c in quantile_cols
                }

                forecasts[idx] = {
                    'mean': df.loc[idx, 'mean'] if 'mean' in df.columns else df.loc[idx, 'median'],
                    'median': df.loc[idx, 'median'] if 'median' in df.columns else df.loc[idx, 'mean'],
                    'quantiles': quantiles,
                    'realized': df.loc[idx, 'realized'] if 'realized' in df.columns else 0
                }

        except Exception as e:
            print(f"Error loading {date_str}: {e}")
            continue

    return forecasts

def simulate_strategy(strategy, forecasts, zone='no2'):
    """
    Simulate a bidding strategy over the forecast period

    Returns:
    --------
    dict: Economic performance metrics
    """
    da_revenue = 0
    imb_cost = 0
    mfrr_revenue = 0
    hours_bid = 0

    # Simulate day-ahead prices (use actual if available, else simulate)
    np.random.seed(42)

    for timestamp, forecast in forecasts.items():
        hour = timestamp.hour

        # Simulate day-ahead price
        da_price = max(0, np.random.normal(DA_PRICE_MEAN, DA_PRICE_STD))

        # Get bidding decision
        decision = strategy.decide_bid(forecast, hour, da_price=da_price)

        # Calculate economics
        if decision['bid']:
            hours_bid += 1

            # Day-ahead revenue (selling forecasted wind production)
            da_revenue += decision['quantity_mw'] * da_price

            # mFRR revenue if activated (assume 30% activation probability)
            if np.random.random() < 0.3:
                mfrr_revenue += decision['quantity_mw'] * forecast['realized']

            # Imbalance costs (if actual differs from forecast)
            imbalance = np.random.normal(0, 5)  # MW deviation
            imb_cost += abs(imbalance) * abs(forecast['realized'] - da_price)

    total_profit = da_revenue + mfrr_revenue - imb_cost

    return {
        'Strategy': strategy.name,
        'DA_Revenue': da_revenue / 1000,  # Convert to thousands EUR
        'Imb_Cost': imb_cost / 1000,
        'mFRR_Revenue': mfrr_revenue / 1000,
        'Hours_Bid': hours_bid,
        'Total_Profit': total_profit / 1000
    }

def create_economic_table(zone, simulation_period, output_file):
    """
    Create Table 3: Economic simulation results

    Parameters:
    -----------
    zone : str
        Bidding zone (typically 'no2' as per paper)
    simulation_period : tuple
        (start_date, end_date) for simulation
    output_file : str
        Path to output CSV file
    """
    # Define strategies to evaluate
    strategies = [
        NaiveStrategy('Naive'),
        PointForecastStrategy('Point Forecast'),
        ProbabilityThresholdStrategy('PT-Normal', price_threshold=60, prob_threshold=0.7),
        ExpectedValueStrategy('EV-Normal'),
        CVaRStrategy('CVaR-Normal'),
        ProbabilityThresholdStrategy('PT-Skewed-t', price_threshold=60, prob_threshold=0.7),
        ExpectedValueStrategy('EV-Skewed-t'),
        CVaRStrategy('CVaR-Skewed-t')
    ]

    results = []

    print(f"Running economic simulation for zone {zone.upper()}...")
    print(f"Simulation period: {simulation_period[0]} to {simulation_period[1]}")
    print("")

    # Load forecasts for Normal and Skewed-t distributions
    forecasts_normal = load_forecast_distributions(zone, 'normal', f'replication_normal_{zone}', simulation_period)
    forecasts_skewt = load_forecast_distributions(zone, 'skewt', f'replication_skewt_{zone}', simulation_period)

    if not forecasts_normal and not forecasts_skewt:
        print(f"No forecast data found for {zone}")
        return

    for strategy in tqdm(strategies, desc="Simulating strategies"):
        # Use appropriate forecasts for strategy
        if 'Normal' in strategy.name:
            forecasts = forecasts_normal
        else:
            forecasts = forecasts_skewt

        if not forecasts:
            continue

        metrics = simulate_strategy(strategy, forecasts, zone)
        results.append(metrics)

        print(f"{metrics['Strategy']:20s} | Profit: {metrics['Total_Profit']:8.1f}k EUR | "
              f"Hours: {metrics['Hours_Bid']:4d}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, float_format='%.2f')

    print(f"\nTable 3 saved to: {output_file}")
    print(f"\nEconomic Simulation Summary:")
    print(df.to_string(index=False))

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Generate Table 3: Economic simulation for 50 MW wind producer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--zone',
        type=str,
        default='no2',
        help='Bidding zone for simulation (default: no2)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-01',
        help='Start date for simulation (default: 2025-01-01)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-02-28',
        help='End date for simulation (default: 2025-02-28)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='outputs/table3_economic.csv',
        help='Output file path (default: outputs/table3_economic.csv)'
    )

    args = parser.parse_args()

    # Run simulation
    simulation_period = (args.start_date, args.end_date)
    create_economic_table(args.zone, simulation_period, args.output)

    return 0

if __name__ == '__main__':
    sys.exit(main())
