#!/usr/bin/env python
"""
run_tuning.py - Script to run hyperparameter tuning with command-line arguments
Compatible with cluster execution

Usage:
    python run_tuning.py --zone no1 --distribution jsu --run-id my_run
    python run_tuning.py no1 jsu my_run  # Positional arguments also supported
"""
import os
import sys
import argparse
import logging
from datetime import datetime

# Add project root to path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.training.optuna_tuner import OptunaHPTuner

# Valid options
ZONES = ['no1', 'no2', 'no3', 'no4', 'no5']
DISTRIBUTIONS = ['Normal', 'JSU', 'normal', 'jsu', 'skewt']

def main():
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for Deep Distributional Neural Networks (DDNN)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using named arguments (recommended)
  python run_tuning.py --zone no1 --distribution jsu --run-id replication_001

  # Using positional arguments (legacy)
  python run_tuning.py no1 jsu replication_001

  # Specify number of trials
  python run_tuning.py --zone no1 --distribution jsu --run-id test --n-trials 32

  # Specify config file
  python run_tuning.py --zone no1 --distribution jsu --run-id test --config config/custom_config.yml

  # Non-interactive mode (for cluster execution)
  python run_tuning.py --zone no1 --distribution jsu --run-id test --no-confirm
        """
    )

    parser.add_argument(
        'zone_positional',
        nargs='?',
        help='Bidding zone (positional argument, legacy support)'
    )

    parser.add_argument(
        'distribution_positional',
        nargs='?',
        help='Probability distribution (positional argument, legacy support)'
    )

    parser.add_argument(
        'run_id_positional',
        nargs='?',
        help='Run identifier (positional argument, legacy support)'
    )

    parser.add_argument(
        '--zone', '-z',
        type=str,
        choices=ZONES,
        help='Bidding zone (no1, no2, no3, no4, no5)'
    )

    parser.add_argument(
        '--distribution', '-d',
        type=str,
        help='Probability distribution (jsu, normal, skewt)'
    )

    parser.add_argument(
        '--run-id', '-r',
        type=str,
        help='Unique run identifier'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default_config.yml',
        help='Path to configuration file (default: config/default_config.yml)'
    )

    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=None,
        help='Number of Optuna trials (default: from config file, typically 128)'
    )

    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt (for non-interactive execution)'
    )

    args = parser.parse_args()

    # Handle positional arguments (legacy support)
    zone = args.zone or args.zone_positional
    distribution = args.distribution or args.distribution_positional
    run_id = args.run_id or args.run_id_positional

    # Validate required arguments
    if not zone:
        parser.error('zone is required (use --zone or provide as first positional argument)')
    if not distribution:
        parser.error('distribution is required (use --distribution or provide as second positional argument)')
    if not run_id:
        parser.error('run-id is required (use --run-id or provide as third positional argument)')

    # Normalize inputs
    zone = zone.lower()
    distribution = distribution.lower()

    # Validate inputs
    if zone not in ZONES:
        parser.error(f'zone must be one of {ZONES}, got: {zone}')

    distribution_map = {
        'normal': 'Normal',
        'jsu': 'JSU',
        'skewt': 'skewt',
        'studentt': 'skewt',
        't': 'skewt'
    }
    if distribution.lower() not in distribution_map:
        parser.error(f'distribution must be one of {list(distribution_map.keys())}, got: {distribution}')

    # Display configuration
    print(f"\n{'='*60}")
    print(f"DDNN Hyperparameter Tuning Configuration")
    print(f"{'='*60}")
    print(f"Zone:         {zone.upper()}")
    print(f"Distribution: {distribution.upper()}")
    print(f"Run ID:       {run_id}")
    print(f"Config file:  {args.config}")
    print(f"N trials:     {args.n_trials if args.n_trials else 'from config (typically 128)'}")
    print(f"{'='*60}\n")

    # Confirmation prompt (unless --no-confirm)
    if not args.no_confirm:
        inp = input('Press Enter to continue or Ctrl+C to cancel...')
        if inp.strip() != '':
            print('Note: Press Enter without typing to continue, or Ctrl+C to cancel')
            inp = input('Press Enter to continue or Ctrl+C to cancel...')

    # Set environment variable for thread control
    os.environ["OMP_NUM_THREADS"] = "1"

    # Initialize tuner
    print("\nInitializing hyperparameter tuner...")
    tuner = OptunaHPTuner(
        config_path=args.config,
        zone=zone,
        distribution=distribution,
        run_id=run_id
    )

    # Run optimization
    print(f"\nStarting optimization with {args.n_trials or 'default'} trials...")
    n_trials = args.n_trials if args.n_trials else 128
    best_params = tuner.run_optimization(n_trials=n_trials)

    # Success message
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"{'='*60}")
    print(f"Best parameters saved to:")
    print(f"  results/models/{zone}_{distribution.lower()}_{run_id}/")
    print(f"{'='*60}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())