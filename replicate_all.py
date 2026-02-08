#!/usr/bin/env python
"""
replicate_all.py - One-command replication script for

This script automates the complete replication pipeline:
1. Generate synthetic data for all zones
2. Train all models (DDNN-Normal, DDNN-JSU, DDNN-Skewed-t, LQR, XGBoost)
3. Generate predictions for test period
4. Evaluate forecasting performance (Table 2)
5. Run economic simulation (Table 3)
6. Generate all figures (Figures 1-2)

Usage:
    python replicate_all.py                    # Run full replication
    python replicate_all.py --quick-test       # Fast test with reduced trials
    python replicate_all.py --zone no1         # Replicate for single zone only

Expected runtime:
    - Full replication (all zones): 6-8 hours on CPU, 2-4 hours on GPU
    - Single zone: 1-2 hours on CPU, 30-60 minutes on GPU
    - Quick test: 30-45 minutes

Author: Knut Skjevrak
Last updated: February 2026
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Setup logging
def setup_logging(log_dir='outputs/logs'):
    """Configure logging for the replication script"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'replication_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def run_command(cmd, description, logger):
    """Execute a shell command and log results"""
    logger.info(f"{'='*80}")
    logger.info(f"STEP: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*80}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"✓ {description} completed successfully")
        if result.stdout:
            logger.info(f"Output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"stderr:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"✗ {description} failed with exception: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Complete replication pipeline ',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full replication (all zones, all models)
  python replicate_all.py

  # Quick test (reduced trials, single zone)
  python replicate_all.py --quick-test

  # Single zone only
  python replicate_all.py --zone no1

  # Skip data generation (data already exists)
  python replicate_all.py --skip-data-generation

  # Custom configuration
  python replicate_all.py --config config/custom_config.yml
        """
    )

    parser.add_argument(
        '--zone',
        type=str,
        choices=['no1', 'no2', 'no3', 'no4', 'no5'],
        help='Replicate for single zone only (default: all zones)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Fast test mode with reduced trials (10 instead of 128)'
    )

    parser.add_argument(
        '--skip-data-generation',
        action='store_true',
        help='Skip synthetic data generation (use existing data)'
    )

    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training (use existing trained models)'
    )

    parser.add_argument(
        '--skip-predictions',
        action='store_true',
        help='Skip prediction generation (use existing predictions)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config/default_config.yml',
        help='Path to configuration file (default: config/default_config.yml)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs/)'
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(f'{args.output_dir}/logs', exist_ok=True)

    # Setup logging
    log_file = setup_logging(f'{args.output_dir}/logs')
    logger = logging.getLogger(__name__)

    logger.info("="*80)
    logger.info("PROBABILISTIC FORECASTING OF IMBALANCE PRICES - REPLICATION PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Configuration: {args.config}")

    # Define zones and models
    zones = [args.zone] if args.zone else ['no1', 'no2', 'no3', 'no4', 'no5']
    distributions = ['jsu', 'normal', 'skewt']
    n_trials = 10 if args.quick_test else 128

    logger.info(f"Zones: {', '.join(zones)}")
    logger.info(f"Distributions: {', '.join(distributions)}")
    logger.info(f"Hyperparameter tuning trials: {n_trials}")
    logger.info(f"Quick test mode: {args.quick_test}")
    logger.info("")

    # Track overall success
    all_steps_successful = True
    step_results = {}

    # =========================================================================
    # STEP 1: Generate Synthetic Data
    # =========================================================================
    if not args.skip_data_generation:
        logger.info("\n" + "="*80)
        logger.info("PHASE 1: SYNTHETIC DATA GENERATION")
        logger.info("="*80 + "\n")

        zone_arg = f"--zone {args.zone}" if args.zone else "--all-zones"
        cmd = [
            sys.executable,
            'src/data/synthetic_data.py',
            *zone_arg.split(),
            '--start-date', '2019-08-25',
            '--end-date', '2025-04-25'
        ]

        success = run_command(cmd, "Synthetic data generation", logger)
        step_results['data_generation'] = success
        all_steps_successful &= success
    else:
        logger.info("Skipping data generation (using existing data)")
        step_results['data_generation'] = 'skipped'

    # =========================================================================
    # STEP 2: Model Training
    # =========================================================================
    if not args.skip_training:
        logger.info("\n" + "="*80)
        logger.info("PHASE 2: MODEL TRAINING")
        logger.info("="*80 + "\n")

        training_results = {}

        # Train DDNN models for each zone and distribution
        for zone in zones:
            for dist in distributions:
                run_id = f"replication_{dist}_{zone}"

                cmd = [
                    sys.executable,
                    'run_tuning.py',
                    '--zone', zone,
                    '--distribution', dist,
                    '--run-id', run_id,
                    '--n-trials', str(n_trials),
                    '--config', args.config,
                    '--no-confirm'
                ]

                success = run_command(
                    cmd,
                    f"Training DDNN-{dist.upper()} for zone {zone.upper()}",
                    logger
                )
                training_results[f'DDNN-{dist}_{zone}'] = success
                all_steps_successful &= success

        # Train Linear Quantile Regression
        for zone in zones:
            run_id = f"replication_lqr_{zone}"
            cmd = [
                sys.executable,
                'run_lqa_tuning.py',
                zone,
                run_id
            ]

            success = run_command(
                cmd,
                f"Training Linear Quantile Regression for zone {zone.upper()}",
                logger
            )
            training_results[f'LQR_{zone}'] = success
            all_steps_successful &= success

        # Train XGBoost Quantile Regression
        for zone in zones:
            run_id = f"replication_xgb_{zone}"
            cmd = [
                sys.executable,
                'run_xgb_tuning.py',
                zone,
                run_id
            ]

            success = run_command(
                cmd,
                f"Training XGBoost Quantile Regression for zone {zone.upper()}",
                logger
            )
            training_results[f'XGB_{zone}'] = success
            all_steps_successful &= success

        step_results['training'] = training_results
    else:
        logger.info("Skipping model training (using existing trained models)")
        step_results['training'] = 'skipped'

    # =========================================================================
    # STEP 3: Generate Predictions
    # =========================================================================
    if not args.skip_predictions:
        logger.info("\n" + "="*80)
        logger.info("PHASE 3: PREDICTION GENERATION")
        logger.info("="*80 + "\n")

        # Test period for predictions
        test_start = '2024-04-26'
        test_end = '2024-05-26' if not args.quick_test else '2024-04-30'

        prediction_results = {}

        # Generate predictions for DDNN models
        for zone in zones:
            for dist in distributions:
                run_id = f"replication_{dist}_{zone}"

                cmd = [
                    sys.executable,
                    'run_pred.py',
                    '--zone', zone,
                    '--distribution', dist,
                    '--run-id', run_id,
                    '--start-date', test_start,
                    '--end-date', test_end
                ]

                success = run_command(
                    cmd,
                    f"Generating predictions for DDNN-{dist.upper()} zone {zone.upper()}",
                    logger
                )
                prediction_results[f'DDNN-{dist}_{zone}'] = success
                all_steps_successful &= success

        # Generate predictions for LQR
        for zone in zones:
            run_id = f"replication_lqr_{zone}"
            cmd = [
                sys.executable,
                'run_lqa_pred.py',
                zone,
                run_id,
                '--start-date', test_start,
                '--end-date', test_end
            ]

            success = run_command(
                cmd,
                f"Generating predictions for LQR zone {zone.upper()}",
                logger
            )
            prediction_results[f'LQR_{zone}'] = success
            all_steps_successful &= success

        # Generate predictions for XGBoost
        for zone in zones:
            run_id = f"replication_xgb_{zone}"
            cmd = [
                sys.executable,
                'run_xgb_pred.py',
                zone,
                run_id,
                '--start-date', test_start,
                '--end-date', test_end
            ]

            success = run_command(
                cmd,
                f"Generating predictions for XGBoost zone {zone.upper()}",
                logger
            )
            prediction_results[f'XGB_{zone}'] = success
            all_steps_successful &= success

        step_results['predictions'] = prediction_results
    else:
        logger.info("Skipping prediction generation (using existing predictions)")
        step_results['predictions'] = 'skipped'

    # =========================================================================
    # STEP 4: Evaluation and Output Generation
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: EVALUATION AND OUTPUT GENERATION")
    logger.info("="*80 + "\n")

    evaluation_results = {}

    # Generate Table 1: Descriptive Statistics
    cmd = [
        sys.executable,
        'src/data/synthetic_data.py',
        '--all-zones' if not args.zone else f'--zone {args.zone}',
        '--start-date', '2019-08-25',
        '--end-date', '2024-04-25',
        '--save-stats', f'{args.output_dir}/table1_descriptive_stats.csv'
    ]
    # Flatten the command list properly
    cmd = [c for c in cmd if c]  # Remove empty strings
    if not args.zone:
        cmd[2] = '--all-zones'
    success = run_command(cmd, "Generating Table 1 (Descriptive Statistics)", logger)
    evaluation_results['table1'] = success
    all_steps_successful &= success

    # Generate Table 2: Forecasting Performance
    cmd = [
        sys.executable,
        'scripts/evaluate_models.py',
        '--zones', *zones,
        '--output', f'{args.output_dir}/table2_performance.csv'
    ]
    success = run_command(cmd, "Generating Table 2 (Forecasting Performance)", logger)
    evaluation_results['table2'] = success
    all_steps_successful &= success

    # Generate Table 3: Economic Simulation
    # For Table 3, we use zone NO2 as per paper specification
    table3_zone = 'no2' if 'no2' in zones else zones[0]
    cmd = [
        sys.executable,
        'scripts/simulate_bidding.py',
        '--zone', table3_zone,
        '--output', f'{args.output_dir}/table3_economic.csv'
    ]
    success = run_command(cmd, "Generating Table 3 (Economic Simulation)", logger)
    evaluation_results['table3'] = success
    all_steps_successful &= success

    # Generate Figure 1: Calibration Comparison (7 panels)
    # For Figure 1, we use zone NO2 as per paper specification
    figure1_zone = 'no2' if 'no2' in zones else zones[0]
    cmd = [
        sys.executable,
        'scripts/plot_calibration.py',
        '--zone', figure1_zone,
        '--output', f'{args.output_dir}/figure1_calibration.pdf'
    ]
    success = run_command(cmd, "Generating Figure 1 (Calibration Comparison)", logger)
    evaluation_results['figure1'] = success
    all_steps_successful &= success

    # Generate Figure 2: Regional CRPS Comparison
    cmd = [
        sys.executable,
        'scripts/plot_regional.py',
        '--zones', *zones,
        '--output', f'{args.output_dir}/figure2_regional.pdf'
    ]
    success = run_command(cmd, "Generating Figure 2 (Regional Performance)", logger)
    evaluation_results['figure2'] = success
    all_steps_successful &= success

    step_results['evaluation'] = evaluation_results

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "="*80)
    logger.info("REPLICATION PIPELINE SUMMARY")
    logger.info("="*80 + "\n")

    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("")

    # Print detailed results
    logger.info("Step-by-step results:")
    logger.info("-" * 80)

    for phase, results in step_results.items():
        if results == 'skipped':
            logger.info(f"  {phase.upper()}: SKIPPED")
        elif isinstance(results, dict):
            success_count = sum(1 for v in results.values() if v)
            total_count = len(results)
            logger.info(f"  {phase.upper()}: {success_count}/{total_count} successful")
            for task, success in results.items():
                status = "✓" if success else "✗"
                logger.info(f"    {status} {task}")
        else:
            status = "✓" if results else "✗"
            logger.info(f"  {status} {phase.upper()}")

    logger.info("-" * 80)
    logger.info("")

    # Save summary to JSON
    summary_file = f'{args.output_dir}/replication_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'zones': zones,
            'distributions': distributions,
            'n_trials': n_trials,
            'quick_test': args.quick_test,
            'results': step_results,
            'overall_success': all_steps_successful
        }, f, indent=2)

    logger.info(f"Summary saved to: {summary_file}")
    logger.info("")

    if all_steps_successful:
        logger.info("="*80)
        logger.info("✓ REPLICATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("")
        logger.info(f"All outputs are available in: {args.output_dir}/")
        logger.info("  - table1_descriptive_stats.csv")
        logger.info("  - table2_performance.csv")
        logger.info("  - table3_economic.csv")
        logger.info("  - figure1_calibration.pdf")
        logger.info("  - figure2_regional.pdf")
        logger.info("")
        logger.info("Note: Results obtained with synthetic data will deviate from published")
        logger.info("values. See README.md section 'Synthetic Data Deviations' for details.")
        return 0
    else:
        logger.error("="*80)
        logger.error("✗ REPLICATION PIPELINE COMPLETED WITH ERRORS")
        logger.error("="*80)
        logger.error("")
        logger.error(f"Check the log file for details: {log_file}")
        logger.error("Some steps failed - please review the output above and rerun failed steps manually.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
