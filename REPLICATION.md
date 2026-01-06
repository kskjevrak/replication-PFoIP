# Replication Guide

This document provides detailed step-by-step instructions for replicating the results from "Probabilistic Forecasting of Imbalance Prices: An Application to the Norwegian Electricity Market".

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Data Preparation](#data-preparation)
4. [Model Training](#model-training)
5. [Generating Predictions](#generating-predictions)
6. [Evaluation](#evaluation)
7. [Troubleshooting](#troubleshooting)
8. [Expected Runtime](#expected-runtime)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 16 GB
- Storage: 10 GB free space
- OS: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)

**Recommended for Faster Training:**
- CPU: 8+ cores
- RAM: 32 GB
- GPU: NVIDIA GPU with 8+ GB VRAM (optional but recommended)
- Storage: 20 GB SSD

### Software Requirements

- Python 3.10 or higher
- Conda (Anaconda or Miniconda) OR pip
- Git (for cloning the repository)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/replication-PFoIP.git
cd replication-PFoIP
```

### Step 2: Create Python Environment

#### Option A: Using Conda (Recommended)

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate tio4900-model

# Verify installation
python -c "import torch; import pandas; import optuna; print('All packages installed successfully!')"
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import pandas; import optuna; print('All packages installed successfully!')"
```

### Step 3: Verify Directory Structure

```bash
# Check that all directories exist
ls -la

# Expected output should include:
# - config/
# - src/
# - run_tuning.py, run_pred.py, etc.
# - README.md, REPLICATION.md
```

## Data Preparation

### Option 1: Using Synthetic Data (Quick Start)

For testing the pipeline without access to original data:

```bash
# Generate synthetic data for all zones
python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2024-04-26

# Verify data was created
ls -la src/data/no1/
# Should see: merged_dataset_no1.parquet
```

Expected output:
```
Generating synthetic data for all zones: ['no1', 'no2', 'no3', 'no4', 'no5']
Date range: 2019-08-25 to 2024-04-26

============================================================
Processing zone: NO1
============================================================
Synthetic data saved to: src/data/no1/merged_dataset_no1.parquet
  Shape: (41040, 5)
  Date range: 2019-08-25 00:00:00+02:00 to 2024-04-26 23:00:00+02:00
  Columns: ['premium', 'mFRR_price_up', 'mFRR_price_down', 'mFRR_vol_up', 'mFRR_vol_down']
```

### Option 2: Using Original Data

If you have obtained the original data (see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)):

1. Convert data to Parquet format with required columns
2. Place files in: `src/data/{zone}/merged_dataset_{zone}.parquet`
3. Verify data structure:

```python
import pandas as pd

# Load and inspect data
data = pd.read_parquet('src/data/no1/merged_dataset_no1.parquet')

print(f"Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print(f"Date range: {data.index.min()} to {data.index.max()}")
print(f"Missing values:\n{data.isnull().sum()}")

# Expected columns: ['premium', 'mFRR_price_up', 'mFRR_price_down',
#                    'mFRR_vol_up', 'mFRR_vol_down']
# Index: DatetimeIndex with timezone Europe/Oslo
```

## Model Training

### Step 1: Configure Training Parameters

Edit `config/default_config.yml` if needed:

```yaml
training:
  batch_size: 16          # Reduce if running out of memory
  max_epochs: 1500        # Maximum training epochs
  patience: 50            # Early stopping patience

tuning:
  n_trials: 128           # Number of Optuna trials (reduce for faster testing)
  cross_validation_folds: 6

prediction:
  window_size: 730        # Days of historical data (2 years)
  hours: 24               # Forecast horizon
```

### Step 2: Hyperparameter Tuning

Run hyperparameter optimization for each model type:

#### Deep Distributional Neural Network (DDNN)

```bash
# Tune for zone NO1 with JSU distribution (interactive mode)
python run_tuning.py --zone no1 --distribution jsu --run-id replication_run_001

# Or non-interactive mode (recommended for cluster/automated execution)
python run_tuning.py --zone no1 --distribution jsu --run-id replication_run_001 --no-confirm

# The script will:
# 1. Load and preprocess data
# 2. Run 128 Optuna trials with 6-fold time-series cross-validation
# 3. Save best hyperparameters to results/models/no1_jsu_replication_run_001/
```

**Expected output:**
```
Zone: no1, distribution: jsu, run_id: replication_run_001
Press enter to continue or any key to exit...

[I 2024-11-30 12:00:00,000] A new study created...
[I 2024-11-30 12:05:00,000] Trial 0 finished with value: -45.23
[I 2024-11-30 12:10:00,000] Trial 1 finished with value: -42.15
...
[I 2024-11-30 18:00:00,000] Trial 127 finished with value: -38.92

Best trial:
  Value: -38.92
  Params:
    hidden_layers: [256, 128]
    learning_rate: 0.0023
    batch_size: 16
    ...

Optimization complete! Best parameters saved to results/models/no1_jsu_replication_run_001/
```

#### Linear Quantile Regression (LQR)

```bash
python run_lqa_tuning.py no1 replication_lqr_001
```

#### XGBoost Quantile Regression

```bash
python run_xgb_tuning.py no1 replication_xgb_001
```

### Step 3: Train for All Zones (Optional)

To replicate full paper results, train for all five zones:

```bash
# DDNN - All zones
for zone in no1 no2 no3 no4 no5; do
    python run_tuning.py $zone jsu replication_run_001
done

# LQR - All zones
for zone in no1 no2 no3 no4 no5; do
    python run_lqa_tuning.py $zone replication_lqr_001
done

# XGBoost - All zones
for zone in no1 no2 no3 no4 no5; do
    python run_xgb_tuning.py $zone replication_xgb_001
done
```

## Generating Predictions

### Step 1: Single Date Prediction

Generate forecasts for a specific date:

```bash
# DDNN prediction
python run_pred.py no1 jsu replication_run_001 --start-date 2024-04-26 --end-date 2024-04-26

# LQR prediction
python run_lqa_pred.py no1 replication_lqr_001 --start-date 2024-04-26 --end-date 2024-04-26

# XGBoost prediction
python run_xgb_pred.py no1 replication_xgb_001 --start-date 2024-04-26 --end-date 2024-04-26
```

**Expected output:**
```
Loading model from results/models/no1_jsu_replication_run_001/
Predicting for date: 2024-04-26
Loading data for zone no1...
Loaded 17520 records from 2022-04-26 to 2024-04-27
Preparing features...
Using 729 days for training, 1 day for forecasting
After feature selection: 240 features selected
Loading model...
Generating predictions...
Saving predictions...
Predictions saved to: results/forecasts/no1_jsu_replication_run_001/forecast_2024-04-26.parquet

Success! Forecast generated for 2024-04-26
```

### Step 2: Multi-Date Rolling Predictions

Generate forecasts for a date range:

```bash
# Generate forecasts for one month
python run_pred.py no1 jsu replication_run_001 \
    --start-date 2024-04-26 \
    --end-date 2024-05-26
```

This will generate forecasts for each day in the range and save them to:
```
results/forecasts/no1_jsu_replication_run_001/
    forecast_2024-04-26.parquet
    forecast_2024-04-27.parquet
    ...
    forecast_2024-05-26.parquet
```

### Step 3: Verify Prediction Output

Check that forecasts were generated:

```python
import pandas as pd

# Load a forecast
forecast = pd.read_parquet('results/forecasts/no1_jsu_replication_run_001/forecast_2024-04-26.parquet')

print(f"Forecast shape: {forecast.shape}")
print(f"Columns: {list(forecast.columns)}")

# Expected: 24 rows (one per hour), columns for distribution parameters and quantiles
```

## Evaluation

### Step 1: Calculate Performance Metrics

```python
from src.evaluation.metrics import calculate_all_metrics

# Calculate metrics for the forecast period
metrics = calculate_all_metrics(
    distribution='jsu',
    nr='replication_run_001',
    zone='no1',
    date_range=('2024-04-26', '2024-05-26'),
    target_column='premium'
)

print(f"Average CRPS: {metrics['mean_crps']:.2f}")
print(f"Average Quantile Score: {metrics['mean_quantile_score']:.2f}")
print(f"Coverage 90%: {metrics['coverage_90']:.2%}")
```

### Step 2: Visualize Forecasts

```python
from src.evaluation.visualization import plot_probabilistic_forecast

# Plot forecast for specific date
plot_probabilistic_forecast(
    zone='no1',
    distribution='jsu',
    run_id='replication_run_001',
    date='2024-04-26'
)
```

### Step 3: Compare Models

Generate comparison table:

```python
# Compare DDNN, LQR, and XGBoost
models = [
    ('DDNN-JSU', 'jsu', 'replication_run_001'),
    ('LQR', 'lqr', 'replication_lqr_001'),
    ('XGBoost', 'xgb', 'replication_xgb_001')
]

results = []
for name, dist, run_id in models:
    metrics = calculate_all_metrics(
        distribution=dist,
        nr=run_id,
        zone='no1',
        date_range=('2024-04-26', '2024-05-26')
    )
    results.append({
        'Model': name,
        'CRPS': metrics['mean_crps'],
        'Quantile Score': metrics['mean_quantile_score'],
        'Coverage 90%': metrics['coverage_90']
    })

import pandas as pd
comparison_df = pd.DataFrame(results)
print(comparison_df.to_string(index=False))
```

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory Error

**Symptom:** `RuntimeError: CUDA out of memory` or system freezes during training

**Solution:**
```yaml
# In config/default_config.yml, reduce:
training:
  batch_size: 8  # or even 4
```

Or force CPU-only execution:
```python
# Add at start of script
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

#### Issue 2: Data Not Found Error

**Symptom:** `FileNotFoundError: Data file src/data/no1/merged_dataset_no1.parquet not found`

**Solution:**
```bash
# Generate synthetic data first
python src/data/synthetic_data.py --zone no1 --start-date 2019-08-25 --end-date 2024-04-26
```

#### Issue 3: Module Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```bash
# Ensure you're in the project root directory
cd /path/to/replication-PFoIP

# Verify Python can find modules
python -c "import sys; print('\n'.join(sys.path))"

# Should show current directory (.) in the path
```

#### Issue 4: Slow Training

**Symptom:** Training takes very long (> 8 hours for one trial)

**Solution:**
```yaml
# Reduce tuning trials in config/default_config.yml:
tuning:
  n_trials: 32  # Instead of 128 for testing

# Or reduce dataset size for testing:
prediction:
  window_size: 365  # Instead of 730
```

### Getting Help

If you encounter issues not covered here:

1. Check the [README.md](README.md) for basic usage
2. Review [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for data questions
3. Open an issue on GitHub with:
   - Error message
   - Steps to reproduce
   - System information (OS, Python version)
   - Config file settings

## Expected Runtime

Approximate times on recommended hardware (8-core CPU, 32GB RAM, NVIDIA GPU):

| Task | Single Zone | All 5 Zones |
|------|------------|-------------|
| Synthetic data generation | < 1 min | < 5 min |
| DDNN hyperparameter tuning (128 trials) | 6-8 hours | 30-40 hours |
| LQR hyperparameter tuning | 2-3 hours | 10-15 hours |
| XGBoost hyperparameter tuning | 3-4 hours | 15-20 hours |
| Single day prediction | < 1 min | < 5 min |
| 30-day rolling prediction | 5-10 min | 25-50 min |

**Note:** Times may vary significantly based on hardware and configuration.

## Minimal Replication Test

For a quick test to verify everything works:

```bash
# 1. Generate minimal synthetic data
python src/data/synthetic_data.py --zone no1 --start-date 2023-01-01 --end-date 2024-04-26

# 2. Quick tuning test (only 10 trials)
# Edit config/default_config.yml: set n_trials: 10
python run_tuning.py no1 jsu quick_test

# 3. Generate one prediction
python run_pred.py no1 jsu quick_test --start-date 2024-04-26 --end-date 2024-04-26

# 4. Verify output exists
ls results/forecasts/no1_jsu_quick_test/
```

Expected total time: 30-45 minutes

## Full Replication

To fully replicate all paper results:

```bash
# 1. Generate or obtain all zone data
python src/data/synthetic_data.py --all-zones

# 2. Tune all models for all zones (will take several days)
./scripts/train_all_zones.sh  # If provided, or run manually

# 3. Generate all predictions for test period
./scripts/predict_all_zones.sh

# 4. Calculate and export all metrics
python scripts/calculate_all_metrics.py

# 5. Generate all figures
python scripts/generate_figures.py
```

Expected total time: 5-7 days on recommended hardware

---

**Questions?** See the main [README.md](README.md) or open an issue on GitHub.

*Last updated: January 2025*
