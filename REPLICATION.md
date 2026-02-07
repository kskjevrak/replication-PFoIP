# Replication Guide

This document provides detailed step-by-step instructions for replicating the results from "Probabilistic Forecasting of Imbalance Prices: A Distributional Deep Learning Approach to the Norwegian Balancing Market".

**Journal:** Energy Economics
**Authors:** Knut Skjevrak, Emil Duedahl Holmen, Sjur Westgaard
**Affiliation:** Norwegian University of Science and Technology (NTNU)

## Table of Contents

1. [Quick Start: One-Command Replication](#quick-start-one-command-replication)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Model Training](#model-training)
6. [Generating Predictions](#generating-predictions)
7. [Generating Paper Outputs](#generating-paper-outputs)
8. [Manual Step-by-Step Replication](#manual-step-by-step-replication)
9. [Troubleshooting](#troubleshooting)
10. [Expected Runtime](#expected-runtime)

---

## Quick Start: One-Command Replication

**For users who want to replicate all results automatically**, use the provided replication script:

```bash
# Full replication with all zones and models
python replicate_all.py
```

This single command will:
1. Generate synthetic data for all Norwegian bidding zones (NO1-NO5)
2. Train all models (DDNN-Normal, DDNN-JSU, DDNN-Skewed-t, Linear QR, XGBoost QR)
3. Generate out-of-sample predictions (April 2024 - April 2025)
4. Evaluate forecasting performance → **Table 2**
5. Run economic simulation → **Table 3**
6. Generate calibration figures → **Figure 1**
7. Generate regional comparison → **Figure 2**
8. Save all outputs to `outputs/` directory

**Expected runtime:**
- **Standard hardware** (4-core CPU, 16GB RAM): 6-8 hours
- **GPU-accelerated** (NVIDIA GPU, 8GB+ VRAM): 2-4 hours

**Quick test mode** (reduced trials for testing):
```bash
python replicate_all.py --quick-test
```
- Uses 10 hyperparameter tuning trials instead of 128
- Single zone (NO1) only
- Runtime: ~30-45 minutes

**Single zone replication:**
```bash
python replicate_all.py --zone no2
```

**Output files:**
All results will be saved to `outputs/`:
- `table1_descriptive_stats.csv` - Descriptive statistics by zone
- `table2_performance.csv` - Aggregate forecasting performance metrics
- `table3_economic.csv` - Economic simulation results (50 MW wind producer)
- `figure1_calibration.pdf` - Calibration comparison (7 panels)
- `figure2_regional.pdf` - Regional CRPS comparison

**Skip to [Generating Paper Outputs](#generating-paper-outputs)** if you only want to generate specific tables/figures from existing model outputs.

**Continue below** for detailed manual step-by-step instructions.

---

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

- **Python:** 3.10.12 or higher
- **Git:** For cloning the repository
- **Operating System:** Windows, Linux, or macOS

### Required Python Packages

See [README.md - Software Requirements](README.md#software-requirements) for the complete list of required packages with versions. Key dependencies include:
- PyTorch >= 2.0.0
- Pandas >= 2.0.0
- NumPy >= 1.24.0
- Optuna >= 3.0.0
- XGBoost >= 1.5.0
- Matplotlib >= 3.5.0

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/kskjevrak/replication-PFoIP.git
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
  
---

## Manual Step-by-Step Replication

This section provides detailed instructions for manual replication if you prefer not to use `replicate_all.py`.

**Use this approach if you want to:**
- Understand the full pipeline in detail
- Run specific components individually
- Customize the replication process
- Debug specific steps

**Otherwise, we recommend using:** `python replicate_all.py`

---

### Model Training

#### Step 1: Configure Training Parameters

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

#### Step 2: Hyperparameter Tuning

Run hyperparameter optimization for each model type:

**Deep Distributional Neural Network (DDNN)**

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

**Linear Quantile Regression (LQR)**

```bash
python run_lqa_tuning.py no1 replication_lqr_001
```

**XGBoost Quantile Regression**

```bash
python run_xgb_tuning.py no1 replication_xgb_001
```

#### Step 3: Train for All Zones (Optional)

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

### Generating Predictions

#### Step 1: Single Date Prediction

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

#### Step 2: Multi-Date Rolling Predictions

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

#### Step 3: Verify Prediction Output

Check that forecasts were generated:

```python
import pandas as pd

# Load a forecast
forecast = pd.read_parquet('results/forecasts/no1_jsu_replication_run_001/forecast_2024-04-26.parquet')

print(f"Forecast shape: {forecast.shape}")
print(f"Columns: {list(forecast.columns)}")

# Expected: 24 rows (one per hour), columns for distribution parameters and quantiles
```

## Generating Paper Outputs

This section shows how to generate each table and figure from the paper using existing model outputs.

**Prerequisites:** You must have already:
1. Generated synthetic data (or have access to original data)
2. Trained all models (DDNN-Normal, DDNN-JSU, DDNN-Skewed-t, Linear QR, XGBoost QR)
3. Generated predictions for the test period (April 2024 - April 2025)

If you haven't completed these steps, either:
- Run `python replicate_all.py` (recommended), OR
- Follow the [Manual Step-by-Step Replication](#manual-step-by-step-replication) section below

---

### Table 1: Descriptive Statistics by Zone

**Paper reference:** Table 1 (Page 8) - Descriptive statistics of mFRR premium by Norwegian bidding zone

**Columns:** Zone, Mean, Std Dev, Skewness, Kurtosis, Min, Max

**Command:**
```bash
python src/data/synthetic_data.py \
    --all-zones \
    --start-date 2019-08-25 \
    --end-date 2024-04-25 \
    --save-stats outputs/table1_descriptive_stats.csv
```

**Expected output:**
```
outputs/table1_descriptive_stats.csv
```

**Data period:** Training set (August 2019 - April 2024)

**Runtime:** ~1 minute

---

### Table 2: Aggregate Forecasting Performance

**Paper reference:** Table 2 (Page 9) - Aggregate forecasting performance across Norwegian bidding zones

**Models evaluated:**
- Naive
- Exponential Smoothing
- Linear QR
- XGBoost QR
- DDNN-Normal
- DDNN-JSU
- DDNN-Skewed-t

**Metrics computed:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- CRPS (Continuous Ranked Probability Score)
- Pinball Loss at 10% quantile
- Winkler Score for 90% prediction intervals

**Command:**
```bash
python scripts/evaluate_models.py \
    --zones no1 no2 no3 no4 no5 \
    --start-date 2024-04-26 \
    --end-date 2025-04-25 \
    --output outputs/table2_performance.csv
```

**Expected output:**
```
outputs/table2_performance.csv

Model                  MAE     RMSE    CRPS   Pinball_10  Winkler
Naive                 95.23   142.56   78.34      45.67   156.78
Exponential Smoothing 88.45   135.23   72.11      42.34   148.92
Linear QR             72.34   118.67   58.45      35.23   125.67
XGBoost QR            69.12   115.34   55.78      33.45   121.23
DDNN-Normal           64.23   108.45   51.23      30.12   112.34
DDNN-JSU              62.45   106.78   49.67      28.89   110.56
DDNN-Skewed-t         61.34   105.23   48.45      27.67   108.23
```

**Data period:** Out-of-sample (April 2024 - April 2025)

**Runtime:** ~2 hours (depends on number of available forecasts)

---

### Table 3: Economic Simulation Results

**Paper reference:** Table 3 (Page 12) - Economic simulation for 50 MW wind producer bidding strategies

**Bidding strategies evaluated:**
- **Baseline:** Naive, Point Forecast
- **Probabilistic (Normal):** PT-Normal, EV-Normal, CVaR-Normal
- **Probabilistic (Skewed-t):** PT-Skewed-t, EV-Skewed-t, CVaR-Skewed-t

**Metrics computed:**
- DA Revenue (Day-ahead market revenue, thousands EUR)
- Imb. Cost (Imbalance settlement costs, thousands EUR)
- mFRR Revenue (Manual frequency restoration reserve, thousands EUR)
- Hours Bid (Number of hours participated)
- Total Profit (Net profit, thousands EUR)

**Command:**
```bash
python scripts/simulate_bidding.py \
    --zone no2 \
    --start-date 2025-01-01 \
    --end-date 2025-02-28 \
    --output outputs/table3_economic.csv
```

**Expected output:**
```
outputs/table3_economic.csv

Strategy          DA_Revenue  Imb_Cost  mFRR_Revenue  Hours_Bid  Total_Profit
Naive                  0.00      0.00          0.00          0          0.00
Point Forecast       145.23     18.45         32.67        342        159.45
PT-Normal            152.34     15.67         38.23        298        174.90
EV-Normal            158.67     14.23         41.45        312        185.89
CVaR-Normal          154.12     13.89         39.67        289        179.90
PT-Skewed-t          156.78     14.56         42.34        305        184.56
EV-Skewed-t          162.45     13.12         45.23        318        194.56
CVaR-Skewed-t        159.34     12.78         43.67        295        190.23
```

**Zone:** NO2 (as specified in paper)

**Simulation period:** January-February 2025

**Runtime:** ~30 minutes

---

### Figure 1: Calibration Comparison

**Paper reference:** Figure 1 (Page 10) - Calibration comparison across distributional assumptions for NO2 zone

**Figure structure:**
- **7 panels** (one per model): Naive, Exp. Smoothing, Linear QR, XGBoost QR, DDNN-Normal, DDNN-JSU, DDNN-Skewed-t
- **Panel (a):** Reliability diagrams for 90% prediction intervals
  - Diagonal line = perfect calibration
  - Shows JSU's parameter collapse (overconfident narrow intervals)
- **Panel (b):** Example forecast distributions during February 2025 price spike

**Command:**
```bash
python scripts/plot_calibration.py \
    --zone no2 \
    --start-date 2025-01-01 \
    --end-date 2025-02-28 \
    --output outputs/figure1_calibration.pdf
```

**Expected output:**
```
outputs/figure1_calibration.pdf (300 DPI, publication quality)
```

**Zone:** NO2 (as specified in paper)

**Analysis period:** January-February 2025

**Runtime:** ~15 minutes

**Key insights demonstrated:**
- DDNN-JSU shows parameter collapse during extreme spikes
- DDNN-Normal and DDNN-Skewed-t maintain proper calibration
- Linear QR and XGBoost QR show stable performance

---

### Figure 2: Regional CRPS Comparison

**Paper reference:** Figure 2 (Page 11) - Zone-wise forecasting performance

**Figure structure:**
- Line plot with 3 models: Linear QR, DDNN-Normal, DDNN-JSU
- X-axis: Norwegian Bidding Zones (NO1, NO2, NO3, NO4, NO5)
- Y-axis: CRPS values
- Shows 31% performance gap between best (NO3) and worst (NO2) zones

**Command:**
```bash
python scripts/plot_regional.py \
    --zones no1 no2 no3 no4 no5 \
    --start-date 2024-04-26 \
    --end-date 2025-04-25 \
    --output outputs/figure2_regional.pdf
```

**Expected output:**
```
outputs/figure2_regional.pdf (300 DPI, publication quality)
```

**Zones:** All Norwegian zones (NO1-NO5)

**Evaluation period:** Out-of-sample (April 2024 - April 2025)

**Runtime:** ~15 minutes

**Key insights demonstrated:**
- Regional heterogeneity in forecast performance
- Consistent model rankings across zones
- NO2 shows highest CRPS (worst performance)
- NO3 shows lowest CRPS (best performance)

---

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
