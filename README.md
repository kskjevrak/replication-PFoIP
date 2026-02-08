# Probabilistic Forecasting of Imbalance Prices

**Replication package for:** "Probabilistic Forecasting of Imbalance Prices: A Distributional Deep Learning Approach to the Norwegian Balancing Market"

**Authors:** Knut Skjevrak, Emil Duedahl Holmen, Sjur Westgaard
**Affiliation:** Norwegian University of Science and Technology (NTNU)
**Journal:** Submitted to Applied Energy

This repository contains the complete replication package for reproducing the results presented in the paper.

---

## Table of Contents

- [Overview](#overview)
- [Software Requirements](#software-requirements)
- [Quick Start: One-Command Replication](#quick-start-one-command-replication)
- [File-to-Output Mapping](#file-to-output-mapping)
- [Synthetic Data Deviations](#synthetic-data-deviations)
- [Repository Structure](#repository-structure)
- [Data Availability](#data-availability)
- [Detailed Usage](#detailed-usage)
  - [Installation](#installation)
  - [Synthetic Data Generation](#synthetic-data-generation)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Generating Predictions](#generating-predictions)
- [Models](#models)
- [Configuration](#configuration)
- [Expected Outputs](#expected-outputs)
- [System Requirements](#system-requirements)
- [Replication Guide](#replication-guide)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Overview

This repository implements three probabilistic forecasting approaches for electricity imbalance price premiums in the Norwegian mFRR (manual Frequency Restoration Reserve) market:

1. **Deep Distributional Neural Networks (DDNN)** - Parametric probabilistic forecasts using Neural Networks with Johnson's SU, Normal, or Skewed Student's t distributions
2. **Linear Quantile Regression (LQR)** - Non-parametric quantile forecasting via regularized linear models
3. **XGBoost Quantile Regression** - Gradient boosting approach for quantile estimation

The primary focus is on the DDNN approach, which provides full probabilistic forecasts by modeling distribution parameters directly.

### Key Features

- Hourly electricity price forecasting for 5 Norwegian bidding zones (NO1-NO5)
- Automated hyperparameter optimization using Optuna
- Rolling-window validation and prediction
- Comprehensive evaluation metrics (CRPS, Pinball Loss, Winkler Score)
- Support for both CPU and GPU training

---

## Software Requirements

### Required Software
- **Python:** 3.10.12 or higher
- **Git:** For cloning the repository
- **Operating System:** Windows, Linux, or macOS

### Python Package Versions

The following packages are required (minimum versions specified):

**Core Scientific Computing:**
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- SciPy >= 1.7.0

**Machine Learning:**
- scikit-learn >= 1.0.0

**Deep Learning:**
- PyTorch >= 2.0.0
- torchvision >= 0.15.0

**Hyperparameter Optimization:**
- Optuna >= 3.0.0

**Data Processing:**
- pyarrow >= 10.0.0
- fastparquet >= 2023.0.0
- joblib >= 1.1.0
- PyYAML >= 6.0

**Gradient Boosting:**
- XGBoost >= 1.5.0

**Visualization:**
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0

**Utilities:**
- tqdm >= 4.60.0

**Optional (Development/Analysis):**
- Jupyter >= 1.0.0
- JupyterLab >= 3.0.0
- ipywidgets >= 7.6.0
- ipykernel >= 6.0.0
- black >= 21.5b2
- flake8 >= 3.9.2
- pytest >= 6.0.0

### Installing Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

For GPU support with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Quick Start: One-Command Replication

To reproduce all results with synthetic data:

```bash
python replicate_all.py
```

This command will:
1. **Generate synthetic data** matching statistical properties of the original dataset
2. **Train all models** (DDNN-Normal, DDNN-JSU, DDNN-Skewed-t, Linear Quantile Regression, XGBoost)
3. **Evaluate forecasting performance** and compute metrics for all tables
4. **Run economic simulation** for profitability analysis
5. **Generate all figures** from the paper
6. **Save outputs** to `outputs/` directory

**Expected runtime:** 6-8 hours on standard hardware (4-core CPU, 16GB RAM)
**GPU acceleration:** 2-4 hours with NVIDIA GPU (8GB+ VRAM)

---

## File-to-Output Mapping

This table maps paper outputs to the scripts that generate them:

| Paper Output | Description | Script | Output File | Runtime |
|-------------|-------------|---------|-------------|---------|
| **Table 1** | Descriptive statistics by zone | `src/data/synthetic_data.py` | `outputs/table1_descriptive_stats.csv` | ~1 min |
| **Table 2** | Aggregate forecasting performance metrics | `scripts/evaluate_models.py` | `outputs/table2_performance.csv` | ~2 hours |
| **Table 3** | Economic simulation for 50 MW wind producer | `scripts/simulate_bidding.py` | `outputs/table3_economic.csv` | ~30 min |
| **Figure 1** | Calibration comparison (7 panels: reliability + forecasts) | `scripts/plot_calibration.py` | `outputs/figure1_calibration.pdf` | ~15 min |
| **Figure 2** | Regional CRPS comparison across NO1-NO5 | `scripts/plot_regional.py` | `outputs/figure2_regional.pdf` | ~15 min |

### Reproducing Individual Outputs

To generate specific outputs independently:

```bash
# Table 1: Descriptive Statistics
python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2024-04-25 --save-stats outputs/table1_descriptive_stats.csv

# Table 2: Forecasting Performance (MAE, RMSE, CRPS, Pinball, Winkler)
python scripts/evaluate_models.py --zones no1 no2 no3 no4 no5 --output outputs/table2_performance.csv

# Table 3: Economic Simulation Results
python scripts/simulate_bidding.py --zone no2 --output outputs/table3_economic.csv

# Figure 1: Calibration Comparison (7 panels)
python scripts/plot_calibration.py --zone no2 --output outputs/figure1_calibration.pdf

# Figure 2: Regional CRPS Performance
python scripts/plot_regional.py --zones no1 no2 no3 no4 no5 --output outputs/figure2_regional.pdf
```

---

## Synthetic Data Deviations

**Important:** Results obtained using the provided synthetic data will deviate from published values due to data protection requirements.

### Why Synthetic Data?

The original Norwegian electricity market data is proprietary and licensed from Volue AS. It cannot be publicly shared due to:
- Commercial sensitivity of market data
- Licensing restrictions prohibiting redistribution
- Market participant confidentiality

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for details on accessing the original data.

### Limitations of Synthetic Data

The synthetic data generator preserves key statistical properties but cannot replicate:

1. **Simplified correlation structure:** Synthetic generator preserves marginal distributions but simplifies temporal dependencies and cross-zone correlations
2. **Approximated tail behavior:** Extreme values capped at 99.9th percentile to prevent unrealistic outliers
3. **Smoothed structural breaks:** The November 2021 single-price transition is smoothed in synthetic data
4. **Missing market microstructure:** Strategic bidding behavior and market dynamics are approximated

### Expected Deviations

When using synthetic data, expect the following deviations from published results:

| Metric Category | Expected Deviation | Preserved Property |
|----------------|-------------------|-------------------|
| Point forecast metrics (MAE, RMSE) | 15-20% deviation | Relative model rankings |
| Probabilistic metrics (CRPS, Pinball) | 15-25% deviation | Relative model rankings |
| Economic simulation profits | 20-30% deviation | Sign of profits |
| Distribution parameters | Moderate differences | Qualitative behavior |
| Model rankings | Consistent | Best/worst performers preserved |
| Qualitative findings | Fully reproducible | All insights preserved |

**Key Point:** The synthetic data is designed to enable **methodological replication** while protecting proprietary market data. Quantitative metrics will differ, but qualitative findings and relative model performance remain consistent.

---

## Repository Structure

```
replication-PFoIP/
├── README.md                    # This file
├── REPLICATION.md              # Detailed step-by-step instructions
├── DATA_AVAILABILITY.md        # Data access and restrictions
├── LICENSE                     # MIT License
├── requirements.txt            # Python dependencies (pip)
├── environment.yml             # Conda environment specification
├── replicate_all.py           # One-command replication script
│
├── config/
│   └── default_config.yml      # Model and training configuration
│
├── data/
│   └── synthetic/             # Generated synthetic data (created on first run)
│
├── src/
│   ├── data/
│   │   ├── loader.py          # Data loading and preprocessing
│   │   └── synthetic_data.py  # Synthetic data generator
│   ├── models/
│   │   ├── neural_nets.py     # DDNN implementations
│   │   ├── distributions.py   # Distribution layers (JSU, Normal, Skewed-t)
│   │   ├── layers.py          # Custom neural network layers
│   │   ├── linear_quantile.py # Linear quantile regression
│   │   └── xgboost_quantile.py # XGBoost quantile regression
│   ├── training/
│   │   ├── optuna_tuner.py    # Hyperparameter optimization
│   │   ├── prediction.py      # Prediction pipeline
│   │   └── rolling_pred.py    # Rolling-window forecasts
│   └── evaluation/
│       ├── metrics.py         # Evaluation metrics (CRPS, Pinball, etc.)
│       └── visualization.py   # Plotting utilities
│
├── scripts/
│   ├── evaluate_models.py     # Generate Table 2
│   ├── simulate_bidding.py    # Generate Table 3
│   ├── plot_calibration.py    # Generate Figure 1
│   ├── plot_regional.py       # Generate Figure 2
│   └── verify_installation.py # Installation verification
│
├── outputs/                   # Generated results (created by scripts)
│   ├── table1_descriptive_stats.csv
│   ├── table2_performance.csv
│   ├── table3_economic.csv
│   ├── figure1_calibration.pdf
│   └── figure2_regional.pdf
│
└── results/                   # Model checkpoints and forecasts
    ├── models/               # Trained models and hyperparameters
    ├── forecasts/            # Generated probabilistic forecasts
    └── logs/                 # Training and execution logs
```

---

## Data Availability

**Important:** The original Norwegian electricity market data is proprietary and cannot be publicly shared.

- **Original data source:** Volue AS (commercial license)
- **Public alternative:** Synthetic data generator provided
- **Data access:** See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for:
  - How to request original data from Volue AS
  - Alternative public data sources
  - Synthetic data generation instructions
  - Limitations and expected deviations

---

## Detailed Usage

### Installation

#### Option 1: Conda (Recommended)

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate tio4900-model
```

#### Option 2: pip

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

#### Verify Installation

Test that the environment is set up correctly:

```bash
python scripts/verify_installation.py
```

This script checks that all required packages are installed and imports work correctly.

---

### Synthetic Data Generation

For manual data generation (not needed if using `replicate_all.py`):

```bash
# Generate data for all zones (2019-08-25 to 2025-04-25)
python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2025-04-25

# Or for a single zone
python src/data/synthetic_data.py --zone no1 --start-date 2019-08-25 --end-date 2025-04-25
```

---

### Hyperparameter Tuning

For manual hyperparameter tuning (not needed if using `replicate_all.py`):

```bash
# Tune DDNN with JSU distribution for zone NO1
python run_tuning.py --zone no1 --distribution jsu --run-id test_001 --no-confirm

# With custom number of trials (default is 128)
python run_tuning.py --zone no1 --distribution jsu --run-id test_001 --n-trials 32 --no-confirm
```

**Expected runtime:**
- With GPU: ~2-4 hours (128 trials)
- With CPU: ~8-12 hours (128 trials)

---

### Generating Predictions

```bash
# Generate predictions for a date range
python run_pred.py --zone no1 --distribution jsu --run-id test_001 --start-date 2024-04-26 --end-date 2024-04-26
```

**Expected runtime:** ~2-5 minutes per day

---

## Configuration

The [default_config.yml](config/default_config.yml) file controls all aspects of the modeling pipeline:

### Data Configuration

```yaml
data:
  zones: ['no1', 'no2', 'no3', 'no4', 'no5']
  start_date: '2019-08-25'          # Training data start
  end_date: '2024-04-26'             # Training data end
  validation_start: '2023-04-25'     # Validation split date
```

### Model Configuration

```yaml
model:
  distribution: 'JSU'                # Options: 'JSU', 'Normal', 'skewt'
  architectures:
    - hidden_layers: [256, 128]      # Network architecture options
    - hidden_layers: [512, 256]
```

### Training Configuration

```yaml
training:
  batch_size: 16
  max_epochs: 1500
  patience: 50                       # Early stopping patience
  learning_rate_range: [1e-5, 1e-2]
```

### Tuning Configuration

```yaml
tuning:
  n_trials: 128                      # Optuna optimization trials
  cross_validation_folds: 6          # Time-series CV folds
```

### Prediction Configuration

```yaml
prediction:
  window_size: 730                   # Historical window (days)
  hours: 24                          # Forecast horizon
  test_start: "2024-04-26"
  test_end: "2025-04-26"
```

You can create custom configuration files and specify them using the `--config` flag:

```bash
python run_tuning.py --zone no1 --distribution jsu --run-id custom --config config/my_config.yml
```

---

## Usage

### Synthetic Data Generation

The synthetic data generator creates realistic mFRR price data with the following characteristics:

- **Hourly frequency** with Europe/Oslo timezone
- **Features:** mFRR up/down prices and volumes, spot prices, weather, consumption forecasts
- **Statistical properties:** Match descriptive statistics from thesis (Appendix B)
- **Temporal patterns:** Daily/weekly cycles, seasonal variation, autocorrelation, price spikes

```bash
# Generate for all zones
python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2025-04-25

# Custom output directory
python src/data/synthetic_data.py --zone no1 --output-dir ./my_data

# Set random seed for reproducibility
python src/data/synthetic_data.py --zone no1 --seed 12345
```

**Output:** Parquet files saved to `src/data/{zone}/merged_dataset_{zone}.parquet`

### Hyperparameter Tuning

The tuning script optimizes DDNN hyperparameters using Optuna with time-series cross-validation:

```bash
# Basic usage
python run_tuning.py --zone no1 --distribution jsu --run-id replication_001

# Non-interactive mode (for cluster/automated execution)
python run_tuning.py --zone no1 --distribution jsu --run-id replication_001 --no-confirm

# Specify number of trials and config
python run_tuning.py --zone no1 --distribution jsu --run-id test \
  --n-trials 32 --config config/custom_config.yml --no-confirm

# Legacy positional arguments also supported
python run_tuning.py no1 jsu replication_001
```

**Optimized Hyperparameters:**
- Network architecture (hidden layer sizes)
- Activation functions (ReLU, Tanh, Softplus)
- Dropout rates
- Batch normalization
- Learning rate
- Batch size
- L2 regularization

**Outputs:**
- Best hyperparameters: `results/models/{zone}_{distribution}_{run_id}/best_params.yaml`
- Optuna study database: `results/models/{zone}_{distribution}_{run_id}/optuna_study.db`
- Training logs: `results/logs/{zone}_{distribution}_{run_id}.log`

### Generating Predictions

Generate probabilistic forecasts for specified date ranges:

```bash
# Single day prediction
python run_pred.py --zone no1 --distribution jsu --run-id replication_001 \
  --start-date 2024-04-26 --end-date 2024-04-26

# Multi-day rolling predictions
python run_pred.py --zone no1 --distribution jsu --run-id replication_001 \
  --start-date 2024-04-26 --end-date 2024-05-01

# Parallel processing with multiple workers
python run_pred.py --zone no1 --distribution jsu --run-id replication_001 \
  --start-date 2024-04-26 --end-date 2024-05-01 --workers 4

# Legacy positional arguments
python run_pred.py no1 jsu replication_001 --start-date 2024-04-26 --end-date 2024-04-26
```

**Outputs:**
- Probabilistic forecasts: `results/forecasts/df_forecasts_{zone}_{distribution}_{run_id}/{date}_forecast.csv`
- Distribution parameters: `results/forecasts/distparams_{zone}_{distribution}_{run_id}/{date}_params.csv`
- Prediction logs: `results/logs/prediction_{zone}_{distribution}_{run_id}.log`

---

## Models

### 1. Deep Distributional Neural Networks (DDNN)

The primary model architecture consists of:

- **Shared hidden layers:** Feed-forward network with configurable depth and width
- **Separate parameter heads:** Individual output layers for each distribution parameter
- **Distribution layer:** Transforms raw outputs to valid distribution parameters

**Supported Distributions:**

#### Johnson's SU (JSU)
- **Parameters:** location (μ), scale (σ), skewness (γ), tailweight (δ)
- **Best for:** Heavy-tailed, skewed price distributions with extreme spikes
- **Note:** May exhibit parameter collapse under certain conditions (see Troubleshooting)

#### Normal (Gaussian)
- **Parameters:** location (μ), scale (σ)
- **Best for:** Baseline comparisons, stable markets

#### Skewed Student's t
- **Parameters:** location (μ), scale (σ), skewness (a), tailweight (b)
- **Best for:** Heavy-tailed distributions with moderate skewness

**Loss Function:** Negative log-likelihood with optional L2 regularization

### 2. Linear Quantile Regression (LQR)

Ridge-regularized linear regression for quantile forecasting at specified probability levels (e.g., 0.05, 0.25, 0.50, 0.75, 0.95).

### 3. XGBoost Quantile Regression

Gradient boosting trees with quantile loss for non-linear quantile forecasting.

---

## Expected Outputs

### Hyperparameter Tuning

After successful tuning, you should find:

```
results/models/no1_jsu_replication_001/
├── best_params.yaml           # Optimal hyperparameters
├── optuna_study.db            # Full Optuna optimization history
└── scaler_X.pkl               # Feature scaler
```

Example `best_params.yaml`:
```yaml
hidden_layers:
  - units: 256
    activation: relu
    dropout_rate: 0.1
    batch_norm: true
  - units: 128
    activation: tanh
    dropout_rate: 0.05
    batch_norm: false
learning_rate: 0.001
batch_size: 16
l2_regularization: 0.0001
```

### Prediction Outputs

Forecast files contain probabilistic predictions:

```csv
datetime,mean,median,std,q05,q25,q75,q95,realized
2024-04-26 00:00:00+02:00,85.3,82.1,45.2,25.4,58.7,108.6,175.3,78.9
2024-04-26 01:00:00+02:00,82.7,80.5,43.8,24.1,56.3,105.2,170.8,81.2
...
```

**Columns:**
- `datetime`: Forecast timestamp (Europe/Oslo timezone)
- `mean`: Expected value
- `median`: 50th percentile
- `std`: Standard deviation
- `q05, q25, q75, q95`: Quantile predictions
- `realized`: Actual observed value (if available)

See [docs/example_output/](docs/example_output/) for sample files.

---

## System Requirements

### Minimum Requirements

- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 10GB free space
- **Python:** 3.10+

### Recommended for Full Replication

- **CPU:** 16+ cores OR NVIDIA GPU (RTX 3080 or better)
- **RAM:** 32GB
- **Storage:** 50GB free space (for all zones and experiments)
- **GPU:** CUDA-capable with 8GB+ VRAM

### Runtime Estimates

| Task | CPU (16 cores) | GPU (A100) |
|------|----------------|------------|
| Synthetic data generation (all zones) | 2 minutes | 2 minutes |
| Hyperparameter tuning (128 trials) | 8-12 hours | 2-4 hours |
| Single prediction (24 hours) | 2-5 minutes | 30-60 seconds |
| Full test period (365 days) | 12-30 hours | 3-6 hours |

---

## Replication Guide

For complete step-by-step instructions to reproduce the paper's results, see [REPLICATION.md](REPLICATION.md).

**Summary:**
1. Set up environment
2. Generate synthetic data OR obtain real data
3. Run hyperparameter tuning for each zone and distribution
4. Generate rolling predictions for test period
5. Compute evaluation metrics
6. Create visualizations

**Expected total time:** 5-7 days on CPU, 1-2 days on GPU (for full replication across all zones)

---

## Troubleshooting

### Common Issues

#### 1. JSU Distribution Parameter Collapse

**Symptoms:** Scale parameter approaches zero, extreme skewness/tailweight values, poor forecast quality

**Causes:**
- Insufficient training data
- Inappropriate learning rate
- Extreme target values

**Solutions:**
- Increase training window size
- Adjust `learning_rate_range` in config
- Try Normal or skewt distribution
- Increase L2 regularization

#### 2. CUDA Out of Memory

**Symptoms:** `RuntimeError: CUDA out of memory`

**Solutions:**
- Reduce `batch_size` in config
- Use smaller network architecture
- Run on CPU (slower but works)
```bash
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### 3. Import Errors

**Symptoms:** `ModuleNotFoundError` when running scripts

**Solutions:**
- Verify environment activation: `conda activate tio4900-model`
- Run verification script: `python scripts/verify_installation.py`
- Reinstall environment: `conda env remove -n tio4900-model && conda env create -f environment.yml`

#### 4. Slow Training

**Symptoms:** Hyperparameter tuning takes much longer than expected

**Solutions:**
- Reduce `n_trials` for initial testing (try 16 or 32)
- Use GPU if available
- Reduce `max_epochs` or increase `patience` in config
- Check that data preprocessing is not rerunning unnecessarily

#### 5. Missing Data Files

**Symptoms:** `FileNotFoundError` when loading data

**Solutions:**
- Ensure synthetic data generation completed successfully
- Check that parquet files exist in `src/data/{zone}/`
- Verify file paths in error message match expected locations

### Getting Help

If you encounter issues not covered here:

1. Check the log files in `results/logs/`
2. Review the [REPLICATION.md](REPLICATION.md) guide
3. Verify your environment matches [requirements.txt](requirements.txt)
4. Open an issue on the repository with:
   - Full error message and traceback
   - Command you ran
   - Python version and OS
   - Relevant log file excerpts

---

## Citation

If you use this code or methodology in your research, please cite:

```bibtex
@article{skjevrak2025probabilistic,
  title={Probabilistic Forecasting of Imbalance Prices: An Application to the Norwegian Electricity Market},
  author={Skjevrak, Knut and Holmen, Emil Duedahl and Westgaard, Sjur},
  journal={Manuscript in preparation for Applied Energy},
  year={2025},
  institution={Norwegian University of Science and Technology}
}
```

See [CITATION.cff](CITATION.cff) for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Knut Skjevrak**

---

## Acknowledgments

- Data provided by Volue AS
- Computational resources from NTNU HPC cluster
- Inspired by the Deep Distributional Neural Networks framework

---

**For detailed replication instructions, see [REPLICATION.md](REPLICATION.md)**
**For data access information, see [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)**
