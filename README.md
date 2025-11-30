# Replication Package: Probabilistic Forecasting of Imbalance Prices

This repository contains the replication code and data for the paper "Probabilistic Forecasting of Imbalance Prices: An Application to the Norwegian Electricity Market" submitted to the International Journal of Forecasting.

## Quick Start

### Installation

**Option 1: Conda (Recommended)**
```bash
conda env create -f environment.yml
conda activate tio4900-model
```

**Option 2: pip**
```bash
pip install -r requirements.txt
```

### Data

The original Norwegian electricity market data is confidential. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for access instructions.

For testing, generate synthetic data:
```bash
python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2024-04-26
```

### Model Training

Train a Deep Distributional Neural Network (DDNN):
```bash
python run_tuning.py --zone no1 --distribution jsu --run-id replication_001
```

Generate predictions:
```bash
python run_pred.py no1 jsu replication_001 --start-date 2024-04-26 --end-date 2024-04-26
```

## Models

Three probabilistic forecasting approaches are implemented:

1. **DDNN**: Deep Distributional Neural Networks with parametric distributions (Normal, JSU, Student's t)
2. **LQR**: Linear Quantile Regression for non-parametric quantile forecasting
3. **XGBoost**: Gradient boosting quantile regression

## Repository Structure

```
replication-PFoIP/
├── config/                     # Configuration files
├── src/
│   ├── data/                   # Data loading and preprocessing
│   ├── models/                 # Model architectures
│   ├── training/               # Training and prediction scripts
│   └── evaluation/             # Evaluation metrics
├── run_tuning.py               # Hyperparameter tuning
├── run_pred.py                 # Generate predictions
├── environment.yml             # Conda environment
└── requirements.txt            # pip requirements
```

## Documentation

- **[REPLICATION.md](REPLICATION.md)**: Complete step-by-step replication guide
- **[DATA_AVAILABILITY.md](DATA_AVAILABILITY.md)**: Data access and availability statement
- **[docs/example_output/](docs/example_output/)**: Expected output examples

## System Requirements

- **Software**: Python 3.10+, PyTorch 1.9+
- **Hardware**: 8GB RAM minimum (16GB+ recommended)
- **GPU**: Optional (CPU training supported)
- **Runtime**: ~30-45 minutes for minimal test, 5-7 days for full replication

## License

This project is licensed under the MIT License - see the LICENSE file for details.