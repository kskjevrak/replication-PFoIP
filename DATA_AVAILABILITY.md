# Data Availability Statement

## Original Data Source

The empirical analysis in "Probabilistic Forecasting of Imbalance Prices: An Application to the Norwegian Electricity Market" uses proprietary electricity market data from the Norwegian power system. This dataset includes:

- **mFRR (manual Frequency Restoration Reserve) prices and volumes**
  - Upward and downward regulation prices (hourly and 15-minute resolution)
  - Activated volumes for frequency balancing
  - Coverage: Norwegian bidding zones NO1-NO5
  - Period: August 25, 2019 - April 25, 2025

- **Spot market data**
  - Day-ahead electricity prices (Nord Pool)
  - Forecasted spot prices

- **Weather and system variables**
  - Temperature forecasts
  - Consumption forecasts
  - Hydropower, solar, and wind generation forecasts
  - Cross-border power flows

**Data Provider:** Volue AS (https://www.volue.com/)

## Data Access Restrictions

**The original data used in this study is confidential and cannot be publicly shared.**

The data is licensed from Volue AS under commercial terms that prohibit redistribution. This restriction applies to:

- Raw market data files
- Preprocessed datasets
- Any derivatives that could reconstruct the original time series

### Rationale for Restrictions

1. **Commercial sensitivity:** Market data has commercial value and is sold as a product by Volue AS
2. **Market participant confidentiality:** Some data elements may reveal strategic bidding information
3. **Licensing agreements:** Our research license explicitly prohibits public sharing

## Accessing the Original Data

Researchers wishing to replicate the study using the original data have two options:

### Option 1: Academic Data Request

Contact Volue AS to request academic access to the data:

- **Email:** academic-data@volue.com (or general contact via website)
- **Website:** https://www.volue.com/
- **Information to include in your request:**
  - Research affiliation and purpose
  - Reference to this paper
  - Specific data requirements (zones, time period, variables)
  - Commitment to their data usage terms

Academic licenses may be available at reduced cost or free for non-commercial research purposes.

### Option 2: Alternative Data Sources

Some components of the dataset are publicly available from alternative sources, though assembly and preprocessing would be required:

| Data Type | Alternative Source | URL |
|-----------|-------------------|-----|
| Day-ahead spot prices | Nord Pool | https://www.nordpoolgroup.com/ |
| mFRR activation volumes | Statnett (partial) | https://www.statnett.no/ |
| Weather forecasts | Norwegian Meteorological Institute | https://www.met.no/ |
| Consumption data | Statnett | https://www.statnett.no/en/market-and-operations/ |

**Note:** These alternative sources may not provide the exact same coverage, resolution, or time period as the Volue dataset. Results may differ from those reported in the paper.

## Synthetic Data for Reproduction

To enable code testing and methodological validation, we provide a **synthetic data generator** that creates artificial data matching the statistical properties of the original dataset.

### Generating Synthetic Data

```bash
# Generate data for all zones (2019-08-25 to 2025-04-25)
python src/data/synthetic_data.py --all-zones --start-date 2019-08-25 --end-date 2025-04-25

# Generate for a single zone
python src/data/synthetic_data.py --zone no1 --start-date 2019-08-25 --end-date 2025-04-25
```

### Synthetic Data Characteristics

The synthetic generator ([src/data/synthetic_data.py](src/data/synthetic_data.py)) creates time series with:

**Statistical Properties (based on Thesis Appendix B, Tables B1-B5):**
- **mFRR price premium:** Mean ~85-90 EUR/MWh, Std ~90-100 EUR/MWh
- **Heavy-tailed distribution:** Maximum values >1200 EUR/MWh to simulate price spikes
- **Skewness:** Right-skewed distributions typical of electricity prices
- **Autocorrelation:** Temporal persistence matching market behavior
- **Regime-switching:** Calm periods interspersed with extreme spike events

**Temporal Patterns:**
- Hourly resolution with Europe/Oslo timezone
- Daily cycles (higher prices during peak hours)
- Weekly patterns (lower prices on weekends)
- Seasonal variation (higher prices in winter)
- Autocorrelated price dynamics

**Variables Generated:**
- `premium`: Price premium (target variable)
- `mFRR_price_up`: Upward regulation prices
- `mFRR_price_down`: Downward regulation prices
- `mFRR_vol_up`: Upward activation volumes
- `mFRR_vol_down`: Downward activation volumes

### Limitations of Synthetic Data

**IMPORTANT:** Results obtained with synthetic data **will differ** from those reported in the paper. The synthetic generator is designed for:

1. **Code verification:** Testing that scripts run without errors
2. **Methodology demonstration:** Showing how the modeling pipeline works
3. **Educational purposes:** Learning the forecasting techniques

**The synthetic data cannot replicate:**
- **Actual market dynamics:** Real price formation mechanisms, strategic bidding behavior
- **Extreme events:** Specific historical crisis periods, supply shocks
- **Cross-variable dependencies:** Complex correlations between weather, demand, and prices
- **Non-stationarity:** Structural breaks and regime changes in the actual market
- **High-frequency patterns:** Sub-hourly price movements and intraday cycles

**Quantitative results** (forecast accuracy metrics, distribution parameters, optimal hyperparameters) obtained with synthetic data should **not be compared** to paper results or used for publication.

## Reproducibility Without Original Data

Despite data access restrictions, researchers can still validate our:

1. **Methodology:** The DDNN architecture, training procedure, and evaluation framework are fully documented and implemented in code
2. **Software implementation:** All model architectures and training logic can be inspected and tested
3. **Algorithmic choices:** Hyperparameter optimization strategies, distribution selection, loss functions
4. **Qualitative findings:** General insights about model behavior and performance patterns

Using synthetic data or alternative data sources following our preprocessing pipeline should yield qualitatively similar patterns, even if quantitative metrics differ.

## Preprocessed Data Specification

For researchers with data access, the original data undergoes the following preprocessing (implemented in [src/data/loader.py](src/data/loader.py)):

1. **Feature Engineering:**
   - 24-hour lagged values for all price and volume variables
   - Hour-of-day, day-of-week, month indicator variables
   - Rolling statistics (optional, configured in `default_config.yml`)

2. **Data Cleaning:**
   - Missing value handling (forward fill for short gaps, interpolation for longer gaps)
   - Outlier detection and winsorization (optional)
   - Timezone standardization (Europe/Oslo)

3. **Train/Validation/Test Split:**
   - Training: 2019-08-25 to 2023-04-24
   - Validation: 2023-04-25 to 2024-04-25
   - Test: 2024-04-26 to 2025-04-25
   - Time-series cross-validation during hyperparameter tuning

4. **Scaling:**
   - StandardScaler for features (mean=0, std=1)
   - MinMaxScaler for target variable (optional, configured per experiment)
   - Scalers fitted on training data only, saved for inference

The exact preprocessing configuration is documented in [config/default_config.yml](config/default_config.yml).

## Contact for Data Questions

For questions about:

- **Data access:** Contact Volue AS directly (see contact information above)
- **Data preprocessing and methodology:** Contact the corresponding author:
  - **Name:** Knut Skjevrak
  - **Email:** knut.skjevrak@ntnu.no
  - **Affiliation:** Department of Industrial Economics and Technology Management, NTNU

## Summary

| Aspect | Status | Access Method |
|--------|--------|---------------|
| **Original Volue data** | Confidential | Request from Volue AS |
| **Synthetic data** | Publicly available | Included in repository |
| **Alternative public data** | Partial coverage | Nord Pool, Statnett, met.no |
| **Code and methodology** | Fully open | This repository (MIT License) |
| **Preprocessing pipeline** | Fully documented | [src/data/loader.py](src/data/loader.py) |

---

**Last updated:** January 2025

This data availability statement complies with the International Journal of Forecasting's reproducibility guidelines and the FAIR (Findable, Accessible, Interoperable, Reusable) data principles to the extent possible given commercial data restrictions.
