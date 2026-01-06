# Example Outputs

This directory contains small example output files to demonstrate what users should expect after running the replication package.

## Files

### 1. example_best_params.yaml

Example hyperparameter configuration file saved after Optuna optimization completes.

**Location after tuning:** `results/models/{zone}_{distribution}_{run_id}/best_params.yaml`

### 2. example_forecast.csv

Example probabilistic forecast output for a single day (24 hours).

**Location after prediction:** `results/forecasts/df_forecasts_{zone}_{distribution}_{run_id}/{date}_forecast.csv`

### 3. example_tuning_log.txt

Excerpt from a hyperparameter tuning log showing Optuna trial progression.

**Location during tuning:** `results/logs/{zone}_{distribution}_{run_id}.log`

## Notes

- These are **illustrative examples** with synthetic/dummy data
- Actual output values will differ based on your data and random seed
- File formats and column names should match these examples
- Use these to verify your pipeline is producing correctly formatted outputs
