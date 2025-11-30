import numpy as np
import pandas as pd
from scipy import stats
import os
import glob
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_crps(predictions, actual_values):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) using exact formula
    
    Parameters:
    predictions: array-like, shape (n_samples, 24) - the predicted values
    actual_values: array-like, shape (24,) - the observed/actual values
    
    Returns:
    list: Hourly CRPS scores
    """
    n_samples = predictions.shape[0]
    hourly_crps = []

    for hour in range(24):
        hour_predictions = predictions[:, hour]
        hour_actual = actual_values[hour]
        
        # Term 1: E[|X - x|] where X is forecast and x is observation
        term1 = np.mean(np.abs(hour_predictions - hour_actual))
        
        # Term 2: E[|X - X'|] where X and X' are independent forecast samples
        # Use efficient computation with sorted samples
        sorted_samples = np.sort(hour_predictions)
        indices = np.arange(1, n_samples + 1)
        weights = 2 * indices - n_samples - 1
        term2 = 2 * np.sum(weights * sorted_samples) / (n_samples ** 2)
        
        # CRPS = E[|X - x|] - 0.5 * E[|X - X'|]
        crps_hour = term1 - 0.5 * term2
        hourly_crps.append(crps_hour)
    
    return hourly_crps

def calculate_mae(predictions, actual_values):
    """
    Calculate Mean Absolute Error
    
    Parameters:
    predictions: array-like, shape (n_samples, 24) - the predicted values
    actual_values: array-like, shape (24,) - the observed/actual values
    
    Returns:
    float: Overall MAE score
    """
    # Calculate mean predictions for each hour
    mean_predictions = np.mean(predictions, axis=0)
    return mean_absolute_error(actual_values, mean_predictions)

def calculate_rmse(predictions, actual_values):
    """
    Calculate Root Mean Square Error
    
    Parameters:
    predictions: array-like, shape (n_samples, 24) - the predicted values
    actual_values: array-like, shape (24,) - the observed/actual values
    
    Returns:
    float: Overall RMSE score
    """
    # Calculate mean predictions for each hour
    mean_predictions = np.mean(predictions, axis=0)
    return np.sqrt(mean_squared_error(actual_values, mean_predictions))

def calculate_pinball_loss(predictions, actual_values, quantiles=[0.1, 0.5, 0.9]):
    """
    Calculate Pinball Loss for specified quantiles
    
    Parameters:
    predictions: array-like, shape (n_samples, 24) - the predicted values
    actual_values: array-like, shape (24,) - the observed/actual values
    quantiles: list - quantile levels to evaluate (default: [0.1, 0.5, 0.9])
    
    Returns:
    dict: Pinball losses for each quantile
    """
    results = {}
    
    for q in quantiles:
        q_pred = np.percentile(predictions, q * 100, axis=0)
        hourly_losses = []
        
        for h in range(24):
            if actual_values[h] >= q_pred[h]:
                # Actual value is greater than or equal to the quantile prediction
                loss = q * (actual_values[h] - q_pred[h])
            else:
                # Actual value is less than the quantile prediction
                loss = (1 - q) * (q_pred[h] - actual_values[h])
            hourly_losses.append(loss)
        
        results[f'q{int(q*100)}'] = np.mean(hourly_losses)
    
    # Add average across all quantiles
    results['mean'] = np.mean([v for v in results.values()])
    
    return results

def calculate_winkler_score(predictions, actual_values, alpha=0.9):
    """
    Calculate Winkler Score for prediction intervals
    
    Parameters:
    predictions: array-like, shape (n_samples, 24) - the predicted values
    actual_values: array-like, shape (24,) - the observed/actual values
    alpha: float - nominal coverage of the prediction interval (default: 0.9 for 90% interval)
    
    Returns:
    float: Average Winkler score across all hours
    """
    lower_q = (1 - alpha) / 2
    upper_q = 1 - lower_q
    
    lower_bound = np.percentile(predictions, lower_q * 100, axis=0)
    upper_bound = np.percentile(predictions, upper_q * 100, axis=0)
    
    interval_width = upper_bound - lower_bound
    
    hourly_scores = []
    for h in range(24):
        if lower_bound[h] <= actual_values[h] <= upper_bound[h]:
            # Actual value is inside the interval
            score = interval_width[h]
        else:
            # Actual value is outside the interval
            penalty = 0
            if actual_values[h] < lower_bound[h]:
                penalty = 2 / alpha * (lower_bound[h] - actual_values[h])
            else:  # actual_values[h] > upper_bound[h]
                penalty = 2 / alpha * (actual_values[h] - upper_bound[h])
            
            score = interval_width[h] + penalty
        
        hourly_scores.append(score)
    
    return np.mean(hourly_scores)

def load_samples(distribution, nr, zone='no1', date_str=None):
    """
    Generate prediction samples from distribution parameters in JSON files
    
    Parameters:
    distribution: str - Distribution type ('normal', 'jsu', 'skewt')
    nr: int or str - Model number or ensemble identifier
    zone: str - Price zone (default: 'no1')
    date_str: str - Optional specific date to load in 'YYYY-MM-DD' format
    
    Returns:
    dict: Mapping of dates to prediction sample arrays
    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    samples = {}
    print(root_dir)
    # Primary approach: Load from distribution parameters (JSON files)
    param_dir = os.path.join(root_dir, 'results', 'forecasts', f"distparams_{zone}_{distribution.lower()}_{nr}")
    
    if not os.path.exists(param_dir):
        print(f"Parameter directory not found: {param_dir}")
        return samples
    
    # Get list of parameter files
    if date_str:
        param_files = [os.path.join(param_dir, f"{date_str}.json")]
    else:
        param_files = glob.glob(os.path.join(param_dir, "*.json"))
        # Filter out state files or other non-parameter files
        param_files = [f for f in param_files if not f.endswith('_state.json')]
    
    print(f"Found {len(param_files)} parameter files in {param_dir}")
    
    # Process each parameter file
    for param_file in param_files:
        if not os.path.exists(param_file):
            continue
            
        date = os.path.basename(param_file).split('.')[0]
        #print(f"Processing parameters for date: {date}")
        
        try:
            # Load distribution parameters
            with open(param_file, 'r') as f:
                params = json.load(f)
            
            # Generate samples based on distribution type
            if distribution.lower() == 'normal':
                if 'loc' in params and 'scale' in params:
                    loc = np.array(params['loc'])
                    scale = np.array(params['scale'])
                    
                    # Generate normal samples
                    samples[date] = np.random.normal(
                        loc.reshape(1, -1),
                        scale.reshape(1, -1),
                        size=(10000, len(loc))
                    )
                    #print(f"Generated {samples[date].shape[0]} Normal samples for {date}")
                else:
                    print(f"Missing required parameters for Normal distribution in {param_file}")
                    
            elif distribution.lower() == 'jsu':
                if all(k in params for k in ['loc', 'scale', 'tailweight', 'skewness']):
                    loc = np.array(params['loc'])
                    scale = np.array(params['scale'])
                    tailweight = np.array(params['tailweight'])
                    skewness = np.array(params['skewness'])
                    
                    # Generate JSU samples
                    n_samples = 10000
                    n_hours = len(loc)
                    jsu_samples = np.zeros((n_samples, n_hours))
                    
                    for h in range(n_hours):
                        # Step 1: Generate standard normal samples
                        z = np.random.normal(0, 1, n_samples)
                        
                        # Step 2: Apply Johnson's SU transformation
                        # The formula is: x = loc + scale * sinh((z - skewness) / tailweight)
                        transformed = np.sinh((z - skewness[h]) / tailweight[h])
                        
                        # Step 3: Scale and shift
                        jsu_samples[:, h] = loc[h] + scale[h] * transformed
                    
                    samples[date] = jsu_samples
                    #print(f"Generated {samples[date].shape[0]} JSU samples for {date}")
                else:
                    print(f"Missing required parameters for JSU distribution in {param_file}")
                    
            elif distribution.lower() == 'skewt':
                if all(k in params for k in ['loc', 'scale', 'a', 'b']):
                    loc = np.array(params['loc'])
                    scale = np.array(params['scale'])
                    a = np.array(params['a'])  # Shape parameter
                    b = np.array(params['b'])  # Shape parameter
                    
                    # Generate skew-t samples
                    n_samples = 10000
                    n_hours = len(loc)
                    skewt_samples = np.zeros((n_samples, n_hours))
                    
                    for h in range(n_hours):
                        # Step 1: Generate beta random variables
                        u = np.random.beta(a[h], b[h], n_samples)
                        
                        # Step 2: Transform to skew-t
                        # This is a simplified approach
                        w = (1 - 2*u) * np.sqrt(a[h] * b[h] / (a[h] + b[h] + 1))
                        
                        # Step 3: Scale and shift
                        skewt_samples[:, h] = loc[h] + scale[h] * w
                    
                    samples[date] = skewt_samples
                    #print(f"Generated {samples[date].shape[0]} Skew-t samples for {date}")
                else:
                    print(f"Missing required parameters for Skew-t distribution in {param_file}")
                    
            elif distribution.lower() == 'xgboost':
                if 'quantiles' in params and 'values' in params:
                    quantiles = np.array(params['quantiles'])
                    values = np.array(params['values'])  # shape: (24, n_quantiles)
                    
                    # Generate samples via interpolation
                    n_samples = 10000
                    xgb_samples = np.zeros((n_samples, 24))
                    for hour in range(24):
                        random_quantiles = np.random.uniform(0, 1, n_samples)
                        xgb_samples[:, hour] = np.interp(random_quantiles, quantiles, values[hour])
                    
                    samples[date] = xgb_samples
                else:
                    print(f"Missing required parameters for XGBoost distribution in {param_file}")

            elif distribution.lower() == 'lqa':
                if 'quantiles' in params and 'values' in params:
                    quantiles = np.array(params['quantiles'])
                    values = np.array(params['values'])  # shape: (24, n_quantiles)
                    
                    # Generate samples via interpolation (same as XGBoost approach)
                    n_samples = 10000
                    linear_quantile_samples = np.zeros((n_samples, 24))
                    for hour in range(24):
                        random_quantiles = np.random.uniform(0, 1, n_samples)
                        linear_quantile_samples[:, hour] = np.interp(random_quantiles, quantiles, values[hour])
                    
                    samples[date] = linear_quantile_samples
                    #print(f"Generated {samples[date].shape[0]} Linear Quantile samples for {date}")
                else:
                    print(f"Missing required parameters for Linear Quantile distribution in {param_file}")

            
            
            else:
                print(f"Unsupported distribution type: {distribution}")
                
        except Exception as e:
            print(f"Error processing parameters from {param_file}: {str(e)}")
            import traceback
            traceback.print_exc()
    
       
    # Fallback 1: If still no samples, try to reconstruct from forecast summary statistics
    if not samples:
        print("No raw samples found, trying to reconstruct from forecast summary...")
        forecast_dir = os.path.join(root_dir, 'results', 'forecasts', f"df_forecasts_{zone}_{distribution.lower()}_{nr}")
        
        if os.path.exists(forecast_dir):
            if date_str:
                forecast_files = [os.path.join(forecast_dir, f"{date_str}.csv")]
            else:
                forecast_files = glob.glob(os.path.join(forecast_dir, "*.csv"))
            
            for forecast_file in forecast_files:
                if not os.path.exists(forecast_file):
                    continue
                    
                date = os.path.basename(forecast_file).split('.')[0]
                
                try:
                    # Load forecast summary
                    df = pd.read_csv(forecast_file, index_col=0)
                    
                    # Generate samples from statistics
                    if 'mean' in df.columns and 'std' in df.columns:
                        mean = df['mean'].values
                        std = df['std'].values
                        samples[date] = np.random.normal(
                            mean.reshape(1, -1),
                            std.reshape(1, -1),
                            size=(10000, len(mean))
                        )
                        print(f"Reconstructed samples from summary statistics in {forecast_file}")
                except Exception as e:
                    print(f"Error reconstructing samples from {forecast_file}: {str(e)}")
    
    print(f"Total dates with generated samples: {len(samples)}")
    return samples

def load_actual_values(zone='no1', date_range=None, target_column='premium'):
    """Revised version to handle DST transitions properly"""
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try common data file paths
    path = os.path.join(root_dir, 'src', 'data', zone, f"merged_dataset_{zone}.parquet")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file for zone {zone}")
    
    # Load data - this already has the correct DST handling from the data pull
    data = pd.read_parquet(path)
    
    # Ensure datetime index with timezone
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert('Europe/Oslo')
    elif data.index.tz.zone != 'Europe/Oslo':
        data.index = data.index.tz_convert('Europe/Oslo')
    
    # Check if target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data. Available columns: {data.columns.tolist()}")
    
    # Filter by date range if provided
    if date_range:
        start_date, end_date = date_range
        start_date = pd.Timestamp(start_date, tz='Europe/Oslo')
        end_date = pd.Timestamp(end_date, tz='Europe/Oslo')
        data = data[(data.index >= start_date) & (data.index <= end_date + pd.Timedelta(days=1))]
    
    # Group by date and extract daily values, preserving DST transitions
    actual_values = {}
    for date_str, group in data.groupby(data.index.strftime('%Y-%m-%d')):
        actual_values[date_str] = group[target_column].values
    
    return actual_values

def calculate_all_metrics(distribution, nr, zone='no1', date_range=None, target_column='premium'):
    """
    Calculate all metrics for available dates
    
    Parameters:
    distribution: str - Distribution type ('normal', 'jsu', 'skewt')
    nr: int or str - Model number or ensemble identifier
    zone: str - Price zone (default: 'no1')
    date_range: tuple - Optional (start_date, end_date) in 'YYYY-MM-DD' format
    target_column: str - Column name for the target variable
    
    Returns:
    dict: Dictionary containing overall metrics and detailed results
    """
    # Load predictions and actual values
    samples = load_samples(distribution, nr, zone)
    
    if not samples:
        print(f"No prediction samples found for {zone}_{distribution}_{nr}")
        return None
    
    try:
        actual_values = load_actual_values(zone, date_range, target_column)
    except Exception as e:
        print(f"Error loading actual values: {e}")
        return None
    
    if not actual_values:
        print("No actual values found")
        return None
    
    # Find common dates for evaluation
    common_dates = sorted(set(samples.keys()) & set(actual_values.keys()))
    samples_dates = set(samples.keys())
    actuals_dates = set(actual_values.keys())
    missing_in_samples = actuals_dates - samples_dates
    missing_in_actuals = samples_dates - actuals_dates

    # Remove the day that is missing
    if missing_in_samples:
        common_dates = [date for date in common_dates if date not in missing_in_samples]
    if missing_in_actuals:
        common_dates = [date for date in common_dates if date not in missing_in_actuals]
    #if missing_in_samples:
    #    print(f"Dates in actuals but not in samples: {missing_in_samples}")
    #if missing_in_actuals:
    #    print(f"Dates in samples but not in actuals: {missing_in_actuals}")

    if not common_dates:
        print(f"No common dates found between predictions and actual values")
        return None
    
    print(f"Evaluating {len(common_dates)} days")
    
    # Calculate metrics for each date
    all_results = {
        'crps': [],
        'mae': [],
        'rmse': [],
        'pinball': [],
        'winkler': [],
        'dates': []
    }
    
    for date in common_dates:
        predictions = samples[date]
        actuals = actual_values[date]
        
        # Handle DST transitions
        if predictions.shape[1] != len(actuals):
            if len(actuals) == 25 and predictions.shape[1] == 24:
                # DST fall transition - drop the 4th hour (index 3, usually 1-2 AM duplicate)
                actuals = np.concatenate([actuals[:3], actuals[4:]])
                print(f"DST fall transition for {date}: dropped duplicate hour (index 3)")
            elif len(actuals) == 23 and predictions.shape[1] == 24:
                # DST spring transition - duplicate 3rd hour (index 2) to fill missing 4th hour
                actuals = np.concatenate([actuals[:3], [actuals[2]], actuals[3:]])
                print(f"DST spring transition for {date}: duplicated hour 2 to fill missing hour 3")
            else:
                # Other shape mismatches
                print(f"Shape mismatch for {date}: predictions {predictions.shape}, actuals {len(actuals)}")
                continue
        
        # Calculate metrics
        try:
            crps_result = calculate_crps(predictions, actuals)
            mae_result = calculate_mae(predictions, actuals)
            rmse_result = calculate_rmse(predictions, actuals)
            pinball_result = calculate_pinball_loss(predictions, actuals)
            winkler_result = calculate_winkler_score(predictions, actuals)
            
            all_results['crps'].append(crps_result)
            all_results['mae'].append(mae_result)
            all_results['rmse'].append(rmse_result)
            all_results['pinball'].append(pinball_result)
            all_results['winkler'].append(winkler_result)
            all_results['dates'].append(date)
        except Exception as e:
            print(f"Error calculating metrics for {date}: {e}")
    
    # Calculate overall statistics
    overall_stats = {
        'total_days': len(all_results['dates']),
        'metrics': {
            'crps': {
                'mean': np.mean([np.mean(crps) for crps in all_results['crps']]),
                'median': np.median([np.mean(crps) for crps in all_results['crps']]),
                'std': np.std([np.mean(crps) for crps in all_results['crps']])
            },
            'mae': {
                'mean': np.mean(all_results['mae']),
                'median': np.median(all_results['mae']),
                'std': np.std(all_results['mae'])
            },
            'rmse': {
                'mean': np.mean(all_results['rmse']),
                'median': np.median(all_results['rmse']),
                'std': np.std(all_results['rmse'])
            },
            'pinball': {
                'q10': np.mean([res['q10'] for res in all_results['pinball']]),
                'q50': np.mean([res['q50'] for res in all_results['pinball']]),
                'q90': np.mean([res['q90'] for res in all_results['pinball']]),
                'mean': np.mean([res['mean'] for res in all_results['pinball']])
            },
            'winkler': {
                'mean': np.mean(all_results['winkler']),
                'median': np.median(all_results['winkler']),
                'std': np.std(all_results['winkler'])
            }
        },
        'dates': all_results['dates'],
        'daily_results': {
            date: {
                'crps': np.mean(all_results['crps'][i]),
                'mae': all_results['mae'][i],
                'rmse': all_results['rmse'][i],
                'pinball': all_results['pinball'][i],
                'winkler': all_results['winkler'][i]
            } for i, date in enumerate(all_results['dates'])
        }
    }
    
    return overall_stats

def run_metrics_comparison(zones=None, distributions=None, model_numbers=None, date_range=None):
    """
    Run metrics comparison across different zones, distributions, and model numbers
    and generate LaTeX tables
    
    Parameters:
    zones: list of str - Price zones to evaluate
    distributions: list of str - Distribution types to evaluate
    model_numbers: list of str/int - Model numbers and ensemble types to evaluate
    date_range: tuple - (start_date, end_date) in 'YYYY-MM-DD' format
    
    Returns:
    tuple: (main_table_latex, appendix_tables_latex)
    """
    if zones is None:
        zones = ['no1', 'no2', 'no3', 'no4', 'no5']
    
    if distributions is None:
        distributions = ['normal', 'jsu', 'skewt']
    
    if model_numbers is None:
        model_numbers = [1, 2, 3, 4, 'vEns', 'hEns']
    
    # Prepare results dictionary
    all_results = {}
    
    # Run metrics calculation for each combination
    for zone in zones:
        zone_results = {}
        for distribution in distributions:
            dist_results = {}
            for nr in model_numbers:
                print(f"Evaluating {zone} - {distribution} - {nr}...")
                
                try:
                    results = calculate_all_metrics(
                        distribution=distribution,
                        nr=str(nr),
                        zone=zone,
                        date_range=date_range,
                        target_column='premium'
                    )
                    
                    if results:
                        dist_results[str(nr)] = results
                except Exception as e:
                    print(f"Error evaluating {zone} - {distribution} - {nr}: {str(e)}")
            
            if dist_results:
                zone_results[distribution] = dist_results
        
        if zone_results:
            all_results[zone] = zone_results
    
    # Generate main table for NO1 zone
    main_table = generate_main_latex_table(all_results.get('no1', {}))
    
    # Generate appendix tables for all zones
    appendix_tables = {}
    for zone in all_results:
        appendix_tables[zone] = generate_zone_latex_table(zone, all_results[zone])
    
    return main_table, appendix_tables

def generate_main_latex_table(zone_results):
    """Generate LaTeX table for the main paper (one zone) with new naming scheme"""
    
    naive = {
        'no1': '\\quad Na\\"{i}ve & 11.674 & 19.152 & 10.006 & 4.279 & 5.837 & 4.064 & 61.193 \\\\\n',
        'no2': '\\quad Na\\"{i}ve & 14.397 & 22.482 & 11.965 & 4.885 & 7.199 & 4.655 & 68.462 \\\\\n',
        'no3': '\\quad Na\\"{i}ve & 10.921 & 16.039 & 8.936 & 3.457 & 5.461 & 3.472 & 55.422 \\\\\n',
        'no4': '\\quad Na\\"{i}ve & 13.849 & 19.286 & 11.573 & 4.563 & 6.924 & 4.802 & 63.996 \\\\\n',
        'no5': '\\quad Na\\"{i}ve & 12.330 & 20.789 & 10.160 & 4.270 & 6.165 & 3.846 & 55.820 \\\\\n'
    }
    
    
    exp_smooth = {
        'no1': "\\quad Exponential Smoothing & 8.289 & 13.646 & 6.960 & 3.155 & 4.144 & 2.661 & 42.977 \\\\\n",
        'no2': "\\quad Exponential Smoothing & 10.566 & 16.361 & 8.574 & 3.894 & 5.283 & 2.891 & 49.877 \\\\\n",
        'no3': "\\quad Exponential Smoothing & 7.974 & 11.703 & 6.497 & 2.343 & 3.987 & 2.816 & 41.237 \\\\\n",
        'no4': "\\quad Exponential Smoothing & 10.216 & 14.889 & 8.517 & 3.304 & 5.108 & 3.721 & 46.929 \\\\\n",
        'no5': "\\quad Exponential Smoothing & 8.755 & 14.149 & 7.091 & 2.684 & 4.378 & 2.951 & 41.192 \\\\\n"
    }
    
    if not zone_results:
        return "No results available for main table."
    
    # Start LaTeX table
    latex = "\\begin{table*}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Comparison of forecasting accuracy for different models (NO1 zone)}\n"
    latex += "\\label{tab:forecast_accuracy_main}\n"
    latex += "\\begin{tabular*}{\\textwidth}{@{\extracolsep{\\fill}}lccccccc@{}}\n"
    latex += "\\toprule\n"
    latex += "Model & MAE & RMSE & CRPS & Pinball 10\\% & Pinball 50\\% & Pinball 90\\% & Winkler \\\\\n"
    latex += "\\midrule\n"
    
    # Add benchmark placeholders
    latex += naive['no1']
    latex += exp_smooth['no1']
    
    # Define model order and grouping
    model_order = ['lqa', 'xgboost', 'normal', 'jsu', 'skewt',]
    model_names = {
        'xgboost': 'XGB-QRA',
        'normal': 'DDNN-N',
        'jsu': 'DDNN-JSU',
        'skewt': 'DDNN-S-T',
        'lqa': 'Linear Quantile Regression'
    }
    
    # Add rows for each distribution group
    for distribution in model_order:
        if distribution in zone_results:
            # Sort model numbers: 1,2,3,4,hEns,vEns
            model_numbers = sorted(zone_results[distribution].keys(), 
                                 key=lambda x: (x not in ['1','2','3','4'], 
                                              int(x) if x.isdigit() else float('inf'), 
                                              x))
            
            for nr in model_numbers:
                results = zone_results[distribution][nr]
                
                # Create model name
                base_name = model_names[distribution]
                if distribution.lower() == "lqa":
                    model_name = f"\\quad {base_name}"
                else:
                    model_name = f"\\quad {base_name}-{nr}"
                
                # Format metrics
                mae = f"{results['metrics']['mae']['mean']:.3f}"
                rmse = f"{results['metrics']['rmse']['mean']:.3f}"
                crps = f"{results['metrics']['crps']['mean']:.3f}"
                pinball_10 = f"{results['metrics']['pinball']['q10']:.3f}"
                pinball_50 = f"{results['metrics']['pinball']['q50']:.3f}"
                pinball_90 = f"{results['metrics']['pinball']['q90']:.3f}"
                winkler = f"{results['metrics']['winkler']['mean']:.3f}"
                # Add row
                latex += f"{model_name} & {mae} & {rmse} & {crps} & {pinball_10} & {pinball_50} & {pinball_90} & {winkler} \\\\\n"
            
            # Add space after each model group
            latex += "\\addlinespace\n"
    
    # Finish table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular*}\n"
    latex += "\\end{table*}"
    
    return latex

def generate_zone_latex_table(zone, zone_results):
    """Generate LaTeX table for a specific zone (for appendix) with new naming scheme"""
    if not zone_results:
        return f"No results available for {zone}."
    
    naive = {
        'no1': '\\quad Na\\"{i}ve & 11.674 & 19.152 & 10.006 & 4.279 & 5.837 & 4.064 & 61.193 \\\\\n',
        'no2': '\\quad Na\\"{i}ve & 14.397 & 22.482 & 11.965 & 4.885 & 7.199 & 4.655 & 68.462 \\\\\n',
        'no3': '\\quad Na\\"{i}ve & 10.921 & 16.039 & 8.936 & 3.457 & 5.461 & 3.472 & 55.422 \\\\\n',
        'no4': '\\quad Na\\"{i}ve & 13.849 & 19.286 & 11.573 & 4.563 & 6.924 & 4.802 & 63.996 \\\\\n',
        'no5': '\\quad Na\\"{i}ve & 12.330 & 20.789 & 10.160 & 4.270 & 6.165 & 3.846 & 55.820 \\\\\n'
    }
    
    
    exp_smooth = {
        'no1': "\\quad Exponential Smoothing & 8.289 & 13.646 & 6.960 & 3.155 & 4.144 & 2.661 & 42.977 \\\\\n",
        'no2': "\\quad Exponential Smoothing & 10.566 & 16.361 & 8.574 & 3.894 & 5.283 & 2.891 & 49.877 \\\\\n",
        'no3': "\\quad Exponential Smoothing & 7.974 & 11.703 & 6.497 & 2.343 & 3.987 & 2.816 & 41.237 \\\\\n",
        'no4': "\\quad Exponential Smoothing & 10.216 & 14.889 & 8.517 & 3.304 & 5.108 & 3.721 & 46.929 \\\\\n",
        'no5': "\\quad Exponential Smoothing & 8.755 & 14.149 & 7.091 & 2.684 & 4.378 & 2.951 & 41.192 \\\\\n"
    }

    # Start LaTeX table
    latex = "\\begin{table*}[htbp]\n"
    latex += "\\centering\n"
    latex += f"\\caption{{Detailed forecasting accuracy for {zone.upper()} zone}}\n"
    latex += f"\\label{{tab:forecast_accuracy_{zone}}}\n"
    latex += "\\begin{tabular*}{\\textwidth}{@{\extracolsep{\\fill}}lccccccc@{}}\n"
    latex += "\\toprule\n"
    latex += "Model & MAE & RMSE & CRPS & Pinball 10\\% & Pinball 50\\% & Pinball 90\\% & Winkler \\\\\n"
    latex += "\\midrule\n"
    
    # Add benchmark placeholders
    latex += naive[zone]
    latex += exp_smooth[zone]    
    # Define model order and grouping
    # Define model order and grouping
    model_order = ['lqa', 'xgboost', 'normal', 'jsu', 'skewt',]
    model_names = {
        'xgboost': 'XGB-QRA',
        'normal': 'DDNN-N',
        'jsu': 'DDNN-JSU',
        'skewt': 'DDNN-S-T',
        'lqa': 'Linear Quantile Regression'
    }
    
    # Add rows for each distribution group
    for distribution in model_order:
        if distribution in zone_results:
            # Sort model numbers: 1,2,3,4,hEns,vEns
            model_numbers = sorted(zone_results[distribution].keys(), 
                                 key=lambda x: (x not in ['1','2','3','4'], 
                                              int(x) if x.isdigit() else float('inf'), 
                                              x))
            
            for nr in model_numbers:
                results = zone_results[distribution][nr]
                
                # Create model name
                base_name = model_names[distribution]
                if distribution.lower() == "lqa":
                    model_name = f"\\quad {base_name}"
                else:
                    model_name = f"\\quad {base_name}-{nr}"
                    
                # Format metrics
                mae = f"{results['metrics']['mae']['mean']:.3f}"
                rmse = f"{results['metrics']['rmse']['mean']:.3f}"
                crps = f"{results['metrics']['crps']['mean']:.3f}"
                pinball_10 = f"{results['metrics']['pinball']['q10']:.3f}"
                pinball_50 = f"{results['metrics']['pinball']['q50']:.3f}"
                pinball_90 = f"{results['metrics']['pinball']['q90']:.3f}"
                winkler = f"{results['metrics']['winkler']['mean']:.3f}"
                # Add row
                latex += f"{model_name} & {mae} & {rmse} & {crps} & {pinball_10} & {pinball_50} & {pinball_90} & {winkler} \\\\\n"
            
            # Add space after each model group
            latex += "\\addlinespace\n"
    
    # Finish table
    latex += "\\bottomrule\n"
    latex += "\\end{tabular*}\n"
    latex += "\\end{table*}"
    
    return latex

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate comprehensive metric comparison tables')
    parser.add_argument('--zones', nargs='+', default=['no1', 'no2', 'no3', 'no4', 'no5'], help='Price zones to evaluate')
    parser.add_argument('--distributions', nargs='+', default=['lqa'], help='Distribution types to evaluate')
    parser.add_argument('--numbers', nargs='+', default=['1', '2', '3', '4', 'vEns', 'hEns'], help='Model numbers and ensemble types')
    parser.add_argument('--start', type=str, default='2024-04-26', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-04-25', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='forecast_tables_lqa.tex', help='Output LaTeX file')
    
    args = parser.parse_args()
    
    date_range = None
    if args.start and args.end:
        date_range = (args.start, args.end)
    
    main_table, appendix_tables = run_metrics_comparison(
        zones=args.zones,
        distributions=args.distributions,
        model_numbers=args.numbers,
        date_range=date_range
    )
    
    # Write output
    with open(args.output, 'w') as f:
        f.write("% Main table for paper body\n")
        f.write(main_table)
        f.write("\n\n")
        f.write("% Appendix tables\n")
        for zone, table in appendix_tables.items():
            f.write(f"% Table for {zone}\n")
            f.write(table)
            f.write("\n\n")
    
    print(f"Tables written to {args.output}")