import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import glob
import json
from scipy import stats
import traceback

def fit_distribution_parameters(samples, distribution_type):
    """
    Fit distribution parameters to the given samples
    
    Parameters:
    -----------
    samples : numpy.ndarray
        Array of shape (n_samples, 24) containing the ensemble predictions
    distribution_type : str
        Type of distribution to fit ('normal', 'jsu', etc.)
        
    Returns:
    --------
    dict: Dictionary of fitted parameters
    """
    n_hours = samples.shape[1]
    params = {}
    
    if distribution_type.lower() == 'normal':
        # Fit normal distribution (mean and standard deviation)
        params['loc'] = np.mean(samples, axis=0).tolist()
        params['scale'] = np.std(samples, axis=0).tolist()
        
    elif distribution_type.lower() == 'jsu':
        # Fit Johnson's SU distribution (location, scale, tailweight, skewness)
        # We'll approximate the JSU parameters using method of moments and skewness/kurtosis
        params['loc'] = np.zeros(n_hours)
        params['scale'] = np.zeros(n_hours)
        params['tailweight'] = np.zeros(n_hours)
        params['skewness'] = np.zeros(n_hours)
        
        for hour in range(n_hours):
            hour_samples = samples[:, hour]
            
            # Calculate moment-based statistics
            mean = np.mean(hour_samples)
            std = np.std(hour_samples)
            skewness = stats.skew(hour_samples)
            kurtosis = stats.kurtosis(hour_samples)
            
            # Approximate JSU parameters
            # This is a simplified approach - in practice, more sophisticated
            # parameter estimation would be used
            params['loc'][hour] = mean
            params['scale'][hour] = std
            
            # Tailweight is related to kurtosis
            # Higher kurtosis = heavier tails = higher tailweight
            # We'll approximate with a reasonable range of values
            params['tailweight'][hour] = 1.0 + np.clip(kurtosis / 2, 0, 10)
            
            # Skewness parameter directly influences distribution skewness
            params['skewness'][hour] = np.clip(skewness, -10, 10)
        
        # Convert to lists for JSON serialization
        params['loc'] = params['loc'].tolist()
        params['scale'] = params['scale'].tolist()
        params['tailweight'] = params['tailweight'].tolist()
        params['skewness'] = params['skewness'].tolist()
        
    elif distribution_type.lower() == 'skewt':
        # Fit Skew-t distribution (location, scale, df, alpha)
        params['loc'] = np.zeros(n_hours)
        params['scale'] = np.zeros(n_hours)
        params['a'] = np.zeros(n_hours)
        params['b'] = np.zeros(n_hours)
        
        for hour in range(n_hours):
            hour_samples = samples[:, hour]
            
            # Calculate moment-based statistics
            mean = np.mean(hour_samples)
            std = np.std(hour_samples)
            skewness = stats.skew(hour_samples)
            
            # Approximate skew-t parameters
            params['loc'][hour] = mean
            params['scale'][hour] = std
            
            # Default values for shape parameters
            params['a'][hour] = 3.0
            params['b'][hour] = 3.0
            
            # Adjust for skewness
            if skewness > 0:
                params['a'][hour] = 3.0 + np.clip(skewness, 0, 5)
            elif skewness < 0:
                params['b'][hour] = 3.0 + np.clip(-skewness, 0, 5)
        
        # Convert to lists for JSON serialization
        params['loc'] = params['loc'].tolist()
        params['scale'] = params['scale'].tolist()
        params['a'] = params['a'].tolist()
        params['b'] = params['b'].tolist()

    elif distribution_type.lower() == 'xgboost':
        # For XGBoost, fit quantiles instead of distribution parameters
        # Use the same quantile levels as individual XGBoost models for consistency
        quantile_levels = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        
        # Calculate quantiles for each hour
        quantile_values = []
        for hour in range(n_hours):
            hour_samples = samples[:, hour]
            hour_quantiles = np.percentile(hour_samples, np.array(quantile_levels) * 100)
            quantile_values.append(hour_quantiles.tolist())
        
        # Return in XGBoost-compatible format
        params = {
            'quantiles': quantile_levels,
            'values': quantile_values,  # shape: (24, n_quantiles)
            'model_type': 'xgboost'
        }

    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
        
    return params

def calculate_forecast_statistics(samples):
    """
    Calculate summary statistics for the forecast
    
    Parameters:
    -----------
    samples : numpy.ndarray
        Array of shape (n_samples, 24) containing the ensemble predictions
        
    Returns:
    --------
    dict: Dictionary of statistics for each hour
    """
    n_hours = samples.shape[1]
    
    # Determine state based on predictions
    state_info = {}
    threshold = 0.5
    
    up_prob = np.mean(samples > threshold, axis=0)
    down_prob = np.mean(samples < -threshold, axis=0)
    normal_prob = 1.0 - up_prob - down_prob
    
    states = np.zeros(n_hours, dtype=int)
    states[up_prob > down_prob] = 1  # Up regulation
    states[down_prob > up_prob] = 2  # Down regulation
    
    state_names = ['normal', 'up_regulation', 'down_regulation']
    state_list = [state_names[s] for s in states]
    
    # Calculate percentiles and statistics
    stats_dict = {
        'state': state_list,
        'normal_prob': normal_prob.tolist(),
        'up_prob': up_prob.tolist(),
        'down_prob': down_prob.tolist(),
        'mean': np.mean(samples, axis=0).tolist(),
        'median': np.median(samples, axis=0).tolist(),
        'std': np.std(samples, axis=0).tolist(),
        'p10': np.percentile(samples, 10, axis=0).tolist(),
        'p25': np.percentile(samples, 25, axis=0).tolist(),
        'p75': np.percentile(samples, 75, axis=0).tolist(),
        'p90': np.percentile(samples, 90, axis=0).tolist()
    }
    
    return stats_dict

def combine_predictions(distribution, zone, nrs=4, method='horizontal'):
    """
    Combine predictions from multiple models using distribution parameters
    
    Parameters:
    -----------
    distribution : str
        Probability distribution type (e.g., 'normal', 'jsu')
    zone : str
        Zone identifier (e.g., 'no1', 'no2')
    nrs : int or list
        Number of models to combine or list of specific model numbers
    method : str
        Ensemble method ('horizontal' or 'vertical')
    """
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'forecasts')
    base_path = Path(root_dir)
    
    # Convert nrs to list if it's an integer
    if isinstance(nrs, int):
        nrs = list(range(1, nrs + 1))
    
    # Determine patterns for input forecasts, output parameters, and output dataframes
    forecasts_dir = base_path
    
    # Ensure required directories exist
    if not forecasts_dir.exists():
        raise ValueError(f"Forecasts directory not found: {forecasts_dir}")
    
    # Define input patterns based on standard structure
    param_pattern = f"distparams_{zone}_{distribution.lower()}_"
    df_pattern = f"df_forecasts_{zone}_{distribution.lower()}_"
    
    # Verify at least one input directory exists
    param_dirs = [forecasts_dir / f"{param_pattern}{nr}" for nr in nrs]
    existing_dirs = [d for d in param_dirs if d.exists()]
    
    if not existing_dirs:
        raise ValueError(f"No input directories found matching pattern {param_pattern}{{nr}}")
    
    # Set up output directories
    ensemble_suffix = "hEns" if method == 'horizontal' else "vEns"
    
    param_dir = forecasts_dir / f"{param_pattern}{ensemble_suffix}"
    df_dir = forecasts_dir / f"{df_pattern}{ensemble_suffix}"
    
    # Create output directories
    os.makedirs(param_dir, exist_ok=True)
    os.makedirs(df_dir, exist_ok=True)
    
    # Find common dates across all input directories
    date_sets = []
    for input_dir in existing_dirs:
        json_files = list(input_dir.glob("*.json"))
        dates = set(f.stem for f in json_files)
        date_sets.append(dates)
    
    # Find intersection of all date sets
    common_dates = set.intersection(*date_sets) if date_sets else set()
    
    if not common_dates:
        print(f"No common dates found across all input directories")
        return
    
    # Sort dates for processing
    dates = sorted(list(common_dates))
    print(f"Found {len(dates)} common dates from {dates[0]} to {dates[-1]}")
    
    # Process each date
    processed_dates = 0
    
    for date in dates:
        try:
            # For each day
            daily_samples = []
            valid_models = 0
            
            for input_dir in existing_dirs:
                json_path = input_dir / f"{date}.json"
                
                if json_path.exists():
                    try:
                        # Load distribution parameters from JSON
                        with open(json_path, 'r') as f:
                            params = json.load(f)
                        
                        # Generate samples based on distribution type
                        samples = generate_samples_from_params(params, distribution, n_samples=10000)
                        daily_samples.append(samples)
                        valid_models += 1
                    except Exception as e:
                        print(f"Error loading or processing {json_path}: {e}")
            
            if valid_models < 2:
                print(f"Not enough valid models for {date}, need at least 2, found {valid_models}")
                continue
            
            # Process the samples
            try:
                # Convert to numpy array for processing
                daily_samples = np.array(daily_samples)
                
                if method == 'horizontal':
                    # Define quantile levels
                    quantile_levels = np.linspace(0, 1, 100)
                    
                    # Calculate quantiles for each model and hour
                    model_quantiles = np.zeros((len(daily_samples), 100, 24))
                    for model_idx, model_samples in enumerate(daily_samples):
                        for hour in range(24):
                            model_quantiles[model_idx, :, hour] = np.quantile(
                                model_samples[:, hour], quantile_levels
                            )
                    
                    # Average quantiles across models
                    avg_quantiles = np.mean(model_quantiles, axis=0)
                    
                    # Generate new samples using interpolation
                    new_samples = np.zeros((10000, 24))
                    for hour in range(24):
                        # Generate uniform random numbers
                        u = np.random.uniform(0, 1, 10000)
                        # Interpolate from averaged quantiles
                        new_samples[:, hour] = np.interp(u, quantile_levels, avg_quantiles[:, hour])
                    
                else:  # vertical
                    # Concatenate all samples and randomly resample
                    combined = np.concatenate(daily_samples, axis=0)
                    indices = np.random.choice(combined.shape[0], size=10000)
                    new_samples = combined[indices]
                
                # Fit distribution parameters
                params = fit_distribution_parameters(new_samples, distribution)
                
                # Calculate forecast statistics
                stats = calculate_forecast_statistics(new_samples)
                
                # Save distribution parameters to JSON
                param_path = param_dir / f"{date}.json"
                with open(param_path, 'w') as f:
                    json.dump(params, f)
                
                # Save forecast statistics to CSV
                # Create a dataframe with hourly index
                date_timestamp = pd.Timestamp(date)
                hours = pd.date_range(start=date_timestamp, periods=24, freq='H')
                df = pd.DataFrame(index=hours)
                
                # Add all statistics
                for stat_name, stat_values in stats.items():
                    if isinstance(stat_values, list) and len(stat_values) == 24:
                        df[stat_name] = stat_values
                    else:
                        # Handle non-list or incorrect length
                        print(f"Warning: Skipping invalid statistic {stat_name} for {date}")
                
                # Save to CSV
                df.to_csv(df_dir / f"{date}.csv")
                
                processed_dates += 1
                if processed_dates % 50 == 0:
                    print(f"Processed {processed_dates} dates...")
                
            except Exception as e:
                print(f"Error processing samples for {date}: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error processing date {date}: {e}")
    
    ensemble_type = "Horizontal" if method == 'horizontal' else "Vertical"
    print(f"{ensemble_type} ensemble parameters saved to {param_dir}")
    print(f"{ensemble_type} ensemble statistics saved to {df_dir}")
    print(f"Successfully processed {processed_dates} out of {len(dates)} dates")

def generate_samples_from_params(params, distribution_type, n_samples=10000):
    """
    Generate samples from distribution parameters
    
    Parameters:
    -----------
    params : dict
        Dictionary of distribution parameters
    distribution_type : str
        Type of distribution ('normal', 'jsu', 'skewt', 'xgboost')
    n_samples : int
        Number of samples to generate
        
    Returns:
    --------
    numpy.ndarray: Array of shape (n_samples, 24) containing generated samples
    """

    n_hours = len(params.get('values', [])) if distribution_type == 'xgboost' else len(params.get('loc', [])) 
    if n_hours == 0:
        raise ValueError("Invalid parameters: 'loc' parameter missing or empty")
    
    samples = np.zeros((n_samples, n_hours))
    
    if distribution_type.lower() == 'normal':
        # Generate samples from normal distribution
        for hour in range(n_hours):
            loc = params['loc'][hour]
            scale = params['scale'][hour]
            samples[:, hour] = np.random.normal(loc=loc, scale=scale, size=n_samples)
            
    elif distribution_type.lower() == 'jsu':
        # Generate samples from Johnson's SU distribution
        for hour in range(n_hours):
            loc = params['loc'][hour]
            scale = params['scale'][hour]
            tailweight = params['tailweight'][hour]
            skewness = params['skewness'][hour]
            
            # Generate standard normal samples
            z = np.random.normal(size=n_samples)
            
            # Transform to JSU distribution
            # This is an approximation of the JSU transformation
            sinh_arg = (z - skewness) / tailweight
            x = np.sinh(sinh_arg)
            samples[:, hour] = loc + scale * x
            
    elif distribution_type.lower() == 'skewt':
        # Generate samples from Skew-t distribution
        for hour in range(n_hours):
            loc = params['loc'][hour]
            scale = params['scale'][hour]
            a = params['a'][hour]
            b = params['b'][hour]
            
            # Generate beta samples for mixing
            beta_samples = np.random.beta(a, b, size=n_samples)
            
            # Generate normal samples
            normal_samples = np.random.normal(size=n_samples)
            
            # Mix based on beta to create skewness
            mixed = np.where(beta_samples > 0.5, 
                            normal_samples, 
                            -normal_samples)
            
            # Scale and shift
            samples[:, hour] = loc + scale * mixed
            
    elif distribution_type.lower() == 'xgboost':
        # Handle XGBoost quantile predictions
        if 'quantiles' not in params or 'values' not in params:
            raise ValueError("XGBoost parameters must contain 'quantiles' and 'values'")
        
        quantiles = np.array(params['quantiles'])
        values = np.array(params['values'])  # shape: (24, n_quantiles)
        
        if len(values.shape) != 2:
            raise ValueError(f"XGBoost values should be 2D array, got shape {values.shape}")
        
        n_hours, n_quantiles = values.shape
        
        if len(quantiles) != n_quantiles:
            raise ValueError(f"Quantiles length ({len(quantiles)}) doesn't match values columns ({n_quantiles})")
        
        samples = np.zeros((n_samples, n_hours))
        
        # Generate samples using inverse transform sampling for each hour
        for hour in range(n_hours):
            # Generate uniform random numbers
            u = np.random.uniform(0, 1, n_samples)
            
            # Get quantile values for this hour
            hour_quantiles = values[hour, :]
            
            # Interpolate to get samples
            # Handle edge cases for extrapolation beyond [0,1]
            samples[:, hour] = np.interp(u, quantiles, hour_quantiles)

    else:
        raise ValueError(f"Unsupported distribution type: {distribution_type}")
        
    return samples

def find_available_runs(distribution, zone):
    """
    Find available model runs for a given distribution and zone
    
    Parameters:
    -----------
    distribution : str
        Distribution type (e.g., 'normal', 'jsu')
    zone : str
        Zone identifier (e.g., 'no1', 'no2')
        
    Returns:
    --------
    list: List of available run IDs
    """
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'forecasts')
    base_path = Path(root_dir)
    pattern = f"*_{zone}_{distribution.lower()}_*"
    
    # Find all matching directories
    matching_dirs = list(base_path.glob(pattern))
    
    # Extract run IDs
    run_ids = []
    for dir_path in matching_dirs:
        if not dir_path.is_dir():
            continue
            
        # Extract the run ID from the directory name
        dir_parts = dir_path.name.split('_')
        if len(dir_parts) >= 4:
            run_id = dir_parts[-1]
            
            # Skip ensemble directories
            if run_id in ['hEns', 'vEns']:
                continue
                
            # Add numeric run IDs
            if run_id.isdigit():
                run_ids.append(int(run_id))
    
    # Sort run IDs
    run_ids = sorted(list(set(run_ids)))
    
    return run_ids

def process_all_zones_distributions():
    """
    Process all available zones and distributions
    
    Parameters:
    -----------
    """
    root_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results', 'forecasts')
    base_path = Path(root_dir)
    
    if not base_path.exists():
        raise ValueError(f"Forecasts directory not found: {base_path}")
    
    # Find all available combinations
    combinations = []
    
    # Pattern: forecasts_no1_normal_1, df_forecasts_no3_jsu_2, etc.
    for dir_path in base_path.glob("*_*_*_*"):
        if not dir_path.is_dir():
            continue
            
        dir_parts = dir_path.name.split('_')
        if len(dir_parts) < 4:
            continue
            
        # Check if it matches known patterns
        if dir_parts[0] in ['df_forecasts', 'distparams']:
            zone = None
            distribution = None
            
            # Check if second part is a zone
            if dir_parts[1].startswith('no') and len(dir_parts[1]) == 3 and dir_parts[1][2].isdigit():
                zone = dir_parts[1]
                distribution = dir_parts[2]
            
            if zone and distribution and dir_parts[-1] not in ['hEns', 'vEns']:
                combinations.append((zone, distribution))
    
    # Remove duplicates
    combinations = list(set(combinations))
    
    if not combinations:
        print("No zone-distribution combinations found")
        return
    
    # Process each combination
    for zone, distribution in sorted(combinations):
        print(f"\n{'='*60}\nProcessing {zone} - {distribution}\n{'='*60}")
        
        # Find available runs
        run_ids = find_available_runs(distribution, zone)
        
        if len(run_ids) < 2:
            print(f"  Not enough runs for ensemble (need at least 2, found {len(run_ids)})")
            continue
            
        print(f"  Found {len(run_ids)} runs: {run_ids}")
        
        # Create ensembles
        try:
            for method in ['horizontal', 'vertical']:
                combine_predictions(
                    distribution=distribution,
                    zone=zone,
                    nrs=run_ids,
                    method=method
                )
        except Exception as e:
            print(f"  Error creating ensembles for {zone} - {distribution}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    #process_all_zones_distributions()
    
    for zone in ['no1','no2','no3','no4','no5']:
        for distribution in ['normal','jsu','skewt']:
            print(f"\n{'='*60}\nProcessing {zone} - {distribution}\n{'='*60}")
            run_ids=4
            combine_predictions(
                distribution=distribution,
                zone=zone,
                nrs=run_ids,
                method='horizontal'
            )
        
            combine_predictions(
                distribution=distribution,
                zone=zone,
                nrs=run_ids,
                method='vertical'
            )

    '''
    parser = argparse.ArgumentParser(description='Create efficient ensemble forecasts')
    parser.add_argument('--distribution', type=str, default='skewt',
                        help='Probability distribution type (e.g., normal, jsu, skewt, xgboost)')
    parser.add_argument('--zone', type=str, default='no5',
                        help='Zone identifier (e.g., no1, no2)')
    parser.add_argument('--nrs', nargs='+', type=int, default=[1, 2, 3, 4],
                        help='Model numbers to combine (default: 1 2 3 4)')
    parser.add_argument('--method', choices=['horizontal', 'vertical', 'both'], default='both',
                        help='Ensemble method (default: both)')
    parser.add_argument('--all', action='store_true',
                        help='Process all available zone-distribution combinations')
    
    args = parser.parse_args()
    
    if args.all:
        # Process all available combinations
        process_all_zones_distributions()
    elif args.distribution and args.zone:
        # Process specific distribution and zone
        if args.method in ['horizontal', 'both']:
            combine_predictions(
                distribution=args.distribution,
                zone=args.zone,
                nrs=args.nrs,
                method='horizontal'
            )
        
        if args.method in ['vertical', 'both']:
            combine_predictions(
                distribution=args.distribution,
                zone=args.zone,
                nrs=args.nrs,
                method='vertical'
            )
    else:
        print("Error: Must specify --all or both --distribution and --zone")
        parser.print_help()
    '''