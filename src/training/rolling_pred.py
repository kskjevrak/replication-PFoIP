
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import sys
import os
import optuna
import time
import joblib
from datetime import datetime, timedelta
import json
import glob
import yaml

from ..models.neural_nets import create_probabilistic_model, create_state_classifier

def process_one_day(inp):
    params, dayno = inp
    
    # Calculate the date for this prediction
    prediction_date = pd.Timestamp('2024-03-18', tz='Europe/Oslo') + pd.Timedelta(days=dayno) 
    
    # Create window of training data (730 days = 2 years)
    end_idx = prediction_date + pd.Timedelta(days=1) # Add one day to include the prediction day
    start_idx = end_idx - pd.Timedelta(days=730+1)
    
    # Get data window
    window_data = data[(data.index >= start_idx) & (data.index < end_idx)]
    
    if len(window_data) < 730 * 24:  # Check if we have enough data
        print(f"Insufficient data for date {prediction_date}")
        return None
        
    # Prepare matrices
    Y = np.zeros((730, 24))
    for d in range(7, 730):
            Y[d, :] = window_data.iloc[d*24:(d+1)*24]['premium'].to_numpy()
    
    X = np.zeros((730+1, INPUT_SIZE))
    try:
        for d in range(7, 730+1):
            # Previous day features
            X[d, :24] = window_data.iloc[(d-1)*24:d*24]['premium'].to_numpy()
            #X[d, 24:48] = window_data.iloc[(d-2)*24:(d-1)*24]['premium'].to_numpy()
            #X[d, 48:72] = window_data.iloc[(d-7)*24:(d-6)*24]['premium'].to_numpy()
            
            #mFRR features
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['mFRR_price_up'].to_numpy()
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['mFRR_price_down'].to_numpy()
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['mFRR_vol_up'].to_numpy()
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['mFRR_vol_down'].to_numpy()

            # Spot price
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['spot_price'].to_numpy()
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['spot_price_forecast'].to_numpy()

            # Weather features
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['weather_temperature'].to_numpy()

            # Consumption forecast
            X[d, :24] = window_data.iloc[d*24:(d+1)*24]['consumption_forecast_ec00ens'].to_numpy()

            #TODO: Add wind/precitipation/reservoir filling rate

    except Exception as e:
        print(f"Error in feature preparation for day {dayno}: {str(e)}")
        return None
    
    # Load scalers
    scaler_path = f'./results/scalers/{zone}_{distribution.lower()}_{nr}'
    scaler_X = joblib.load(f'{scaler_path}/scaler_X.joblib')
    scaler_Y = joblib.load(f'{scaler_path}/scaler_Y.joblib')
    
    # Scale data
    X_scaled = scaler_X.transform(X[:,:-1])
    X_scaled = np.hstack([X_scaled, X[:,-1].reshape(-1,1)])
    Y_scaled = scaler_Y.transform(Y.reshape(-1, 1)).reshape(Y.shape)
    
    Xf_scaled = X_scaled[-1:, :] # Save last feature for forecasting

    # Feature selection based on best parameters
    colmask = [False] * INPUT_SIZE
    
    if params['mFRR_price_up']:
        colmask[:24] = [True] * 24
    if params['mFRR_price_down']:
        colmask[:24] = [True] * 24
    if params['mFRR_vol_up']:
        colmask[:24] = [True] * 24
    if params['mFRR_vol_down']:
        colmask[:24] = [True] * 24
    if params['spot_price']:
        colmask[:24] = [True] * 24
    if params['spot_price_forecast']:
        colmask[:24] = [True] * 24
    if params['weather_temperature']:
        colmask[:24] = [True] * 24
    if params['consumption_forecast_ec00ens']:
        colmask[:24] = [True] * 24

    #TODO: Add wind/precitipation/reservoir filling rate


    
    # Apply feature selection
    X_selected = X_scaled[:, colmask]
    Xf_selected = Xf_scaled[:, colmask]  # For forecasting
    
    # STAGE 1: State Classification
    # Load state classifier model
    classifier_path = f'./results/models/{zone}_{distribution.lower()}_{nr}/state_classifier.pt'
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load classifier model
    classifier = create_state_classifier(
        input_size=X_selected.shape[1],
        hidden_layers_config=[{'units': 64, 'activation': 'relu', 'batch_norm': True, 'dropout_rate': 0.1}]
    )
    classifier.load_state_dict(torch.load(classifier_path))
    classifier.to(device)
    classifier.eval()
    
    # Generate state predictions for forecast day
    with torch.no_grad():
        Xf_tensor = torch.FloatTensor(Xf_selected).to(device)
        _, state_probs = classifier(Xf_tensor)
        
    # Record most likely state
    _, predicted_state = torch.max(state_probs, 1)
    predicted_state = predicted_state.item()  # 0=no regulation, 1=up, 2=down
    
    # STAGE 2: Premium Magnitude Prediction
    # Add state probabilities to features
    Xf_with_states = np.hstack([Xf_selected, state_probs.cpu().numpy()])
    
    # Create training dataset with state information for model training
    X_train = X_selected[7:-1]  # Skip first week and forecast day
    Y_train = Y_scaled[7:]      # Skip first week
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    Y_train_tensor = torch.FloatTensor(Y_train).to(device)
    
    # Add state information to training data
    with torch.no_grad():
        _, train_state_probs = classifier(X_train_tensor)
    
    X_train_with_states = torch.cat([X_train_tensor, train_state_probs], dim=1)
    
    # Create model with expanded input size (includes state probabilities)
    hidden_layers_config = [
        {
            'units': params['neurons_1'],
            'activation': params['activation_1'],
            'batch_norm': True,
            'dropout_rate': params['dropout_rate'] if params['dropout'] else None,
        },
        {
            'units': params['neurons_2'],
            'activation': params['activation_2'],  
            'batch_norm': True,
            'dropout_rate': params['dropout_rate'] if params['dropout'] else None
        }
    ]
    
    # Input size includes state probabilities (3 additional features)
    augmented_input_size = X_train_with_states.shape[1]
    
    # Create premium magnitude model
    try:
        model = create_probabilistic_model(
            distribution=distribution,
            input_size=augmented_input_size,
            hidden_layers_config=hidden_layers_config,
            output_size=24  # Hours to predict
        ).to(device)
    except Exception as e:
        print(f"Error in model creation for day {dayno}: {str(e)}")
        return None
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # Training loop settings
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    # Create train/val split
    try:
        perm = np.random.permutation(np.arange(X_train_with_states.shape[0]))
        VAL_DATA = .2
        trainsubset = perm[:int((1 - VAL_DATA)*len(perm))]
        valsubset = perm[int((1 - VAL_DATA)*len(perm)):]
    except Exception as e:
        print(f"Error in data preparation for day {dayno}: {str(e)}")
        return None

    # Create datasets for train and validation
    train_dataset = TensorDataset(X_train_with_states[trainsubset], Y_train_tensor[trainsubset])
    val_dataset = TensorDataset(X_train_with_states[valsubset], Y_train_tensor[valsubset])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    try:
        # Training loop
        for epoch in range(1500):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                dist = model(batch_x)
                loss = -dist.log_prob(batch_y).mean()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params['max_grad_norm'])
                
                optimizer.step()
                
            # Validation step
            model.eval()
            with torch.no_grad():
                val_losses = []
                for val_x, val_y in val_loader:
                    val_dist = model(val_x)
                    val_loss = -val_dist.log_prob(val_y).mean()
                    val_losses.append(val_loss.item())
                val_loss = np.mean(val_losses)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
    except Exception as e:
        print(f"Error in model training for day {dayno}: {str(e)}")
        return None
        
    # Make prediction for next day
    model.eval()
    with torch.no_grad():
        try:         
            Xf_tensor = torch.FloatTensor(Xf_with_states).to(device)
        except Exception as e:
            print(f"Error in forecast feature preparation for day {dayno}: {str(e)}")
            return None
        
        try:
            # Generate predictions
            dist = model(Xf_tensor)
            samples = dist.sample((10000,))
            
            predictions = scaler_Y.inverse_transform(samples.cpu().numpy().reshape(-1, 24))
            
            # Apply sign correction based on predicted state
            if predicted_state == 2:  # Down regulation
                predictions = -np.abs(predictions)
            elif predicted_state == 1:  # Up regulation
                predictions = np.abs(predictions)
            # For state 0 (no regulation), leave as is (should be close to zero)
            
            df_predictions = pd.DataFrame(index=window_data.index[-24:])
            df_predictions['actual'] = window_data.loc[window_data.index[-24:], 'premium'].to_numpy()
            df_predictions['forecast'] = pd.NA
            df_predictions['state'] = predicted_state
            df_predictions.loc[window_data.index[-24:], 'forecast'] = predictions.mean(axis=0)
        except Exception as e:
            print(f"Error in prediction generation for day {dayno}: {str(e)}")
            return None
            
    # Create output directories if they don't exist
    forecasts_dir = f'./results/forecasts/forecasts_probNN_{distribution.lower()}_{nr}'
    df_forecasts_dir = f'./results/forecasts/df_forecasts_probNN_{distribution.lower()}_{nr}'
    params_dir = f'./results/forecasts/distparams_probNN_{distribution.lower()}_{nr}'
    
    os.makedirs(forecasts_dir, exist_ok=True)
    os.makedirs(df_forecasts_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)
    
    try:
        # Save raw predictions
        np.savez_compressed(
            os.path.join(forecasts_dir, f"{prediction_date.strftime('%Y-%m-%d')}.npz"),
            predictions=predictions
        )
    except Exception as e:
        print(f"Error in prediction saving for day {dayno}: {str(e)}")
        return None
        
    try:     
        # Save distribution parameters
        if distribution == 'Normal':
            params_dict = {
                'loc': dist.loc.cpu().numpy().tolist(),
                'scale': dist.scale.cpu().numpy().tolist()
            }
        elif distribution == 'JSU':
            params_dict = {
                'loc': dist.loc.cpu().numpy().tolist(),
                'scale': dist.scale.cpu().numpy().tolist(),
                'tailweight': dist.tailweight.cpu().numpy().tolist(),
                'skewness': dist.skewness.cpu().numpy().tolist()
            }
        
        with open(os.path.join(params_dir, f"{prediction_date.strftime('%Y-%m-%d')}.json"), 'w') as f:
            json.dump(params_dict, f)
            
        # Save state information
        state_info = {
            'predicted_state': int(predicted_state),
            'state_probabilities': state_probs.cpu().numpy().tolist()[0],
            'state_names': ['no_regulation', 'up_regulation', 'down_regulation']
        }
        
        with open(os.path.join(params_dir, f"{prediction_date.strftime('%Y-%m-%d')}_state.json"), 'w') as f:
            json.dump(state_info, f)
    except Exception as e:
        print(f"Error in distribution saving for day {dayno}: {str(e)}")
        return None
            
    # Save prediction dataframe
    np.savetxt(
        os.path.join(df_forecasts_dir, f"{prediction_date.strftime('%Y-%m-%d')}.csv"),
        df_predictions,
        delimiter=',',
        fmt='%.3f'
    )
    
    print(f"Completed predictions for {prediction_date}, state: {state_info['state_names'][predicted_state]}")
    return prediction_date.strftime('%Y-%m-%d')

if __name__ == "__main__":
    try:
        # Parse command-line arguments
        if len(sys.argv) < 4:
            print("Usage: python rolling_pred.py <zone> <distribution> <run_id>")
            sys.exit(1)
            
        zone = sys.argv[1]
        distribution = sys.argv[2]
        nr = sys.argv[3]
        
        # Load best parameters from study
        study_name = f'{zone}_selection_prob_{distribution.lower()}_{nr}'
        storage_name = f"sqlite:///{os.path.abspath(f'./results/{study_name}.db')}"
        
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            best_params = study.best_params
        except Exception as e:
            sys.exit(f"Error in loading best parameters: {str(e)}")
        
        print("Loaded best parameters:", best_params)
        
        # Load data
        data = pd.read_parquet(f'./data/premium_{zone}.parquet', engine='pyarrow')
        data.index = pd.to_datetime(data.index, utc=True).tz_convert('Europe/Oslo')
        
        # Define input size based on feature engineering
        INPUT_SIZE = 241
        
        # Calculate number of days to predict
        config_path = f'./config/default_config.yml'  # Or another path you specify
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Get test dates from config with fallbacks to default values
        test_start = pd.Timestamp(config.get('prediction', {}).get('test_start', '2024-03-18'), tz='Europe/Oslo')
        test_end = pd.Timestamp(config.get('prediction', {}).get('test_end', '2024-10-31'), tz='Europe/Oslo')
        total_days = (test_end - test_start).days + 1
        
        # Create list of arguments for parallel processing
        inputlist = [(best_params, day) for day in range(total_days)]
        
        # Process days in parallel
        from multiprocessing import Pool
        
        try:
            # Run parallel processing with explicit process count
            num_processes = min(os.cpu_count() // 2, 4)  # Limit to 4 processes
            with Pool(processes=num_processes) as pool:
                results = pool.map(process_one_day, inputlist)
        except Exception as e:
            print(f"Error in parallel execution: {str(e)}")
            sys.exit(1)
            
        # Filter out None results and print summary
        completed_dates = [r for r in results if r is not None]
        print(f"Completed predictions for {len(completed_dates)} out of {total_days} days")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)
