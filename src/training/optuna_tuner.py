import os
import yaml
import numpy as np
import logging
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
import pickle

from ..models.neural_nets import create_probabilistic_model
from ..data.loader import DataProcessor

class OptunaHPTuner:
    """
    Hyperparameter tuner using Optuna for Deep Distributional Neural Networks (DDNN).

    This class implements automated hyperparameter optimization for probabilistic
    forecasting models using the Optuna framework with time-series cross-validation.

    The optimization searches over:
    - Network architecture (number of layers, units per layer)
    - Activation functions (ReLU, Tanh, Softplus)
    - Regularization (dropout rates, L2 penalty, batch normalization)
    - Training parameters (learning rate, batch size)

    Key Features:
    - Time-series aware cross-validation (no data leakage)
    - Early stopping to prevent overfitting
    - Automatic model checkpointing and parameter saving
    - Scalers saved for reproducible inference

    Attributes
    ----------
    config_path : str
        Path to YAML configuration file
    zone : str
        Bidding zone identifier (no1-no5)
    distribution : str
        Probability distribution (JSU, Normal, skewt)
    run_id : str
        Unique identifier for this optimization run
    device : torch.device
        Computation device (cuda or cpu)
    results_dir : str
        Directory where results are saved

    Example
    -------
    >>> tuner = OptunaHPTuner(
    ...     config_path='config/default_config.yml',
    ...     zone='no1',
    ...     distribution='jsu',
    ...     run_id='replication_001'
    ... )
    >>> best_params = tuner.run_optimization(n_trials=128)
    >>> print(f"Best NLL: {best_params['best_value']}")

    Notes
    -----
    - Optimization typically takes 6-12 hours for 128 trials (CPU)
    - Results are saved to results/models/{zone}_{distribution}_{run_id}/
    - Optuna study database allows resuming interrupted optimizations
    - See config/default_config.yml for available configuration options

    References
    ----------
    Optuna: A hyperparameter optimization framework. https://optuna.org/
    """

    def __init__(self, config_path, zone, distribution, run_id):
        """
        Initialize the hyperparameter tuner.

        Parameters
        ----------
        config_path : str
            Path to YAML configuration file containing data ranges,
            model architectures, and training parameters
        zone : str
            Bidding zone identifier (no1, no2, no3, no4, no5)
        distribution : str
            Probability distribution for DDNN output layer:
            - 'JSU': Johnson's SU (4 parameters: location, scale, skewness, tailweight)
            - 'Normal': Gaussian (2 parameters: location, scale)
            - 'skewt': Skewed Student's t (4 parameters)
        run_id : str
            Unique identifier for this optimization run. Used in output paths
            and allows running multiple experiments with same zone/distribution.
        """
        self.config_path = config_path
        self.zone = zone
        self.distribution = distribution
        self.run_id = run_id
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Set up directories
        self.results_dir = f'./results/models/{zone}_{distribution.lower()}_{run_id}'
        self.scaler_path = f'./results/scalers/{zone}_{distribution.lower()}_{run_id}'
        self.logs_dir = './results/logs'
        self.preproc_dir = f'./src/data/preprocessing/{self.zone}_{self.distribution.lower()}_{self.run_id}'
        
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.scaler_path, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.preproc_dir, exist_ok=True)
        
        # Configure logging
        self._setup_logging()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize variables
        self.data_loaded = False
        self.input_size = None
        self.activations = ['relu', 'tanh', 'softplus']
        self.binary_options = [True, False]
        # Feature groups
        self.feature_groups = {}
        '''
        self.feature_groups = {
            'mFRR_price_up': {'start_idx': 0, 'size': 24},
            'mFRR_price_down': {'start_idx': 24, 'size': 24},
            'mFRR_vol_up': {'start_idx': 48, 'size': 24},
            'mFRR_vol_down': {'start_idx': 72, 'size': 24},
            'spot_price': {'start_idx': 96, 'size': 24},
            'spot_price_forecast': {'start_idx': 120, 'size': 24},
            'weather_temperature': {'start_idx': 144, 'size': 24},
            'consumption_forecast_ec00ens': {'start_idx': 168, 'size': 24}
        }
        '''
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.logs_dir}/{self.zone}_{self.distribution.lower()}_{self.run_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(f"{self.zone}_{self.distribution.lower()}_{self.run_id}")
        
    def _load_data(self):
        """Load data for modeling"""
        self.logger.info("Loading and preparing data...")
        
        try:
            data_processor = DataProcessor(
                zone=self.zone,
                config=self.config
            )
            #data_processor.load_data()
            
            self.train_loader_dataset, self.val_loader_dataset, self.feature_groups, self.scaler_X, self.scaler_Y = data_processor.prepare_torch_datasets()
            print("In _load_data, number of tensors in train_loader dataset:", len(self.train_loader_dataset.dataset.tensors))

            # Validate feature groups structure
            if not self.feature_groups:
                self.logger.error("Feature groups dictionary is empty! Check DataProcessor.")
                raise ValueError("Empty feature groups dictionary")
                
            #self.logger.info(f"Loaded feature groups: {list(self.feature_groups.keys())}")
            
            self.X_train, self.Y_train = self.train_loader_dataset.dataset.tensors
            self.X_val, self.Y_val = self.val_loader_dataset.dataset.tensors
            self.logger.info(f"Data loaded successfully with sizes: X_train: {self.X_train.shape}, Y_train {self.Y_train.shape}, X_val {self.X_val.shape}, Y_val {self.Y_val.shape}")
            
            for feature_name, scaler in self.scaler_X.items():
                joblib.dump(scaler, f"{self.scaler_path}/scaler_X_{feature_name}.joblib")
            joblib.dump(self.scaler_Y, f"{self.scaler_path}/scaler_Y.joblib")
            
            self.data_loaded = True
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        if not self.data_loaded:
            self._load_data()
        
        # Suggest hyperparameters
        hidden_layers_config = []
        n_layers = trial.suggest_int('n_layers', 1, 4)
        
        for i in range(n_layers):
            hidden_size = trial.suggest_int(f'hidden_size_{i}', 32, 256)
            dropout_rate = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5)
            hidden_layers_config.append({
                'units': hidden_size,
                'activation': trial.suggest_categorical(f'activation_{i}', self.activations),
                'batch_norm': True,
                'dropout_rate': dropout_rate
            })
            
        # Get filtered input size using colmask
        colmask = torch.zeros(self.X_train.shape[1], dtype=torch.bool) 

        # Perform feature selection for each feature group
        for feature_name, feature_info in self.feature_groups.items():
            # Suggest whether to include this feature group
            if trial.suggest_categorical(feature_name, [True, False]):
                start_idx = feature_info['start_idx']
                size = feature_info['size']
                colmask[start_idx:start_idx + size] = True

        # Calculate filtered input size
        filtered_input_size = int(torch.sum(colmask).item())  # Convert to Python int
        
         # Ensure at least one feature is selected
        if filtered_input_size == 0:
            # Force selection of at least one feature group
            default_feature = list(self.feature_groups.keys())[0]
            feature_info = self.feature_groups[default_feature]
            colmask[feature_info['start_idx']:feature_info['start_idx'] + feature_info['size']] = True
            filtered_input_size = feature_info['size']
            self.logger.warning(f"No features selected. Forcing selection of {default_feature}.")

        # Create datasets directly for premium prediction
        train_dataset = TensorDataset(self.X_train[:, colmask], self.Y_train)
        val_dataset = TensorDataset(self.X_val[:, colmask], self.Y_val)
        
        # Create DataLoaders
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
              
        # Determine distribution parameter names based on distribution type
        if self.distribution.lower() == 'normal':
            param_names = ['loc', 'scale']
        elif self.distribution.lower() == 'jsu':
            param_names = ['loc', 'scale', 'tailweight', 'skewness']
        elif self.distribution.lower() == 'skewt':
            param_names = ['loc', 'scale', 'a', 'b']
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")
        
         # LAYER-SPECIFIC regularization parameters (following DDNN paper)
        lambda_params = {}
        
        # Regularization for each hidden layer (Equation 3)
        for i in range(n_layers):
            lambda_params[f'network_layer_{i}'] = trial.suggest_float(
                f'lambda_network_layer_{i}', 1e-6, 1e-1, log=True
            )
            lambda_params[f'bias_layer_{i}'] = trial.suggest_float(
                f'lambda_bias_layer_{i}', 1e-7, 1e-2, log=True
            )
        
        # Distribution-specific regularization parameters (following DDNN paper Eq. 4)
        for param_name in param_names:
            lambda_params[f'dist_weight_{param_name}'] = trial.suggest_float(
                f'lambda_dist_weight_{param_name}', 1e-8, 1e-2, log=True
            )
            lambda_params[f'dist_bias_{param_name}'] = trial.suggest_float(
                f'lambda_dist_bias_{param_name}', 1e-9, 1e-3, log=True
            )
        
        # Location-scale regularization parameters
        ls_decay_rate = trial.suggest_float('ls_decay_rate', 1.0, 10.0)
        ls_reduction_factor = trial.suggest_float('ls_reduction_factor', 0.2, 0.8)

        # Training parameters
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        max_epochs = 1500
        patience = trial.suggest_int('patience', 10, 100)
        #l2_lambda = trial.suggest_float('l2_lambda', 1e-6, 1e-3, log=True)
        max_grad_norm = trial.suggest_float('max_grad_norm', 0.1, 10.0)
        
        # Create model
        model = create_probabilistic_model(
            distribution=self.distribution,
            input_size=filtered_input_size,
            hidden_layers_config=hidden_layers_config,
            output_size=24,
            ls_decay_rate=ls_decay_rate,
            ls_reduction_factor=ls_reduction_factor
        )

        # Initialize optimizer and loss function
        optimizer = self._create_optimizer_with_param_groups(model, lr, lambda_params, n_layers)
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
               
        # Train and evaluate model
        best_val_loss = self._train_and_evaluate(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=max_epochs,
            patience=patience,
            max_grad_norm=max_grad_norm
        )
        
        return best_val_loss

    def run_optimization(self, n_jobs=1, n_trials=128):
         # FIRST STEP: Preprocess and save data before parallelization
        self.logger.info("Preprocessing data before optimization...")
        
        # Create a data processor
        data_processor = DataProcessor(
            zone=self.zone,
            config=self.config
        )
        
        # Load and process the data
        train_loader, val_loader, feature_groups, scaler_X, scaler_Y = data_processor.prepare_torch_datasets()
        
        # Check if lagged features are present
        has_lagged_features = any('D-' in feature for feature in feature_groups.keys())
        self.logger.info(f"Lagged features {'are' if has_lagged_features else 'are not'} present in the dataset")
        
        # Extract feature groups
        self.feature_groups = feature_groups

        # Extract tensors from dataloaders
        X_train, Y_train = train_loader.dataset.tensors
        X_val, Y_val = val_loader.dataset.tensors
        
        # Save tensors and scalers
        torch.save(X_train, f'{self.preproc_dir}/X_train.pt')
        torch.save(Y_train, f'{self.preproc_dir}/Y_train.pt')
        torch.save(X_val, f'{self.preproc_dir}/X_val.pt')
        torch.save(Y_val, f'{self.preproc_dir}/Y_val.pt')
        
        # Save scalers
        for feature_name, scaler in scaler_X.items():
            joblib.dump(scaler, f"{self.scaler_path}/scaler_X_{feature_name}.joblib")
        joblib.dump(scaler_Y, f"{self.scaler_path}/scaler_Y.joblib")
        
        # Save feature groups
        with open(f'{self.preproc_dir}/feature_groups.pkl', 'wb') as f:
            pickle.dump(feature_groups, f)
        
        self.logger.info(f"Data preprocessed and saved to {self.preproc_dir}")
        
         # Set up storage
        study_name = f'{self.zone}_{self.distribution.lower()}_{self.run_id}'
        db_dir = f'./results/models/{self.zone}_{self.distribution.lower()}_{self.run_id}'
        os.makedirs(db_dir, exist_ok=True)
        storage_path = os.path.abspath(os.path.join(db_dir, f"{study_name}.db"))
        storage_name = f'sqlite:///{storage_path}'
        
        # Check if the study already exists and delete it if it does
        if os.path.exists(storage_path):
            self.logger.info(f"Found existing study at {storage_path}. Deleting to start fresh.")
            try:
                # remove old study
                os.remove(storage_path)
                self.logger.info("Database file removed.")
            except Exception as e:
                self.logger.warning(f"Error while trying to delete existing study: {str(e)}")
                self.logger.warning("Will attempt to continue by creating a new study.")
        
        self.logger.info(f"Starting optimization with {n_trials} trials")
        
        try:
            pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=4)
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=20,
                n_ei_candidates=24,
                seed=42
            )

            study = optuna.create_study(
                study_name=study_name, 
                storage=storage_name, 
                load_if_exists=False,
                direction='minimize',
                pruner=pruner,
                sampler=sampler
            )
            study.optimize(self.objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True) # Edit n_jobs to select number of parallel processes
            
            # Save best parameters
            best_params = study.best_params
                        
            # Log the best parameters
            self.logger.info(f"Best value: {study.best_value}")
            self.logger.info(f"Best params: {best_params}")
                        
            # Simply save all parameters from the best trial
            with open(f'{self.results_dir}/best_params.yaml', 'w') as f:
                yaml.dump(best_params, f)
            
            # Rename the best model (remove existing file first on Windows)
            best_model_temp = f"{self.results_dir}/best_model_temp.pt"
            best_model_final = f"{self.results_dir}/best_model.pt"

            # Remove existing final model if it exists (required on Windows)
            if os.path.exists(best_model_final):
                os.remove(best_model_final)

            os.rename(best_model_temp, best_model_final)
                
            return best_params
            
        except Exception as e:
            self.logger.error(f"Error during optimization: {str(e)}")
            raise
    '''
    def _create_optimizer_with_param_groups(self, model, lr, l2_lambda):
        """Create optimizer with different weight decay for different parameter groups"""
        
        # Separate parameters into groups
        network_params = []
        bias_params = []  
        final_layer_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Final layer parameters (distribution outputs)
            if any(x in name for x in ['network.' + str(len(model.network)-1), 'distribution_layer']):
                final_layer_params.append(param)
            # Bias parameters  
            elif 'bias' in name:
                bias_params.append(param)
            # Regular network weights
            else:
                network_params.append(param)
        
        # Create parameter groups with different weight decay
        param_groups = [
            {'params': network_params, 'weight_decay': l2_lambda},           # Full regularization
            {'params': bias_params, 'weight_decay': l2_lambda * 0.1},       # Light bias regularization  
            {'params': final_layer_params, 'weight_decay': l2_lambda * 0.01} # Very light for distribution params
        ]
        
        return torch.optim.Adam(param_groups, lr=lr)
    '''

    def _train_and_evaluate(self, model, optimizer, train_loader, val_loader, max_epochs, patience, max_grad_norm=1.0):
        """
        Train and evaluate a model with L2 regularization and gradient clipping
        
        Parameters:
        -----------
        model : nn.Module
            PyTorch model to train
        optimizer : torch.optim.Optimizer
            Optimizer for training
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        max_epochs : int
            Maximum number of epochs to train
        patience : int
            Number of epochs to wait for improvement before early stopping
        l2_lambda : float
            L2 regularization strength
        max_grad_norm : float
            Maximum norm for gradient clipping
            
        Returns:
        --------
        float
            Validation loss
        """
        model.to(self.device)
        best_val_loss = float('inf')
        best_epoch = 0
        no_improvement = 0
        
        # Create better optimizer with parameter groups (ignore passed optimizer)
        #optimizer = self._create_optimizer_with_param_groups(model, optimizer.param_groups[0]['lr'], l2_lambda)
        
        for epoch in range(max_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
                
            for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                dist = model(X_batch)
                
                # Simple negative log-likelihood - let proper zero-inflation handle the mixture
                nll_loss = -dist.log_prob(y_batch).mean()
                
                # No manual regularization needed - handled by optimizer weight_decay
                loss = nll_loss
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                optimizer.step()
                train_loss += nll_loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    dist = model(X_batch)
                    
                    # Simple negative log-likelihood
                    loss = -dist.log_prob(y_batch).mean()
                        
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
           # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                no_improvement = 0
                
                # Save best model
                torch.save(model.state_dict(), f"{self.results_dir}/best_model_temp.pt")
            else:
                no_improvement += 1
                
            # Log progress every 10 epochs
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                self.logger.info(f"Epoch {epoch}/{max_epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
            # Early stopping
            if no_improvement >= patience:
                self.logger.info(f"Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val loss {best_val_loss:.6f}")
                break

        return best_val_loss

    def _create_optimizer_with_param_groups(self, model, lr, lambda_params, n_layers):
        """
        Create optimizer with layer-specific regularization following DDNN paper Eq(3)+(4)
        """
        # Organize parameters by layer and type
        layer_weights = {i: [] for i in range(n_layers)}
        layer_biases = {i: [] for i in range(n_layers)}
        distribution_param_groups = {}
        
        # Initialize distribution parameter groups
        for param_name in model.param_names:
            distribution_param_groups[param_name] = {'weights': [], 'biases': []}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Shared network parameters (hidden layers) - GROUP BY LAYER
            if name.startswith('shared_network'):
                # Extract layer number from parameter name
                # e.g., 'shared_network.0.weight' -> layer 0
                #       'shared_network.4.weight' -> layer 1 (since 4/4=1, accounting for BN/activation/dropout)
                parts = name.split('.')
                if len(parts) >= 2 and parts[1].isdigit():
                    layer_idx = int(parts[1])
                    
                    # Map sequential layer indices to logical layer numbers
                    # shared_network.0 = layer 0, shared_network.4 = layer 1, etc.
                    # This assumes each layer has 4 components: Linear, BN, Activation, Dropout
                    logical_layer = layer_idx // 4
                    
                    if logical_layer < n_layers:  # Make sure we don't exceed expected layers
                        if name.endswith('.weight') and 'BatchNorm' not in name:
                            layer_weights[logical_layer].append(param)
                        elif name.endswith('.bias') and 'BatchNorm' not in name:
                            layer_biases[logical_layer].append(param)
            
            # Distribution parameter layers (same as before)
            else:
                param_assigned = False
                for param_name in model.param_names:
                    if name == f'{param_name}_layer.weight':
                        distribution_param_groups[param_name]['weights'].append(param)
                        param_assigned = True
                        break
                    elif name == f'{param_name}_layer.bias':
                        distribution_param_groups[param_name]['biases'].append(param)
                        param_assigned = True
                        break
                
                if not param_assigned:
                    print(f"WARNING: Parameter {name} was not assigned to any group!")
        
        # Create parameter groups
        param_groups = []
        
        # Add layer-specific parameter groups
        for layer_idx in range(n_layers):
            if layer_weights[layer_idx]:
                param_groups.append({
                    'params': layer_weights[layer_idx],
                    'weight_decay': lambda_params.get(f'network_layer_{layer_idx}', 1e-5)
                })
            
            if layer_biases[layer_idx]:
                param_groups.append({
                    'params': layer_biases[layer_idx],
                    'weight_decay': lambda_params.get(f'bias_layer_{layer_idx}', 1e-6)
                })
        
        # Add distribution parameter groups (same as before)
        for param_name in model.param_names:
            if distribution_param_groups[param_name]['weights']:
                param_groups.append({
                    'params': distribution_param_groups[param_name]['weights'],
                    'weight_decay': lambda_params.get(f'dist_weight_{param_name}', 1e-7)
                })
            
            if distribution_param_groups[param_name]['biases']:
                param_groups.append({
                    'params': distribution_param_groups[param_name]['biases'],
                    'weight_decay': lambda_params.get(f'dist_bias_{param_name}', 1e-8)
                })
         # Count parameters and validate regularization groups
        total_params = 0
        for group in param_groups:
            total_params += sum(p.numel() for p in group['params'])
        
        # Expected number of regularization groups:
        # - 2 groups per hidden layer (weights + biases) = 2 * n_layers
        # - 2 groups per distribution parameter (weights + biases) = 2 * len(param_names)
        expected_groups = 2 * n_layers + 2 * len(model.param_names)
        actual_groups = len(param_groups)
        
        status = "OK" if actual_groups == expected_groups else f"MISMATCH (expected {expected_groups})"
        self.logger.info(f"Layer-wise regularization {status}: {actual_groups} groups, {total_params:,} params")
        return torch.optim.Adam(param_groups, lr=lr)
