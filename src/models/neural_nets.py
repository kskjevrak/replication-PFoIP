import torch
import torch.nn as nn
from .layers import LSJSULayer, LSNormalLayer, LSJFSkewTLayer

def create_probabilistic_model(distribution, input_size, hidden_layers_config, output_size=24, ls_decay_rate=5.0, ls_reduction_factor=0.5):
    """
    Create a Deep Distributional Neural Network (DDNN) for probabilistic forecasting.

    This factory function builds a neural network that outputs parameters of a
    probability distribution rather than point forecasts. The architecture follows
    the DDNN framework with:
    - Shared hidden layers for feature extraction
    - Separate output heads for each distribution parameter
    - Distribution-specific transformation layers (non-learnable)

    The model is trained to minimize negative log-likelihood, learning to predict
    the full conditional distribution of the target variable given input features.

    Parameters
    ----------
    distribution : str
        Probability distribution family. Supported options:
        - 'normal' or 'Normal': Gaussian distribution (2 parameters: μ, σ)
        - 'jsu' or 'JSU': Johnson's SU (4 parameters: μ, σ, γ, δ)
        - 'skewt' or 'skewt': Skewed Student's t (4 parameters: μ, σ, a, b)
    input_size : int
        Number of input features. Must match preprocessed data dimensions.
    hidden_layers_config : list of dict
        Configuration for hidden layers. Each dict specifies:
        - 'units' (int): Number of neurons
        - 'activation' (str): Activation function ('relu', 'tanh', 'softplus')
        - 'dropout_rate' (float, optional): Dropout probability
        - 'batch_norm' (bool, optional): Whether to use batch normalization
    output_size : int, default=24
        Number of forecast time steps (hours). Default 24 for daily forecasts.
    ls_decay_rate : float, default=5.0
        Decay rate for layer normalization in distribution transformation.
    ls_reduction_factor : float, default=0.5
        Reduction factor for layer normalization.

    Returns
    -------
    torch.nn.Module
        DDNN model ready for training or inference

    Raises
    ------
    ValueError
        If distribution is not supported or input_size is invalid

    Example
    -------
    >>> hidden_config = [
    ...     {'units': 256, 'activation': 'relu', 'dropout_rate': 0.1, 'batch_norm': True},
    ...     {'units': 128, 'activation': 'tanh', 'dropout_rate': 0.05, 'batch_norm': False}
    ... ]
    >>> model = create_probabilistic_model(
    ...     distribution='jsu',
    ...     input_size=192,
    ...     hidden_layers_config=hidden_config,
    ...     output_size=24
    ... )
    >>> print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    Notes
    -----
    Architecture Details:
    - Shared hidden network: Processes all input features identically
    - Parameter-specific heads: Each distribution parameter gets its own linear layer
    - Transformation layer: Ensures parameters satisfy constraints (e.g., σ > 0)

    Distribution Parameters:
    - Normal: μ (location), σ (scale)
    - JSU: μ (location), σ (scale), γ (skewness), δ (tailweight)
    - Skewed-t: μ (location), σ (scale), a (skewness), b (degrees of freedom)

    The model is trained end-to-end with negative log-likelihood loss.

    See Also
    --------
    OptunaHPTuner : For optimizing hidden_layers_config
    LSJSULayer, LSNormalLayer, LSJFSkewTLayer : Distribution transformation layers

    References
    ----------
    Deep Distributional Neural Networks framework for probabilistic forecasting.
    """
    if not isinstance(input_size, int):
        raise ValueError(f"input_size must be an integer, got {type(input_size)}")
    if input_size <= 0:
        raise ValueError(f"Invalid input_size: {input_size}")
    
    print(f"Creating DDNN with distribution={distribution}, input_size={input_size}, output_size={output_size}")
    
    class DDNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            
            self.output_size = output_size
            self.distribution = distribution
            self.ls_decay_rate = ls_decay_rate
            self.ls_reduction_factor = ls_reduction_factor
            
            # Build shared hidden layers (same as before)
            layers = []
            current_size = input_size
            
            activation_map = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'softplus': nn.Softplus
            }
            
            for i, config in enumerate(hidden_layers_config):
                activation_fn = activation_map[config['activation'].lower()]
                layers.extend([
                    nn.Linear(current_size, config['units']),
                    nn.BatchNorm1d(config['units']) if config.get('batch_norm', False) else nn.Identity(),
                    activation_fn(),
                    nn.Dropout(config['dropout_rate']) if config.get('dropout_rate') else nn.Identity()
                ])
                current_size = config['units']
                print(f"  Hidden Layer {i+1}: {current_size} units, {config['activation']} activation")
            
            # Shared hidden network
            self.shared_network = nn.Sequential(*layers)
            
            # Separate final layers for each distribution parameter
            if distribution.lower() == 'normal':
                self.param_names = ['loc', 'scale']
                self.loc_layer = nn.Linear(current_size, output_size)
                self.scale_layer = nn.Linear(current_size, output_size)
                
                print(f"  Distribution layers: loc ({current_size}->{output_size}), scale ({current_size}->{output_size})")
                
            elif distribution.lower() == 'jsu':
                self.param_names = ['loc', 'scale', 'tailweight', 'skewness']
                self.loc_layer = nn.Linear(current_size, output_size)
                self.scale_layer = nn.Linear(current_size, output_size)
                self.tailweight_layer = nn.Linear(current_size, output_size)
                self.skewness_layer = nn.Linear(current_size, output_size)
                
                print(f"  Distribution layers: loc, scale, tailweight, skewness ({current_size}->{output_size} each)")
                
            elif distribution.lower() == 'skewt':
                self.param_names = ['loc', 'scale', 'a', 'b']
                self.loc_layer = nn.Linear(current_size, output_size)
                self.scale_layer = nn.Linear(current_size, output_size)
                self.a_layer = nn.Linear(current_size, output_size)
                self.b_layer = nn.Linear(current_size, output_size)
                
                print(f"  Distribution layers: loc, scale, a, b ({current_size}->{output_size} each)")
                
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            
            # Distribution processing layer (no learnable parameters)
            if distribution.lower() == 'normal':
                self.distribution_layer = LSNormalLayer(output_size, ls_decay_rate, ls_reduction_factor)
            elif distribution.lower() == 'jsu':
                self.distribution_layer = LSJSULayer(output_size, ls_decay_rate, ls_reduction_factor)
            elif distribution.lower() == 'skewt':
                self.distribution_layer = LSJFSkewTLayer(output_size, ls_decay_rate, ls_reduction_factor)
            
            # Initialize weights
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        def forward(self, x):
            # Forward through shared hidden layers
            hidden_output = self.shared_network(x)
            
            # Forward through separate distribution parameter layers
            if self.distribution.lower() == 'normal':
                loc_out = self.loc_layer(hidden_output)
                scale_out = self.scale_layer(hidden_output)
                # Concatenate in the expected order
                combined_output = torch.cat([loc_out, scale_out], dim=-1)
                
            elif self.distribution.lower() == 'jsu':
                loc_out = self.loc_layer(hidden_output)
                scale_out = self.scale_layer(hidden_output)
                tailweight_out = self.tailweight_layer(hidden_output)
                skewness_out = self.skewness_layer(hidden_output)
                # Concatenate in the expected order
                combined_output = torch.cat([loc_out, scale_out, tailweight_out, skewness_out], dim=-1)
                
            elif self.distribution.lower() == 'skewt':
                loc_out = self.loc_layer(hidden_output)
                scale_out = self.scale_layer(hidden_output)
                a_out = self.a_layer(hidden_output)
                b_out = self.b_layer(hidden_output)
                # Concatenate in the expected order
                combined_output = torch.cat([loc_out, scale_out, a_out, b_out], dim=-1)
            
            # Process through distribution layer
            dist = self.distribution_layer(combined_output)
            return dist
    
    return DDNNModel()

'''
def create_probabilistic_model(distribution, input_size, hidden_layers_config, output_size=24, ls_decay_rate=5.0, ls_reduction_factor=0.5):
    """
    Factory function that creates a probabilistic neural network model with residual connections.
    
    Parameters:
    -----------
    distribution : str
        Distribution type ('normal', 'jsu', or 'skewt')
    input_size : int
        Size of the input features
    hidden_layers_config : list of dict
        Configuration for hidden layers. Each dict should contain:
        - 'units': number of units
        - 'activation': activation function ('relu', 'tanh', 'softplus')
        - 'batch_norm': whether to use batch normalization (bool)
        - 'dropout_rate': dropout rate (float or None)
    output_size : int, optional
        Size of the output dimension, default is 24 (hours)
            
    Returns:
    --------
    nn.Module
        Configured probabilistic neural network model
    """
    # Add input validation at function level
    if not isinstance(input_size, int):
        raise ValueError(f"input_size must be an integer, got {type(input_size)}")
    if input_size <= 0:
        raise ValueError(
            f"Invalid input_size: {input_size}. Input size must be greater than 0.\n"
            "This usually means no features were selected during feature selection.\n"
            "Check the colmask calculation in optuna_tuner.py and ensure at least one feature group is selected."
        )
    
    print(f"Creating probabilistic model with distribution={distribution}, input_size={input_size}, output_size={output_size}")
    
    class ProbModelWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            current_size = input_size

            self.output_size = output_size  # Store for initialization
            self.distribution = distribution  # Store for initialization

            # Store LS parameters
            self.ls_decay_rate = ls_decay_rate
            self.ls_reduction_factor = ls_reduction_factor
            
            print(f"ProbModelWrapper initialized with input_size={input_size}")
            
            # Map string activation names to PyTorch classes
            activation_map = {
                'relu': nn.ReLU,
                'tanh': nn.Tanh,
                'softplus': nn.Softplus
            }
            
            # Build network layers with residual connections
            for i, config in enumerate(hidden_layers_config):
                activation_fn = activation_map[config['activation'].lower()]
                layers.extend([
                    nn.Linear(current_size, config['units']),
                    nn.BatchNorm1d(config['units']) if config.get('batch_norm', False) else nn.Identity(),
                    activation_fn(),
                    nn.Dropout(config['dropout_rate']) if config.get('dropout_rate') else nn.Identity()
                ])
                current_size = config['units']
                print(f"  Layer {i+1}: {current_size} units, {config['activation']} activation")
            
            if distribution.lower() == 'normal':
                final_size = output_size * 2  # loc, scale
            elif distribution.lower() == 'jsu':
                final_size = output_size * 4  # loc, scale, tailweight, skewness
            elif distribution.lower() == 'skewt':
                final_size = output_size * 4  # loc, scale, a, b
            
            self.final_size = final_size


            layers.append(nn.Linear(current_size, final_size))
            print(f"  Output layer: {current_size} -> {final_size} (for {distribution} distribution)")
            
            self.network = nn.Sequential(*layers)
            
            # Create appropriate distribution layer with LS parameters
            if distribution.lower() == 'normal':
                self.distribution_layer = LSNormalLayer(
                    output_size, ls_decay_rate, ls_reduction_factor
                )
            elif distribution.lower() == 'jsu':
                self.distribution_layer = LSJSULayer(
                    output_size, ls_decay_rate, ls_reduction_factor
                )
            elif distribution.lower() == 'skewt':
                self.distribution_layer = LSJFSkewTLayer(
                    output_size, ls_decay_rate, ls_reduction_factor
                )
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")
            
            # Initialize weights using Kaiming initialization
            self.apply(self._init_weights)
            
        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            """
            Forward pass through the network.
            
            Parameters:
            -----------
            x : torch.Tensor
                Input features
                
            Returns:
            --------
            torch.distributions.Distribution
                Probability distribution over the output space
            """
            # Debug input shape
            if hasattr(torch, 'is_tensor') and torch.is_tensor(x) and x.dim() > 0:
                expected_input_size = input_size
                actual_input_size = x.shape[1] if x.dim() > 1 else x.shape[0]
                
                if actual_input_size != expected_input_size:
                    print(f"WARNING: Model expected input size {expected_input_size}, got {actual_input_size}")
            
            # Forward pass through network
            outputs = self.network(x)
            
            # Create distribution
            dist = self.distribution_layer(outputs)
            
            return dist
    
    return ProbModelWrapper()
'''