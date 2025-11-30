import torch
import torch.nn as nn
from torch.distributions import Normal
from .distributions import JSUDistribution, JFSkewTDistribution, ZeroInflatedJSUDistribution, ZeroInflatedNormalDistribution, ZeroInflatedJFSkewTDistribution

class NormalLayer(nn.Module):
    """
    Layer that outputs parameters for a Normal distribution.
    
    Parameters:
    -----------
    output_size : int
        Size of the output dimension
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs):
        """
        Forward pass that transforms inputs into Normal distribution parameters.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor with shape (..., 2*output_size)
            
        Returns:
        --------
        torch.distributions.Normal
            Normal distribution parameterized by the inputs
        """
        print(f"Distribution layer input shape: {inputs.shape}")
        
        # Add input validation
        if torch.isnan(inputs).any():
            print("NaN detected in inputs to NormalLayer")
            
        # Clamp inputs to prevent extreme values
        inputs = torch.clamp(inputs, min=-1e6, max=1e6)
        
        loc = inputs[..., :self.output_size]
        # More conservative clamping for scale
        scale = torch.clamp(
            1e-3 + torch.nn.functional.softplus(inputs[..., self.output_size:2*self.output_size]),
            min=1e-3,
            max=10.0
        )
        
        return Normal(loc=loc, scale=scale)

class JSULayer(nn.Module):
    """
    Layer that outputs parameters for a JSU (Johnson's SU) distribution.
    
    Parameters:
    -----------
    output_size : int
        Size of the output dimension
    """
    
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, inputs):
        """
        Forward pass that transforms inputs into JSU distribution parameters.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor with shape (..., 4*output_size)
            
        Returns:
        --------
        JSUDistribution
            JSU distribution parameterized by the inputs
        """
        inputs = torch.clamp(inputs, min=-1e6, max=1e6)
        
        loc = inputs[..., :self.output_size]
        scale = torch.clamp(
            1e-3 + torch.nn.functional.softplus(inputs[..., self.output_size:2*self.output_size]), 
            min=1e-6,
            max=1e2
        )
        tailweight = torch.clamp(
            1.0 + torch.nn.functional.softplus(inputs[..., 2*self.output_size:3*self.output_size]),
            min=1e-6,
            max=1e2
        )
        skewness = torch.clamp(
            inputs[..., 3*self.output_size:4*self.output_size],
            min=-10,
            max=10
        )

        # Add a regularization factor that encourages small scale parameters
        # when the location parameter is near zero
        zero_proximity = torch.exp(-torch.abs(loc) * 5.0)
        regularized_scale = scale * (1.0 - zero_proximity * 0.5)

        return JSUDistribution(loc=loc, scale=regularized_scale, tailweight=tailweight, skewness=skewness)

class ZeroInflatedJSULayer(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, inputs):
        # Need 5*output_size inputs: 4 for JSU + 1 for zero probability
        jsu_inputs = inputs[..., :4*self.output_size]
        zero_logits = inputs[..., 4*self.output_size:5*self.output_size]
        
        # JSU parameters (relaxed constraints)
        loc = jsu_inputs[..., :self.output_size]
        scale = torch.clamp(
            1e-3 + torch.nn.functional.softplus(jsu_inputs[..., self.output_size:2*self.output_size]), 
            min=1e-4, max=10.0
        )
        tailweight = torch.clamp(
            0.5 + torch.nn.functional.softplus(jsu_inputs[..., 2*self.output_size:3*self.output_size]),
            min=0.1, max=20.0
        )
        skewness = torch.clamp(
            jsu_inputs[..., 3*self.output_size:4*self.output_size],
            min=-10, max=10
        )
        
        # Zero inflation probability (sigmoid ensures [0,1])
        zero_prob = torch.sigmoid(zero_logits)
        
        return ZeroInflatedJSUDistribution(loc, scale, tailweight, skewness, zero_prob)

class LSJSULayer(nn.Module):
    def __init__(self, output_size, ls_decay_rate=5.0, ls_reduction_factor=0.5):
        super().__init__()
        self.output_size = output_size
        self.ls_decay_rate = ls_decay_rate  # ADD THIS
        self.ls_reduction_factor = ls_reduction_factor  # ADD THIS
    def forward(self, inputs):
        # Basic parameters 
        loc = inputs[..., :self.output_size]
        scale = torch.clamp(
            1e-3 + torch.nn.functional.softplus(inputs[..., self.output_size:2*self.output_size]), 
            min=1e-6, max=1e2
        )
        tailweight = torch.clamp(
            1.0 + torch.nn.functional.softplus(inputs[..., 2*self.output_size:3*self.output_size]),
            min=1e-6, max=1e2
        )
        skewness = torch.clamp(
            inputs[..., 3*self.output_size:4*self.output_size],
            min=-10, max=10
        )
        
        # Regularize scale for near-zero locations to handle zero-inflation
        #zero_proximity = torch.exp(-torch.abs(loc) * 5.0)
        #regularized_scale = scale * (1.0 - zero_proximity * 0.5)
        
         # Apply tunable location-scale regularization
        zero_proximity = torch.exp(-torch.abs(loc) * self.ls_decay_rate)
        regularized_scale = scale * (1.0 - zero_proximity * self.ls_reduction_factor)

        #return JSUDistribution(loc=loc, scale=regularized_scale, tailweight=tailweight, skewness=skewness)  
        return JSUDistribution(loc=loc, scale=scale, tailweight=tailweight, skewness=skewness)  

class ZeroInflatedNormalLayer(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def forward(self, inputs):
        # Need 3*output_size inputs: 2 for Normal + 1 for zero probability
        normal_inputs = inputs[..., :2*self.output_size]
        zero_logits = inputs[..., 2*self.output_size:3*self.output_size]
        
        # Normal parameters
        loc = normal_inputs[..., :self.output_size]
        scale = torch.clamp(
            1e-3 + torch.nn.functional.softplus(normal_inputs[..., self.output_size:2*self.output_size]),
            min=1e-4, max=10.0
        )
        
        # Zero inflation probability (sigmoid ensures [0,1])
        zero_prob = torch.sigmoid(zero_logits)
        
        return ZeroInflatedNormalDistribution(loc, scale, zero_prob)

class LSNormalLayer(nn.Module):
    def __init__(self, output_size, ls_decay_rate=5.0, ls_reduction_factor=0.5):
        super().__init__()
        self.output_size = output_size
        self.ls_decay_rate = ls_decay_rate  # ADD THIS
        self.ls_reduction_factor = ls_reduction_factor  # ADD THIS

    def forward(self, inputs):
        # Parse parameters
        loc = inputs[..., :self.output_size]
        scale = torch.clamp(
            1e-3 + torch.nn.functional.softplus(inputs[..., self.output_size:2*self.output_size]),
            min=1e-3, max=10.0
        )
        
        # Apply loc-scale handling
        #zero_proximity = torch.exp(-torch.abs(loc) * 5.0)
        #regularized_scale = scale * (1.0 - zero_proximity * 0.5)
        
        # Apply tunable location-scale zero-inflation handling
        zero_proximity = torch.exp(-torch.abs(loc) * self.ls_decay_rate)
        regularized_scale = scale * (1.0 - zero_proximity * self.ls_reduction_factor)

        return Normal(loc=loc, scale=regularized_scale)
    
class JFSkewTLayer(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, inputs):
        # Strong initial clamping on network outputs
        inputs = torch.clamp(inputs, min=-20.0, max=20.0)
        
        # Parse the parameters
        loc = inputs[..., :self.output_size]
        
        # More conservative scale parameter
        scale_input = inputs[..., self.output_size:2*self.output_size]
        scale = 1e-2 + torch.nn.functional.softplus(scale_input)
        scale = torch.clamp(scale, min=1e-3, max=1e2)
        
        # More conservative a parameter
        a_input = inputs[..., 2*self.output_size:3*self.output_size]
        a = 2.0 + torch.nn.functional.softplus(a_input)
        a = torch.clamp(a, min=2.0, max=30.0)
        
        # More conservative b parameter
        b_input = inputs[..., 3*self.output_size:4*self.output_size]
        b = 2.0 + torch.nn.functional.softplus(b_input)
        b = torch.clamp(b, min=2.0, max=30.0)
        
        # Check for NaNs in parameters
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(a).any() or torch.isnan(b).any():
            print("Warning: NaN detected in JFSkewTLayer parameters")
            
            # Replace NaNs with reasonable defaults
            if torch.isnan(loc).any():
                loc = torch.where(torch.isnan(loc), torch.zeros_like(loc), loc)
            if torch.isnan(scale).any():
                scale = torch.where(torch.isnan(scale), torch.ones_like(scale), scale)
            if torch.isnan(a).any():
                a = torch.where(torch.isnan(a), 3.0 * torch.ones_like(a), a)
            if torch.isnan(b).any():
                b = torch.where(torch.isnan(b), 3.0 * torch.ones_like(b), b)
        
        return JFSkewTDistribution(loc=loc, scale=scale, a=a, b=b)

class ZeroInflatedJFSkewTLayer(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
    
    def forward(self, inputs):
        # Need 5*output_size inputs: 4 for JFSkewT + 1 for zero probability
        skewt_inputs = inputs[..., :4*self.output_size]
        zero_logits = inputs[..., 4*self.output_size:5*self.output_size]
        
        # Strong initial clamping on network outputs
        skewt_inputs = torch.clamp(skewt_inputs, min=-20.0, max=20.0)
        
        # Parse the JF Skew-t parameters
        loc = skewt_inputs[..., :self.output_size]
        
        # More conservative scale parameter
        scale_input = skewt_inputs[..., self.output_size:2*self.output_size]
        scale = 1e-2 + torch.nn.functional.softplus(scale_input)
        scale = torch.clamp(scale, min=1e-3, max=1e2)
        
        # More conservative a parameter
        a_input = skewt_inputs[..., 2*self.output_size:3*self.output_size]
        a = 2.0 + torch.nn.functional.softplus(a_input)
        a = torch.clamp(a, min=2.0, max=30.0)
        
        # More conservative b parameter
        b_input = skewt_inputs[..., 3*self.output_size:4*self.output_size]
        b = 2.0 + torch.nn.functional.softplus(b_input)
        b = torch.clamp(b, min=2.0, max=30.0)
        
        # Check for NaNs in parameters
        if torch.isnan(loc).any() or torch.isnan(scale).any() or torch.isnan(a).any() or torch.isnan(b).any():
            print("Warning: NaN detected in JFSkewTLayer parameters")
            
            # Replace NaNs with reasonable defaults
            if torch.isnan(loc).any():
                loc = torch.where(torch.isnan(loc), torch.zeros_like(loc), loc)
            if torch.isnan(scale).any():
                scale = torch.where(torch.isnan(scale), torch.ones_like(scale), scale)
            if torch.isnan(a).any():
                a = torch.where(torch.isnan(a), 3.0 * torch.ones_like(a), a)
            if torch.isnan(b).any():
                b = torch.where(torch.isnan(b), 3.0 * torch.ones_like(b), b)
        
        # Zero inflation probability (sigmoid ensures [0,1])
        zero_prob = torch.sigmoid(zero_logits)
        
        return ZeroInflatedJFSkewTDistribution(loc=loc, scale=scale, a=a, b=b, zero_prob=zero_prob)

class LSJFSkewTLayer(nn.Module):
    def __init__(self, output_size, ls_decay_rate=5.0, ls_reduction_factor=0.5):
        super().__init__()
        self.output_size = output_size
        self.jf_layer = JFSkewTLayer(output_size)  # Reuse the base layer
        self.ls_decay_rate = ls_decay_rate  # ADD THIS
        self.ls_reduction_factor = ls_reduction_factor  # ADD THIS
    
    def forward(self, inputs):
        # Get base distribution
        dist = self.jf_layer(inputs)
        
        # Create zero-inflation regularization
        #zero_proximity = torch.exp(-torch.abs(dist.loc) * 5.0)
        #regularized_scale = dist.scale * (1.0 - zero_proximity * 0.5)
        
        # Apply tunable location-scale zero-inflation regularization
        zero_proximity = torch.exp(-torch.abs(dist.loc) * self.ls_decay_rate)
        regularized_scale = dist.scale * (1.0 - zero_proximity * self.ls_reduction_factor)

        # Create new distribution with regularized scale
        return JFSkewTDistribution(
            loc=dist.loc,
            scale=regularized_scale,
            a=dist.a,
            b=dist.b
        )