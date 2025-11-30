import torch
import torch.nn as nn
from torch.distributions import Distribution, constraints, Normal
from torch.distributions.utils import broadcast_all
import math
import numpy as np
from scipy import stats

class ZeroInflatedNormalDistribution(Distribution):
    """True zero-inflated Normal with discrete point mass at zero"""
    
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'zero_prob': constraints.unit_interval
    }
    support = constraints.real
    has_rsample = True
    
    def __init__(self, loc, scale, zero_prob, validate_args=None):
        self.normal = Normal(loc, scale)
        self.zero_prob = zero_prob
        batch_shape = zero_prob.shape
        super().__init__(batch_shape, validate_args=False)
    
    def log_prob(self, value):
        # Point mass at zero
        zero_mask = (value.abs() < 1e-3).float()
        
        # Log probabilities
        zero_log_prob = torch.log(self.zero_prob + 1e-8)
        nonzero_log_prob = torch.log(1 - self.zero_prob + 1e-8) + self.normal.log_prob(value)
        
        # Mixture: p(x=0)*I(x=0) + p(x≠0)*p_Normal(x)*I(x≠0)
        return zero_mask * zero_log_prob + (1 - zero_mask) * nonzero_log_prob
    
    def rsample(self, sample_shape=torch.Size()):
        # Sample from Bernoulli for zero/nonzero
        shape = self._extended_shape(sample_shape)
        bernoulli_samples = torch.bernoulli(self.zero_prob.expand(shape))
        
        # Sample from Normal for non-zero component
        normal_samples = self.normal.rsample(sample_shape)
        
        # Mixture: zeros where Bernoulli=1, Normal samples where Bernoulli=0
        return normal_samples * (1 - bernoulli_samples)

class JSUDistribution(Distribution):
    """
    Johnson's SU Distribution implementation for PyTorch.
    
    Parameters:
    -----------
    loc : torch.Tensor
        Location parameter
    scale : torch.Tensor
        Scale parameter (positive)
    tailweight : torch.Tensor
        Tail weight parameter (positive)
    skewness : torch.Tensor
        Skewness parameter
    validate_args : bool, optional
        Whether to validate distribution parameters
    """
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'tailweight': constraints.positive,
        'skewness': constraints.real
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, tailweight, skewness, validate_args=None):
        self.loc, self.scale, self.tailweight, self.skewness = broadcast_all(
            loc, scale, tailweight, skewness
        )
        batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=False)
        
        # Validate parameters
        scale = torch.clamp(scale, min=1e-6)
        tailweight = torch.clamp(tailweight, min=1e-6)
        
        self.loc = loc
        self.scale = scale
        self.tailweight = tailweight
        self.skewness = skewness

    def rsample(self, sample_shape=torch.Size()):
        """
        Generate random samples from the distribution.
        
        Parameters:
        -----------
        sample_shape : torch.Size
            Shape of the samples to generate
            
        Returns:
        --------
        torch.Tensor
            Samples from the JSU distribution
        """
        shape = self._extended_shape(sample_shape)
        eps = torch.randn(shape, device=self.loc.device)
        u = torch.sinh((eps - self.skewness) / self.tailweight)
        return self.loc + self.scale * u

    def log_prob(self, value):
        """
        Calculate log probability density at given value(s).
        
        Parameters:
        -----------
        value : torch.Tensor
            Points at which to evaluate log probability
            
        Returns:
        --------
        torch.Tensor
            Log probability density
        """
        if self._validate_args:
            self._validate_sample(value)
            
        z = (value - self.loc) / self.scale
        w = torch.asinh(z)
        r = (w - self.skewness) / self.tailweight
        
        logprob = (torch.log(1.0 / self.tailweight)
                  - torch.log(self.scale)
                  - 0.5 * torch.log(1.0 + z * z)
                  - 0.5 * r * r
                  - 0.5 * torch.log(2.0 * torch.tensor(torch.pi, device=self.loc.device)))
        
        return torch.clamp(logprob, min=-100)
    
class ZeroInflatedJSUDistribution(Distribution):
    """True zero-inflated JSU with discrete point mass at zero"""
    
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'tailweight': constraints.positive,
        'skewness': constraints.real,
        'zero_prob': constraints.unit_interval
    }
    support = constraints.real
    has_rsample = True
    
    def __init__(self, loc, scale, tailweight, skewness, zero_prob, validate_args=None):
        self.jsu = JSUDistribution(loc, scale, tailweight, skewness)
        self.zero_prob = zero_prob
        batch_shape = zero_prob.shape
        super().__init__(batch_shape, validate_args=False)
    
    def log_prob(self, value):
        # Point mass at zero
        zero_mask = (value.abs() < 1e-3).float()
        
        # Log probabilities
        zero_log_prob = torch.log(self.zero_prob + 1e-8)
        nonzero_log_prob = torch.log(1 - self.zero_prob + 1e-8) + self.jsu.log_prob(value)
        
        # Mixture: p(x=0)*I(x=0) + p(x≠0)*p_JSU(x)*I(x≠0)
        return zero_mask * zero_log_prob + (1 - zero_mask) * nonzero_log_prob
    
    def rsample(self, sample_shape=torch.Size()):
        # Sample from Bernoulli for zero/nonzero
        shape = self._extended_shape(sample_shape)
        bernoulli_samples = torch.bernoulli(self.zero_prob.expand(shape))
        
        # Sample from JSU for non-zero component
        jsu_samples = self.jsu.rsample(sample_shape)
        
        # Mixture: zeros where Bernoulli=1, JSU samples where Bernoulli=0
        return jsu_samples * (1 - bernoulli_samples)

    
class JFSkewTDistribution(Distribution):
    """
    Jones and Faddy skewed t-Distribution with enhanced numerical stability.
    """
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'a': constraints.positive,
        'b': constraints.positive
    }
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        # Apply stricter constraints before broadcasting to prevent NaNs
        scale = torch.clamp(scale, min=1e-3, max=1e2)
        a = torch.clamp(a, min=2.0, max=30.0)  # Stricter bounds for a
        b = torch.clamp(b, min=2.0, max=30.0)  # Stricter bounds for b
        
        self.loc, self.scale, self.a, self.b = broadcast_all(
            loc, scale, a, b
        )
        batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=False)
        
    def rsample(self, sample_shape=torch.Size()):
        """Generate random samples with proper error handling."""
        shape = self._extended_shape(sample_shape)
        
        try:
            # Generate standard Normal samples  
            z = torch.randn(shape, device=self.loc.device)
            
            # Create t-distribution samples with df = 2*(a+b)
            df = 2.0 * (self.a + self.b)
            t_denom = torch.sqrt(torch.clamp(df - 2, min=0.1) * 
                                torch.rand(shape, device=self.loc.device) / df)
            
            # Create t-distributed samples
            t_samples = z / torch.clamp(t_denom, min=1e-3)
            
            # Apply skewness transformation with proper broadcasting
            with torch.no_grad():
                # Ensure skew_factor has the same shape as t_samples for proper broadcasting
                skew_factor = torch.sqrt(self.b / self.a)
                
                # Broadcast skew_factor to match t_samples shape
                skew_factor = skew_factor.expand_as(t_samples)
                
                # Apply skewness transformation element-wise
                mask = t_samples < 0
                t_samples = torch.where(
                    mask,
                    t_samples / skew_factor,  # Negative values: divide by skew_factor
                    t_samples * skew_factor   # Positive values: multiply by skew_factor
                )
                
                # Apply location and scale transformation
                samples = self.loc + self.scale * t_samples
            
            # Check for NaNs and replace with safe values
            if torch.isnan(samples).any():
                nan_mask = torch.isnan(samples)
                # Broadcast loc to match samples shape for replacement
                loc_expanded = self.loc.expand_as(samples)
                samples = torch.where(nan_mask, loc_expanded, samples)
            
            return samples
            
        except Exception as e:
            # Fallback to normal distribution if everything else fails
            print(f"Warning: JFSkewT sampling failed with error {e}, falling back to normal distribution")
            return self.loc + self.scale * torch.randn(shape, device=self.loc.device)
        
    def log_prob(self, value):
        """Calculate log probability with enhanced numerical stability."""
        if self._validate_args:
            self._validate_sample(value)
        
        # Use safer parameter constraints
        a = torch.clamp(self.a, min=2.0, max=30.0)
        b = torch.clamp(self.b, min=2.0, max=30.0)
        scale = torch.clamp(self.scale, min=1e-3, max=1e2)
        
        # Standardize value with bounds
        z = torch.clamp((value - self.loc) / scale, min=-50.0, max=50.0)
        
        # Add small epsilon to denominators
        epsilon = 1e-8
        
        # Calculate JF skewed t log-PDF using the formula:
        # log(C) + (a+1/2)*log(1+z/sqrt(a+b+z²)) + (b+1/2)*log(1-z/sqrt(a+b+z²)) - log(scale)
        # where C is the normalization constant
        
        # Calculate sqrt term safely
        term = a + b + z**2 + epsilon
        sqrt_term = torch.sqrt(term)
        
        # Calculate terms inside logs with clamping to prevent log(0)
        term1 = torch.clamp(1 + z/sqrt_term, min=epsilon)
        term2 = torch.clamp(1 - z/sqrt_term, min=epsilon)
        
        # Log normalization constant
        log_const = ((a + b - 1) * torch.log(torch.tensor(2.0, device=z.device)) + 
                   torch.lgamma(a + b) - torch.lgamma(a) - torch.lgamma(b) +
                   0.5 * torch.log(a + b))
        
        # Calculate log terms
        log_term1 = (a + 0.5) * torch.log(term1)
        log_term2 = (b + 0.5) * torch.log(term2)
        log_scale = torch.log(scale)
        
        # Full log PDF
        log_pdf = log_const + log_term1 + log_term2 - log_scale
        
        # Check for NaNs and replace with safe values
        if torch.isnan(log_pdf).any():
            # Compute standard normal log PDF as fallback
            std_normal_log_pdf = -0.5 * z**2 - 0.5 * torch.log(torch.tensor(2 * math.pi, device=z.device)) - log_scale
            log_pdf = torch.where(torch.isnan(log_pdf), std_normal_log_pdf, log_pdf)
        
        return torch.clamp(log_pdf, min=-100.0, max=100.0)

class ZeroInflatedJFSkewTDistribution(Distribution):
    """True zero-inflated JF Skew-t with discrete point mass at zero"""
    
    arg_constraints = {
        'loc': constraints.real,
        'scale': constraints.positive,
        'a': constraints.positive,
        'b': constraints.positive,
        'zero_prob': constraints.unit_interval
    }
    support = constraints.real
    has_rsample = True
    
    def __init__(self, loc, scale, a, b, zero_prob, validate_args=None):
        self.skewt = JFSkewTDistribution(loc, scale, a, b)
        self.zero_prob = zero_prob
        batch_shape = zero_prob.shape
        super().__init__(batch_shape, validate_args=False)
    
    def log_prob(self, value):
        # Point mass at zero
        zero_mask = (value.abs() < 1e-3).float()
        
        # Log probabilities
        zero_log_prob = torch.log(self.zero_prob + 1e-8)
        nonzero_log_prob = torch.log(1 - self.zero_prob + 1e-8) + self.skewt.log_prob(value)
        
        # Mixture: p(x=0)*I(x=0) + p(x≠0)*p_SkewT(x)*I(x≠0)
        return zero_mask * zero_log_prob + (1 - zero_mask) * nonzero_log_prob
    
    def rsample(self, sample_shape=torch.Size()):
        # Sample from Bernoulli for zero/nonzero
        shape = self._extended_shape(sample_shape)
        bernoulli_samples = torch.bernoulli(self.zero_prob.expand(shape))
        
        # Sample from Skew-t for non-zero component
        skewt_samples = self.skewt.rsample(sample_shape)
        
        # Mixture: zeros where Bernoulli=1, Skew-t samples where Bernoulli=0
        return skewt_samples * (1 - bernoulli_samples)