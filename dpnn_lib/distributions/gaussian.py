import torch
from .base import BaseDistribution

class GaussianDistribution(BaseDistribution):
    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        super().__init__(params={'mu': mu, 'var': var}) # Pass tensors directly
        self.mu = mu
        self.var = torch.clamp(var, min=1e-6)

    def sample(self):
        return torch.normal(self.mu, torch.sqrt(self.var))

    def log_prob(self, x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        coeff = -0.5 * torch.log(2 * torch.pi * self.var)
        exponent = -((x_tensor - self.mu)**2) / (2 * self.var)
        return (coeff + exponent)

    def combine(self, other: 'GaussianDistribution'):
        # Combine two Gaussian distributions using precision-weighted averaging
        # This is more stable than simple averaging of mu and var

        # Convert to precision (lambda) and precision-weighted mean (nu)
        lambda_self = 1.0 / self.var
        nu_self = self.mu / self.var

        lambda_other = 1.0 / other.var
        nu_other = other.mu / other.var

        # Combine precision and precision-weighted mean
        combined_lambda = lambda_self + lambda_other
        combined_nu = nu_self + nu_other

        # Convert back to mu and var
        new_mu = combined_nu / combined_lambda
        new_var = 1.0 / combined_lambda

        return GaussianDistribution(mu=new_mu, var=new_var)
