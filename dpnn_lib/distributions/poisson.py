import numpy as np
from scipy.special import factorial
from .base import BaseDistribution

class PoissonDistribution(BaseDistribution):
    def __init__(self, lambda_param: float):
        super().__init__({'lambda': lambda_param})
        self.lambda_param = lambda_param

    def sample(self):
        return np.random.poisson(self.lambda_param)

    def log_prob(self, x):
        # Implement log probability calculation
        if not isinstance(x, int) or x < 0:
            raise ValueError("x must be a non-negative integer for Poisson log_prob.")
        if self.lambda_param <= 0:
            raise ValueError("Lambda must be positive for Poisson log_prob.")
        return x * np.log(self.lambda_param) - self.lambda_param - np.log(factorial(x))

    def combine(self, other: 'PoissonDistribution'):
        # Implement combination logic (e.g., averaging lambda)
        new_lambda = (self.lambda_param + other.lambda_param) / 2
        return PoissonDistribution(lambda_param=new_lambda)
