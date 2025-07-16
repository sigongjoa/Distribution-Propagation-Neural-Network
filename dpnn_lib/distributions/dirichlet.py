import numpy as np
from scipy.special import gammaln
from .base import BaseDistribution

class DirichletDistribution(BaseDistribution):
    def __init__(self, alpha_vector: list):
        super().__init__({'alpha': alpha_vector})
        self.alpha_vector = alpha_vector

    def sample(self):
        return np.random.dirichlet(self.alpha_vector)

    def log_prob(self, x):
        # Implement log probability calculation
        # x is expected to be a probability vector (sums to 1)
        # alpha_vector elements must be > 0
        if not np.isclose(np.sum(x), 1.0):
            raise ValueError("x must be a probability vector (sum to 1).")
        if any(a <= 0 for a in self.alpha_vector):
            raise ValueError("Alpha vector elements must be positive for log_prob.")

        alpha_sum = np.sum(self.alpha_vector)
        log_beta = np.sum(np.array([gammaln(a) for a in self.alpha_vector])) - gammaln(alpha_sum)

        log_prob_val = np.sum([(a - 1) * np.log(xi) for a, xi in zip(self.alpha_vector, x)]) - log_beta
        return log_prob_val

    def combine(self, other: 'DirichletDistribution'):
        # Implement combination logic (e.g., parameter addition)
        # This assumes alpha_vector elements can be directly added
        new_alpha = [a + b for a, b in zip(self.alpha_vector, other.alpha_vector)]
        return DirichletDistribution(alpha_vector=new_alpha)
