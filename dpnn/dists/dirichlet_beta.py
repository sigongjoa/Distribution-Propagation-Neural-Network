import torch
from typing import Tuple, Optional
from ..core.distribution import BaseDistribution

class Dirichlet(BaseDistribution):
    """
    디리클레 분포를 나타냅니다.
    """
    def __init__(self, concentration: torch.Tensor):
        self.concentration = concentration
        self.event_shape = concentration.shape[-1:]
        self.batch_shape = concentration.shape[:-1]

    def mean(self) -> torch.Tensor:
        return self.concentration / torch.sum(self.concentration, dim=-1, keepdim=True)

    def var(self) -> torch.Tensor:
        alpha0 = torch.sum(self.concentration, dim=-1, keepdim=True)
        return (self.concentration * (alpha0 - self.concentration)) / (alpha0**2 * (alpha0 + 1))

    def entropy(self) -> torch.Tensor:
        # Placeholder for Dirichlet entropy
        raise NotImplementedError("Entropy not implemented for Dirichlet distribution.")

    def sample(self, n: Optional[int] = None) -> torch.Tensor:
        base_dist = torch.distributions.Dirichlet(self.concentration)
        if n is None:
            return base_dist.sample()
        else:
            return base_dist.sample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        base_dist = torch.distributions.Dirichlet(self.concentration)
        return base_dist.log_prob(x)

    def kl(self, other: 'BaseDistribution') -> torch.Tensor:
        # Placeholder for KL divergence
        raise NotImplementedError("KL divergence not implemented for Dirichlet distribution.")

    def to_tensor(self) -> torch.Tensor:
        return self.mean()
