import torch # Import torch
import numpy as np
from scipy.special import gammaln
from .base import BaseDistribution

class DirichletDistribution(BaseDistribution):
    """
    Dirichlet 분포를 나타내는 클래스.

    Args:
        alpha_vector (torch.Tensor): 분포의 alpha 벡터
    """
    def __init__(self, alpha_vector: torch.Tensor):
        super().__init__(params={'alpha': alpha_vector})
        self.alpha_vector = alpha_vector

    def sample(self):
        """
        Dirichlet 분포에서 샘플을 생성합니다.

        Returns:
            torch.Tensor: 샘플링된 값
        """
        # Ensure alpha_vector is on CPU for numpy operation
        return torch.tensor(np.random.dirichlet(self.alpha_vector.cpu().numpy()), device=self.alpha_vector.device, dtype=self.alpha_vector.dtype)

    def log_prob(self, x):
        """
        주어진 값에 대한 로그 확률을 계산합니다.

        Args:
            x (torch.Tensor): 관측값 (확률 벡터, 합이 1)

        Returns:
            torch.Tensor: log P(x)
        """
        # Implement log probability calculation
        # x is expected to be a probability vector (sums to 1)
        # alpha_vector elements must be > 0
        if not torch.isclose(torch.sum(x), torch.tensor(1.0, device=x.device)): # Use torch.isclose
            raise ValueError("x must be a probability vector (sum to 1).")
        if torch.any(self.alpha_vector <= 0): # Use torch.any
            raise ValueError("Alpha vector elements must be positive for log_prob.")

        alpha_sum = torch.sum(self.alpha_vector)
        log_beta = torch.sum(torch.lgamma(self.alpha_vector)) - torch.lgamma(alpha_sum) # Use torch.lgamma

        log_prob_val = torch.sum((self.alpha_vector - 1) * torch.log(x)) - log_beta
        return log_prob_val

    def combine(self, other: 'DirichletDistribution'):
        """
        두 Dirichlet 분포를 결합합니다.

        Args:
            other (DirichletDistribution): 결합할 다른 Dirichlet 분포

        Returns:
            DirichletDistribution: 결합된 Dirichlet 분포
        """
        # Implement combination logic (e.g., parameter addition)
        # This assumes alpha_vector elements can be directly added
        new_alpha = self.alpha_vector + other.alpha_vector # Direct tensor addition
        return DirichletDistribution(alpha_vector=new_alpha)
