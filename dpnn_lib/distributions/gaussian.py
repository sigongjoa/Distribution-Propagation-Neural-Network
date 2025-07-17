import torch
from .base import BaseDistribution

class GaussianDistribution(BaseDistribution):
    """
    Gaussian (정규) 분포를 나타내는 클래스.

    Args:
        mu (torch.Tensor): 분포의 평균 (mean).
        var (torch.Tensor): 분포의 분산 (variance).
    """
    def __init__(self, mu: torch.Tensor, var: torch.Tensor):
        super().__init__(params={'mu': mu, 'var': var}) # Pass tensors directly
        self.mu = mu
        self.var = var

    def sample(self):
        """
        Gaussian 분포에서 샘플을 생성합니다.

        Returns:
            torch.Tensor: 샘플링된 값.
        """
        return torch.normal(self.mu, torch.sqrt(torch.max(self.var, torch.tensor(0.0, device=self.var.device)) + 1e-6))

    def log_prob(self, x):
        """
        주어진 값에 대한 로그 확률을 계산합니다.

        Args:
            x (torch.Tensor): 관측값.

        Returns:
            torch.Tensor: log P(x).
        """
        x_tensor = x.detach().clone().float()
        coeff = -0.5 * torch.log(2 * torch.pi * self.var)
        exponent = -((x_tensor - self.mu)**2) / (2 * self.var)
        return (coeff + exponent)

    def combine(self, other: 'GaussianDistribution'):
        """
        두 Gaussian 분포를 결합합니다.

        Args:
            other (GaussianDistribution): 결합할 다른 Gaussian 분포.

        Returns:
            GaussianDistribution: 결합된 Gaussian 분포.
        """
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