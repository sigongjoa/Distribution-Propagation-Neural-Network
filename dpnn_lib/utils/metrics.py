import torch
import torch.nn.functional as F
from typing import List
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.distributions.gaussian import GaussianDistribution

class Metrics:
    """
    모델 성능 평가를 위한 다양한 메트릭을 제공하는 클래스입니다.
    """
    @staticmethod
    def kl_divergence(p: BaseDistribution, q: BaseDistribution):
        """
        두 분포 간의 KL 발산을 계산합니다.

        Args:
            p (BaseDistribution): 첫 번째 분포.
            q (BaseDistribution): 두 번째 분포.

        Returns:
            torch.Tensor: 계산된 KL 발산 값.
        """
        # Implement KL divergence calculation
        from dpnn_lib.distributions.gaussian import GaussianDistribution

        if not isinstance(p, GaussianDistribution) or not isinstance(q, GaussianDistribution):
            raise TypeError("KL divergence for now only supported for GaussianDistribution.")

        # KL divergence between two Gaussian distributions N(mu_p, var_p) and N(mu_q, var_q)
        # KL(P || Q) = 0.5 * (log(var_q / var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1)
        if p.var <= 0 or q.var <= 0:
            raise ValueError("Variances must be positive for KL divergence calculation.")

        # Convert to torch tensors for autograd
        mu_p, var_p = p.mu, p.var
        mu_q, var_q = q.mu, q.var

        kl_div = 0.5 * (torch.log(var_q / var_p) + (var_p + (mu_p - mu_q)**2) / var_q - 1)
        return kl_div

    @staticmethod
    def perplexity(logits, labels):
        """
        주어진 로짓과 레이블에 대한 Perplexity를 계산합니다.

        Args:
            logits (torch.Tensor): 모델의 출력 로짓.
            labels (torch.Tensor): 실제 레이블.

        Returns:
            torch.Tensor: 계산된 Perplexity 값.
        """
        cross_entropy = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return torch.exp(cross_entropy)

    @staticmethod
    def accuracy(logits, labels):
        """
        주어진 로짓과 레이블에 대한 정확도를 계산합니다.

        Args:
            logits (torch.Tensor): 모델의 출력 로짓.
            labels (torch.Tensor): 실제 레이블.

        Returns:
            torch.Tensor: 계산된 정확도 값.
        """
        predictions = torch.argmax(logits, dim=-1)
        return (predictions == labels).float().mean()

    @staticmethod
    def visualize_distribution(distribution: BaseDistribution):
        """
        분포를 시각화합니다 (구현되지 않음).

        Args:
            distribution (BaseDistribution): 시각화할 분포.
        """
        # Implement distribution visualization (e.g., using matplotlib)
        pass
