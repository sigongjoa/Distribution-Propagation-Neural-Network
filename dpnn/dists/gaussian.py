import torch
from typing import Tuple, Optional
from ..core.distribution import BaseDistribution
from ..core.numerics import stable_softplus

class GaussianDiag(BaseDistribution):
    """
    대각 공분산 행렬을 가진 가우시안 분포를 나타냅니다.
    """
    def __init__(self, loc: torch.Tensor, log_scale: torch.Tensor):
        self.loc = loc
        self.log_scale = log_scale
        self.scale = stable_softplus(log_scale)
        self.event_shape = loc.shape[-1:]
        self.batch_shape = loc.shape[:-1]

    @classmethod
    def from_tensor(cls, x: torch.Tensor, init_std: float = 0.1):
        """
        텐서로부터 GaussianDiag 분포를 생성합니다. 초기 분산은 작게 설정됩니다.
        """
        loc = x
        log_scale = torch.log(torch.full_like(x, init_std))
        return cls(loc, log_scale)

    def mean(self) -> torch.Tensor:
        return self.loc

    def var(self) -> torch.Tensor:
        return self.scale**2

    def cov(self) -> torch.Tensor:
        """
        공분산 행렬을 반환합니다 (대각 행렬).
        """
        return torch.diag_embed(self.var())

    def entropy(self) -> torch.Tensor:
        # H(X) = 0.5 * log((2 * pi * e)^k * |Sigma|)
        # For diagonal covariance, |Sigma| = product(var)
        k = self.event_shape[0]
        return 0.5 * torch.sum(torch.log(2 * torch.pi * torch.e * self.var()), dim=-1)

    def sample(self, n: Optional[int] = None) -> torch.Tensor:
        base_dist = torch.distributions.Normal(self.loc, self.scale)
        if n is None:
            return base_dist.sample()
        else:
            return base_dist.sample((n,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        base_dist = torch.distributions.Normal(self.loc, self.scale)
        return torch.sum(base_dist.log_prob(x), dim=-1)

    def kl(self, other: 'BaseDistribution') -> torch.Tensor:
        if not isinstance(other, GaussianDiag):
            raise NotImplementedError("KL divergence only implemented for GaussianDiag to GaussianDiag.")
        
        # KL(P || Q) for P=N(mu_p, sigma_p^2), Q=N(mu_q, sigma_q^2)
        # 0.5 * (log(sigma_q^2 / sigma_p^2) + (sigma_p^2 + (mu_p - mu_q)^2) / sigma_q^2 - 1)
        kl_div = 0.5 * (torch.log(other.var() / self.var()) +
                        (self.var() + (self.loc - other.loc)**2) / other.var() - 1)
        return torch.sum(kl_div, dim=-1)

    def to_tensor(self) -> torch.Tensor:
        return self.mean()

    def affine(self, W: torch.Tensor, b: torch.Tensor) -> 'GaussianDiag':
        """
        GaussianDiag 분포에 대한 아핀 변환을 수행합니다.
        Y = WX + b
        E[Y] = W E[X] + b
        Var[Y] = W^2 Var[X] (element-wise for diagonal covariance)
        """
        new_loc = torch.matmul(self.loc, W.T) + b
        # Assuming W is a linear layer weight matrix, so W.shape = (out_features, in_features)
        # For diagonal covariance, we need element-wise square of W for variance propagation
        # This assumes W is applied to each element independently, which is not strictly true for matrix multiplication.
        # For a proper affine transformation with diagonal covariance, W should be a diagonal matrix or we need to handle full covariance.
        # For now, let's assume W is a simple scaling or we are only interested in diagonal elements of the resulting covariance.
        # A more rigorous approach for general W would involve: new_cov = W @ self.cov() @ W.T
        # If W is (out_features, in_features) and self.var() is (batch_shape, in_features),
        # then new_var should be (batch_shape, out_features).
        # This is a simplification for diagonal Gaussian.
        
        # For a linear layer (matrix multiplication), the variance propagation is more complex.
        # If Y = WX, then Cov(Y) = W Cov(X) W^T.
        # If Cov(X) is diagonal (diag(var_x)), then Cov(Y)_ij = sum_k W_ik W_jk var_xk.
        # The diagonal elements are Cov(Y)_ii = sum_k W_ik^2 var_xk.
        # So, new_var_i = sum_k W_ik^2 * var_xk
        
        # Assuming W is (out_features, in_features)
        # self.var() is (..., in_features)
        # new_var should be (..., out_features)
        new_var = torch.matmul(self.var(), W.T**2) # Element-wise square of W.T
        
        new_log_scale = 0.5 * torch.log(new_var + 1e-6) # Add epsilon for stability
        return GaussianDiag(new_loc, new_log_scale)
