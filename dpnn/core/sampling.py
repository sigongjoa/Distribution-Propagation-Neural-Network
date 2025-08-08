import torch
from typing import Tuple, Callable

def sigma_points(mu: torch.Tensor, cov: torch.Tensor, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unscented Kalman Filter (UKF)를 위한 시그마 포인트를 생성합니다.
    참고: https://en.wikipedia.org/wiki/Unscented_Kalman_filter
    """
    n = mu.size(-1) # Dimension of the state
    lambda_ = alpha**2 * (n + kappa) - n

    # Weights
    Wm = torch.full((2 * n + 1,), 0.5 / (n + lambda_), dtype=mu.dtype, device=mu.device)
    Wc = torch.full((2 * n + 1,), 0.5 / (n + lambda_), dtype=mu.dtype, device=mu.device)
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    # Sigma points
    # S = sqrt((n + lambda_) * cov)
    try:
        S = torch.linalg.cholesky((n + lambda_) * cov)
    except RuntimeError:
        # Add a small diagonal to covariance for numerical stability if not positive definite
        cov_stable = cov + torch.eye(n, dtype=cov.dtype, device=cov.device) * 1e-6
        S = torch.linalg.cholesky((n + lambda_) * cov_stable)

    X = torch.zeros((2 * n + 1, n), dtype=mu.dtype, device=mu.device)
    X[0] = mu
    X[1:n+1] = mu + S
    X[n+1:] = mu - S

    return X, Wm, Wc

class GaussianApprox:
    """
    UKF 전파 후 가우시안 근사를 위한 간단한 클래스.
    """
    def __init__(self, mu: torch.Tensor, cov: torch.Tensor):
        self._mu = mu
        self._cov = cov

    def mean(self) -> torch.Tensor:
        return self._mu

    def cov(self) -> torch.Tensor:
        return self._cov

def propagate_ukf(dist, fn: Callable) -> GaussianApprox:
    """
    Unscented Kalman Filter (UKF)를 사용하여 분포를 비선형 함수를 통해 전파합니다.
    """
    mu = dist.mean()
    cov = dist.cov() # Assuming dist has a cov() method

    X, Wm, Wc = sigma_points(mu, cov)
    
    # Apply the function to each sigma point
    Y = torch.stack([fn(x) for x in X])

    # Recover mean and covariance of the transformed distribution
    mu_y = torch.sum(Wm.unsqueeze(-1) * Y, dim=0)
    cov_y = torch.sum(Wc.unsqueeze(-1).unsqueeze(-1) * (Y - mu_y).unsqueeze(1) @ (Y - mu_y).unsqueeze(2), dim=0)

    return GaussianApprox(mu_y, cov_y)
