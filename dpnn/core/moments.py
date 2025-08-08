import torch
import torch.nn.functional as F
from .config import DistConfig, DistMode
from .distribution import BaseDistribution
from torch.distributions import Normal
from ..dists.dirichlet_beta import Dirichlet # Import Dirichlet

# dpnn/core/moments.py
def relu_moment(mu: torch.Tensor, var: torch.Tensor):
    """
    Gaussian 분포에 대한 ReLU의 기댓값과 분산을 근사합니다.
    참고: https://www.cs.toronto.edu/~duvenaud/papers/uncertainty-relu.pdf
    """
    std = torch.sqrt(var + 1e-6) # Add epsilon for numerical stability
    alpha = mu / std

    # E[ReLU(X)]
    # phi(alpha) = PDF of standard normal at alpha
    # Phi(alpha) = CDF of standard normal at alpha
    normal = torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
    phi_alpha = torch.exp(normal.log_prob(alpha))
    Phi_alpha = normal.cdf(alpha)

    mean_relu = std * phi_alpha + mu * Phi_alpha

    # E[ReLU(X)^2]
    # E[X^2] = mu^2 + var
    # E[X^2 * I(X>0)] = (mu^2 + var) * Phi(alpha) + mu * std * phi(alpha)
    mean_sq_relu = (mu**2 + var) * Phi_alpha + mu * std * phi_alpha

    var_relu = mean_sq_relu - mean_relu**2

    return mean_relu, var_relu

def delta_method(mu: torch.Tensor, var: torch.Tensor, f, fprime):
    """
    1차 델타 메서드를 사용하여 함수 f(X)의 기댓값과 분산을 근사합니다.
    """
    mean_f_x = f(mu)
    var_f_x = (fprime(mu)**2) * var
    return mean_f_x, var_f_x

def approx_curvature(dist, nonlin) -> torch.Tensor:
    """
    비선형 함수의 곡률을 근사합니다. (예: f''(μ)의 규모)
    이것은 실제 구현에서 더 복잡할 수 있으며, 여기서는 플레이스홀더입니다。
    """
    # TODO: 실제 곡률 근사 로직 구현
    return torch.tensor(0.0) # Placeholder

def approx_entropy(dist: BaseDistribution) -> torch.Tensor:
    """
    분포의 정규화된 엔트로피를 근사합니다.
    """
    if hasattr(dist, 'var'):
        v = dist.var()
        # Assuming GaussianDiag for now, entropy = 0.5 * log(2 * pi * e * var)
        # Normalizing by the dimension of the event space
        # For a single scalar, it's 0.5 * log(2 * pi * e * var)
        # For a vector, it's sum(0.5 * log(2 * pi * e * var_i))
        # Here, we return the mean of the per-element entropy for simplicity
        return 0.5 * torch.log(2 * torch.pi * torch.e * v + 1e-6).mean()
    return torch.tensor(0.0, device=dist.mean().device if hasattr(dist, 'mean') else None)

def decide_mode(dist, nonlin, cfg: DistConfig) -> DistMode:
    """
    분포와 비선형성에 따라 전파 모드를 결정합니다.
    """
    H = approx_curvature(dist, nonlin)   # 예: f''(μ) 규모로 근사
    ent = approx_entropy(dist)           # 정규화 엔트로피
    if ent > cfg.entropy_threshold or H > cfg.curvature_threshold:
        # 어려운 구간
        if cfg.use_sigma_points: return DistMode.UKF
        if cfg.mc_k_max > 0:     return DistMode.MC
    # 쉬운 구간
    return DistMode.MOMENT if H < 0.3 else DistMode.DELTA

def softmax_logit_normal(S: BaseDistribution, exclude: torch.Tensor | None = None, eps: float = 1e-8) -> Dirichlet:
    """
    로짓-정규 근사를 사용하여 소프트맥스 확률을 계산합니다.
    S는 로짓 분포 (예: GaussianDiag)입니다.
    exclude는 MC/UKF로 이미 처리된 인덱스입니다.
    """
    mu = S.mean()                                 # (B,H,L)
    var = S.var().clamp_min(1e-6)                 # (B,H,L)
    p = F.softmax(mu, dim=-1)                     # (B,H,L)

    # 분산이 작을수록 “확신 높음” → α0 크게. 너무 크지 않도록 clamp.
    inv_scale = (1.0 / var.mean(dim=-1, keepdim=True).sqrt()).clamp(1.0, 10.0)  # (B,H,1)
    alpha = p * (inv_scale * 50.0)               # α0 대략 50~500 사이

    if exclude is not None:
        # exclude: (B,H,K_top), 마지막 차원이 L의 인덱스
        # α에 직접 scatter로 ε 부여
        # scatter는 업데이트 값을 alpha와 같은 dtype/shape로 요구하므로, 동일 shape의 fill 텐서 준비
        B, H, L = alpha.shape
        if exclude.dim() != 3 or exclude.size(0) != B or exclude.size(1) != H:
            raise ValueError(f"exclude shape must be (B,H,K), got {tuple(exclude.shape)} vs alpha {(B,H,L)}")
        K = exclude.size(-1)
        # 업데이트 값: (B,H,K) 텐서, 모두 eps
        updates = torch.full((B, H, K), eps, dtype=alpha.dtype, device=alpha.device)
        alpha = alpha.scatter(dim=-1, index=exclude, src=updates)

    # α0 재스케일 (안정화)
    alpha0 = alpha.sum(dim=-1, keepdim=True).clamp(1.0, 200.0)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12) * alpha0
    alpha = alpha.clamp_min(eps)
    return Dirichlet(alpha)