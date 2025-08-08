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
    이것은 실제 구현에서 더 복잡할 수 있으며, 여기서는 플레이스홀더입니다.
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

def softmax_logit_normal(S: BaseDistribution, exclude: torch.Tensor = None) -> Dirichlet:
    """
    로짓-정규 근사를 사용하여 소프트맥스 확률을 계산합니다.
    S는 로짓 분포 (예: GaussianDiag)입니다.
    exclude는 MC/UKF로 이미 처리된 인덱스입니다.
    """
    # S의 평균을 가져옵니다.
    mu_s = S.mean()

    # 간단한 근사: 평균에 소프트맥스를 적용하고, 이를 디리클레 분포의 concentration 파라미터로 사용합니다.
    # 알파 = tau-scaled pseudo-counts
    # 여기서 tau는 softmax의 온도가 아니라, 디리클레 분포의 concentration을 조절하는 스케일링 팩터입니다.
    # 임시로 cfg.tau_init을 사용하지만, 실제로는 더 정교한 방법이 필요합니다.
    # 예를 들어, S의 분산을 사용하여 concentration을 조절할 수 있습니다.
    concentration = F.softmax(mu_s / S.var().mean().sqrt(), dim=-1) * 100.0 # 임시 스케일링
    concentration = torch.clamp(concentration, min=1e-6) # Ensure concentration is positive

    # exclude 인덱스에 해당하는 부분은 0으로 설정 (Dirichlet에서는 concentration을 0으로)
    if exclude is not None:
        # TODO: exclude 인덱스 처리 로직 개선
        pass

    return Dirichlet(concentration)
