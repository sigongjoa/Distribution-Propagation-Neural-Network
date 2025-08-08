import torch
import torch.nn.functional as F

def stable_softplus(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    수치적으로 안정적인 softplus 함수를 구현합니다. 양수 값을 보장합니다.
    """
    return F.softplus(x) + eps

def clamp_params(params: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    분포 파라미터가 유효한 범위 내에 있도록 클램핑합니다.
    """
    return torch.clamp(params, min=eps)

def variance_regularizer(dist, coef: float) -> torch.Tensor:
    """
    분산에 대한 정규화 항을 계산합니다.
    """
    # dist는 .var() 메서드를 가지고 있어야 합니다.
    return coef * dist.var().mean()