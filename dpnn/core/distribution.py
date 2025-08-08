import torch.nn as nn
from typing import Tuple, Optional

class BaseDistribution:
    event_shape: Tuple[int, ...]
    batch_shape: Tuple[int, ...]

    def mean(self) -> torch.Tensor: ...
    def var(self) -> torch.Tensor: ...
    def entropy(self) -> torch.Tensor: ...
    def sample(self, n: Optional[int] = None) -> torch.Tensor: ...
    def log_prob(self, x: torch.Tensor) -> torch.Tensor: ...
    def kl(self, other: 'BaseDistribution') -> torch.Tensor: ...
    def to_tensor(self) -> torch.Tensor: ...  # (optional) 대표값 반환 (mean 등)

def affine(self, W: torch.Tensor, b: torch.Tensor):
    # 이 함수는 BaseDistribution의 메서드로 포함되거나, 특정 분포 클래스에서 구현될 것입니다.
    # 여기서는 일반적인 형태만 제시합니다.
    # μ = W @ self.loc + b
    # if diag: σ2 = (W**2) @ self.var()
    # else: Σ = W @ self.cov() @ W.T
    # return self.__class__.from_moments(μ, σ2 or Σ)
    raise NotImplementedError("Affine transformation not implemented for this base distribution.")
