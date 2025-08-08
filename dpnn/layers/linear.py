import torch.nn as nn
import torch
import math
from ..core.distribution import BaseDistribution
from ..dists.gaussian import GaussianDiag

class DistLinear(nn.Module):
    """
    분포를 입력으로 받아 분포를 출력하는 선형 레이어입니다.
    현재는 GaussianDiag 분포에 대해서만 구현되어 있습니다.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, dist: BaseDistribution) -> BaseDistribution:
        if isinstance(dist, GaussianDiag):
            return dist.affine(self.weight, self.bias)
        else:
            raise NotImplementedError(f"DistLinear not implemented for distribution type: {type(dist)}")