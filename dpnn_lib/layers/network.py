import torch.nn as nn
from typing import List
from dpnn_lib.layers.layer import DistributionLayer
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.distributions.base import BaseDistribution

class DistributionNetwork(nn.Module):
    """
    분포 레이어들의 시퀀스로 구성된 신경망입니다.

    Args:
        layers (List[DistributionLayer]): 네트워크를 구성하는 분포 레이어의 리스트.
    """
    def __init__(self, layers: List[DistributionLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, initial_distributions: List[BaseDistribution]) -> List[BaseDistribution]:
        """
        입력 분포를 각 레이어에 순차적으로 전파합니다.

        Args:
            initial_distributions (List[BaseDistribution]): 네트워크에 입력될 초기 분포의 리스트.

        Returns:
            List[BaseDistribution]: 네트워크의 최종 출력 분포 리스트.
        """
        # Propagate through layers
        current_dists = initial_distributions
        for i, layer in enumerate(self.layers):
            # Pass the current distributions to the layer's forward method
            current_dists = layer.forward(current_dists)
        return current_dists # Return the final output distributions

    def get_output(self):
        """
        이 메소드는 더 이상 사용되지 않습니다. forward()를 사용하여 출력 분포를 얻으세요.
        """
        # This method will no longer be needed in its current form
        # The forward method will return the output
        raise NotImplementedError("get_output() is deprecated. Use forward() to get output distributions.")