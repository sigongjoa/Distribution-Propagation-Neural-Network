import torch.optim as optim
from dpnn_lib.layers.network import DistributionNetwork
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.utils.metrics import Metrics
from typing import List

class DistributionOptimizer:
    """
    분포 네트워크를 위한 옵티마이저입니다.

    Args:
        network (DistributionNetwork): 최적화할 분포 네트워크.
        lr (float, optional): 학습률 (기본값: 0.001).
    """
    def __init__(self, network: DistributionNetwork, lr: float = 0.001):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def loss_fn(self, output_distributions: List[BaseDistribution], target_distributions: List[BaseDistribution]):
        """
        출력 분포와 대상 분포 간의 손실(KL 발산)을 계산합니다.

        Args:
            output_distributions (List[BaseDistribution]): 네트워크의 출력 분포 리스트.
            target_distributions (List[BaseDistribution]): 대상 분포 리스트.

        Returns:
            float: 계산된 총 손실.
        """
        if len(output_distributions) != len(target_distributions):
            raise ValueError("Output and target distribution lists must have the same length.")

        total_loss = 0.0
        for output_dist, target_dist in zip(output_distributions, target_distributions):
            # Ensure KL divergence returns a tensor for backprop
            kl_div = Metrics.kl_divergence(output_dist, target_dist)
            total_loss += kl_div
        return total_loss

    def step(self): # Removed 'loss' argument
        """
        옵티마이저의 파라미터를 업데이트하고 그래디언트를 초기화합니다.
        """
        self.optimizer.step()
        self.optimizer.zero_grad()