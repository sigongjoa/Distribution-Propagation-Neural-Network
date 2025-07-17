import torch.optim as optim
from dpnn_lib.layers.network import DistributionNetwork
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.utils.metrics import Metrics
from typing import List

class DistributionOptimizer:
    def __init__(self, network: DistributionNetwork, lr: float = 0.001):
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def loss_fn(self, output_distributions: List[BaseDistribution], target_distributions: List[BaseDistribution]):
        if len(output_distributions) != len(target_distributions):
            raise ValueError("Output and target distribution lists must have the same length.")

        total_loss = 0.0
        for output_dist, target_dist in zip(output_distributions, target_distributions):
            # Ensure KL divergence returns a tensor for backprop
            kl_div = Metrics.kl_divergence(output_dist, target_dist)
            total_loss += kl_div
        return total_loss

    def step(self): # Removed 'loss' argument
        self.optimizer.step()
        self.optimizer.zero_grad()