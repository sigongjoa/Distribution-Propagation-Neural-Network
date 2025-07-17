import torch.nn as nn
from typing import List
from dpnn_lib.layers.layer import DistributionLayer
from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.distributions.base import BaseDistribution

class DistributionNetwork(nn.Module):
    def __init__(self, layers: List[DistributionLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, initial_distributions: List[BaseDistribution]) -> List[BaseDistribution]:
        # Propagate through layers
        current_dists = initial_distributions
        for i, layer in enumerate(self.layers):
            # Pass the current distributions to the layer's forward method
            current_dists = layer.forward(current_dists)
        return current_dists # Return the final output distributions

    def get_output(self):
        # This method will no longer be needed in its current form
        # The forward method will return the output
        raise NotImplementedError("get_output() is deprecated. Use forward() to get output distributions.")