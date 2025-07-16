import torch.nn as nn
from typing import List
from dpnn_lib.layers.layer import DistributionLayer
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistributionNetwork(nn.Module):
    def __init__(self, layers: List[DistributionLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self):
        for layer in self.layers:
            layer.forward()

    def get_output(self):
        # Return the distributions of the cells in the last layer
        if not self.layers:
            return []
        
        output_distributions = []
        for cell in self.layers[-1].cells:
            if isinstance(cell.distribution, GaussianDistribution):
                # Detach the tensors to break the computation graph
                detached_mu = cell.distribution.mu.detach()
                detached_var = cell.distribution.var.detach()
                output_distributions.append(GaussianDistribution(mu=detached_mu, var=detached_var))
            else:
                output_distributions.append(cell.distribution)
        return output_distributions
