import torch
import torch.nn as nn
from typing import List

from dpnn_lib.cells.cell import DistributionCell
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistributionLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.eps = eps
        # Learnable parameters for LayerNorm (gamma and beta)
        # These will operate on the mu and var of the distributions
        # For simplicity, let's assume we normalize mu and var independently
        self.gamma_mu = nn.Parameter(torch.ones(1))
        self.beta_mu = nn.Parameter(torch.zeros(1))
        self.gamma_var = nn.Parameter(torch.ones(1))
        self.beta_var = nn.Parameter(torch.zeros(1))

    def forward(self, input_cells: List[DistributionCell]) -> List[DistributionCell]:
        # Extract mu and var from input distributions
        mus = torch.stack([cell.distribution.mu for cell in input_cells])
        vars = torch.stack([cell.distribution.var for cell in input_cells])

        # Calculate mean and variance for normalization
        mean_mu = mus.mean()
        std_mu = mus.std(unbiased=False) # Population standard deviation

        mean_var = vars.mean()
        std_var = vars.std(unbiased=False)

        # Normalize mu and var
        normalized_mus = self.gamma_mu * (mus - mean_mu) / (std_mu + self.eps) + self.beta_mu
        normalized_vars = self.gamma_var * (vars - mean_var) / (std_var + self.eps) + self.beta_var

        # Create new DistributionCell objects with normalized distributions
        output_cells = []
        for i in range(len(input_cells)):
            # Ensure variance remains positive after normalization
            new_var = torch.relu(normalized_vars[i]) + 1e-6 # Use ReLU to ensure positivity
            output_cells.append(DistributionCell(initial_distribution=GaussianDistribution(mu=normalized_mus[i], var=new_var), embedding_dim=self.embedding_dim))
        
        return output_cells
