import torch
import torch.nn as nn
from typing import List
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.distributions.gaussian import GaussianDistribution # Assuming output is Gaussian

class DistributionCell(nn.Module):
    def __init__(self, initial_distribution: BaseDistribution, embedding_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._distribution = initial_distribution # Store the actual distribution object

        # Network to process current cell's parameters and aggregated neighbor embedding
        # Input: [current_mu, current_var] (2) + aggregated_neighbor_embedding (embedding_dim)
        # Output: [new_mu, new_log_var] (2)
        self.transform_net = nn.Sequential(
            nn.Linear(2 + embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 2)
        )

        # This is for processing neighbor parameters into an embedding
        # Assuming Gaussian: mu, var (2 parameters)
        self.neighbor_param_to_embedding = nn.Linear(2, embedding_dim)

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, new_dist: BaseDistribution):
        self._distribution = new_dist

    def forward(self, neighbor_cells: List['DistributionCell']) -> GaussianDistribution:
        # Extract current cell's parameters
        if not isinstance(self.distribution, GaussianDistribution):
            raise NotImplementedError("DistributionCell only supports GaussianDistribution for now.")
        current_params_tensor = torch.stack([self.distribution.mu, self.distribution.var])

        # Process neighbor parameters and aggregate
        neighbor_embeddings = []
        for neighbor_cell in neighbor_cells:
            if not isinstance(neighbor_cell.distribution, GaussianDistribution):
                raise NotImplementedError("Neighbor DistributionCell only supports GaussianDistribution for now.")
            n_params = torch.stack([neighbor_cell.distribution.mu, neighbor_cell.distribution.var])
            neighbor_embeddings.append(self.neighbor_param_to_embedding(n_params))

        if neighbor_embeddings:
            aggregated_neighbor_embedding = torch.sum(torch.stack(neighbor_embeddings), dim=0)
        else:
            aggregated_neighbor_embedding = torch.zeros(self.embedding_dim, dtype=torch.float32)

        # Combine current cell's parameters and aggregated neighbor embedding
        combined_input = torch.cat([current_params_tensor, aggregated_neighbor_embedding])

        # Predict new parameters
        predicted_params = self.transform_net(combined_input)
        new_mu, new_var_log = predicted_params[0], predicted_params[1]
        new_var = torch.nn.functional.softplus(new_var_log) + 1e-6 # Ensure variance is positive and not too small

        return GaussianDistribution(mu=new_mu, var=new_var)

    def reconstruct(self):
        return self.distribution.sample()