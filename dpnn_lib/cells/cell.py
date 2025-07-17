import torch
import torch.nn as nn
from typing import List
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.distributions.gaussian import GaussianDistribution # Assuming output is Gaussian

class DistributionCell(nn.Module):
    """
    분포를 처리하고 업데이트하는 기본 셀 단위입니다.

    Args:
        initial_distribution (BaseDistribution): 셀의 초기 분포.
        embedding_dim (int, optional): 임베딩 차원 (기본값: 16).
    """
    def __init__(self, initial_distribution: BaseDistribution, embedding_dim: int = 16):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.initial_distribution = initial_distribution # Store initial distribution

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

    def forward(self, current_distribution: BaseDistribution, neighbor_distributions: List[BaseDistribution]) -> GaussianDistribution:
        """
        현재 분포와 이웃 분포를 기반으로 새로운 분포를 계산합니다.

        Args:
            current_distribution (BaseDistribution): 현재 셀의 분포.
            neighbor_distributions (List[BaseDistribution]): 이웃 셀들의 분포 리스트.

        Returns:
            GaussianDistribution: 업데이트된 Gaussian 분포.
        """
        # Extract current cell's parameters
        if not isinstance(current_distribution, GaussianDistribution):
            raise NotImplementedError("DistributionCell only supports GaussianDistribution for now.")
        current_params_tensor = torch.stack([current_distribution.mu, current_distribution.var])

        # Process neighbor parameters and aggregate
        neighbor_embeddings = []
        for neighbor_dist in neighbor_distributions:
            if not isinstance(neighbor_dist, GaussianDistribution):
                raise NotImplementedError("Neighbor DistributionCell only supports GaussianDistribution for now.")
            n_params = torch.stack([neighbor_dist.mu, neighbor_dist.var])
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

    def reconstruct(self, distribution: BaseDistribution): # Takes distribution as argument
        """
        주어진 분포에서 샘플을 재구성합니다.

        Args:
            distribution (BaseDistribution): 재구성할 분포.

        Returns:
            torch.Tensor: 재구성된 샘플.
        """
        return distribution.sample()