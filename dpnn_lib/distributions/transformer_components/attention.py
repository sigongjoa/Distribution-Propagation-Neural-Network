import torch
import torch.nn as nn
import torch.nn.functional as F
from ...diffusion.base_diffusion import BaseDistribution
from ...diffusion.gaussian_diffusion import GaussianDistribution

class DistAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Shared linear layers for Q, K, V projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model) # Shared output linear

    def forward(self, input_dist: BaseDistribution):
        if not isinstance(input_dist, GaussianDistribution):
            raise TypeError("DistAttention currently only supports GaussianDistribution")

        mu = input_dist.mu
        var = input_dist.var

        batch_size, seq_len, _ = mu.size()

        # Apply shared linear layers to both mu and var
        q_mu = self.q_linear(mu).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_mu = self.k_linear(mu).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_mu = self.v_linear(mu).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Perform scaled dot-product attention for mu
        output_mu = F.scaled_dot_product_attention(q_mu, k_mu, v_mu)
        output_mu = output_mu.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output_mu = self.out_linear(output_mu)

        # Simplified variance propagation:
        # Apply the same linear layers to variance, but without a second full attention computation.
        # This is a heuristic to gain speed.
        q_var_proj = self.q_linear(var)
        k_var_proj = self.k_linear(var)
        v_var_proj = self.v_linear(var)

        # For simplicity and speed, we'll just apply the output linear layer to the transformed variance
        # and ensure non-negativity. This avoids a second expensive attention calculation.
        output_var = F.softplus(self.out_linear(v_var_proj)) # Using v_var_proj as a simple representation

        return GaussianDistribution(mu=output_mu, var=output_var)
