
import torch
import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution

class PositionalDistributionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        # 전통적 사인/코사인 대신, 위치마다 learnable μ, logσ²
        self.pos_mu     = nn.Parameter(torch.zeros(max_len, d_model))
        self.pos_logvar = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, dist: GaussianDistribution):
        # dist.mu, dist.var = (B, L, d)
        B, L, d = dist.mu.shape
        mu_pos  = self.pos_mu[:L].unsqueeze(0)      # (1, L, d)
        var_pos = torch.exp(self.pos_logvar[:L]).unsqueeze(0)
        # 분포 병합 (μ_i += μ_pos, var_i += var_pos)
        return GaussianDistribution(
            mu  = dist.mu  + mu_pos,
            var = dist.var + var_pos
        )
