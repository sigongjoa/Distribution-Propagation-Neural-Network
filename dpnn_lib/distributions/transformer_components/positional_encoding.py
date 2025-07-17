
import torch
import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution

class PositionalDistributionEncoding(nn.Module):
    """
    분포에 대한 위치 인코딩을 생성합니다.

    Args:
        max_len (int): 최대 시퀀스 길이.
        d_model (int): 모델 차원.
    """
    def __init__(self, max_len, d_model):
        super().__init__()
        # 전통적 사인/코사인 대신, 위치마다 learnable μ, logσ²
        self.pos_mu     = nn.Parameter(torch.zeros(max_len, d_model))
        self.pos_logvar = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, dist: GaussianDistribution):
        """
        입력 분포에 위치 인코딩을 추가합니다.

        Args:
            dist (GaussianDistribution): 입력 Gaussian 분포.

        Returns:
            GaussianDistribution: 위치 인코딩이 추가된 Gaussian 분포.
        """
        # dist.mu, dist.var = (B, L, d)
        B, L, d = dist.mu.shape
        mu_pos  = self.pos_mu[:L].unsqueeze(0)      # (1, L, d)
        var_pos = torch.exp(self.pos_logvar[:L]).unsqueeze(0)
        # 분포 병합 (μ_i += μ_pos, var_i += var_pos)
        return GaussianDistribution(
            mu  = dist.mu  + mu_pos,
            var = dist.var + var_pos
        )
