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

    def forward(self, x: torch.Tensor):
        """
        입력 텐서에 위치 인코딩을 추가합니다.

        Args:
            x (torch.Tensor): 입력 텐서 (B, L, d).

        Returns:
            torch.Tensor: 위치 인코딩이 추가된 텐서.
        """
        # x = (B, L, d)
        B, L, d = x.shape
        mu_pos  = self.pos_mu[:L].unsqueeze(0)      # (1, L, d)
        var_pos = torch.exp(self.pos_logvar[:L]).unsqueeze(0)
        # 텐서에 위치 인코딩의 평균을 더합니다.
        return x + mu_pos