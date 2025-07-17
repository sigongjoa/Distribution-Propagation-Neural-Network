
import torch
import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistributionEmbedding(nn.Module):
    """
    토큰 ID를 Gaussian 분포 임베딩으로 변환합니다.

    Args:
        vocab_size (int): 어휘 사전의 크기.
        d_model (int): 모델 차원.
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # μ와 logσ²를 위한 두 개의 임베딩 테이블
        self.mu_embed  = nn.Embedding(vocab_size, d_model)
        self.logvar_embed = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):  # token_ids: (batch, seq_len)
        """
        토큰 ID를 Gaussian 분포 임베딩으로 변환합니다.

        Args:
            token_ids (torch.Tensor): 입력 토큰 ID.

        Returns:
            GaussianDistribution: 변환된 Gaussian 분포 임베딩.
        """
        mu     = self.mu_embed(token_ids)         # (B, L, d)
        logvar = self.logvar_embed(token_ids)     # (B, L, d)
        var    = torch.exp(logvar)                # (B, L, d)
        return GaussianDistribution(mu=mu, var=var)
