
import torch
import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistributionEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        # μ와 logσ²를 위한 두 개의 임베딩 테이블
        self.mu_embed  = nn.Embedding(vocab_size, d_model)
        self.logvar_embed = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids):  # token_ids: (batch, seq_len)
        mu     = self.mu_embed(token_ids)         # (B, L, d)
        logvar = self.logvar_embed(token_ids)     # (B, L, d)
        var    = torch.exp(logvar)                # (B, L, d)
        return GaussianDistribution(mu=mu, var=var)
