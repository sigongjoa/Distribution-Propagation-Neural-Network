
import torch.nn as nn

class DistributionLMHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.to_logits = nn.Linear(d_model, vocab_size)
    def forward(self, dist):
        # logits = 토큰별 μ를 그대로 로짓으로
        return self.to_logits(dist.mu)  # (B,L,vocab)
