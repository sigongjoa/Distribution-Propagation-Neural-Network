
import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistributionLMHead(nn.Module):
    """
    분포를 로짓으로 변환하는 언어 모델 헤드입니다.

    Args:
        d_model (int): 모델 차원.
        vocab_size (int): 어휘 사전의 크기.
    """
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.to_logits = nn.Linear(d_model, vocab_size)
    def forward(self, dist: GaussianDistribution):
        """
        입력 분포의 평균을 로짓으로 변환합니다.

        Args:
            dist (GaussianDistribution): 입력 Gaussian 분포.

        Returns:
            torch.Tensor: 변환된 로짓.
        """
        # logits = 토큰별 μ를 그대로 로짓으로
        return self.to_logits(dist.mu)  # (B,L,vocab)
