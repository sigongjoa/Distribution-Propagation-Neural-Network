
import torch.nn as nn
import torch.nn.functional as F
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistFeedForward(nn.Module):
    """
    분포를 처리하는 피드포워드 신경망입니다.

    Args:
        d_model (int): 모델 차원.
        d_ff (int): 피드포워드 신경망의 내부 차원.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1_mu   = nn.Linear(d_model, d_ff)
        self.fc1_var  = nn.Linear(d_model, d_ff)
        self.fc2_mu   = nn.Linear(d_ff, d_model)
        self.fc2_var  = nn.Linear(d_ff, d_model)

    def forward(self, dist: GaussianDistribution):
        """
        입력 분포에 대해 피드포워드 연산을 수행합니다.

        Args:
            dist (GaussianDistribution): 입력 Gaussian 분포.

        Returns:
            GaussianDistribution: 처리된 Gaussian 분포.
        """
        m1 = F.gelu(self.fc1_mu(dist.mu))
        v1 = F.softplus(self.fc1_var(dist.var))
        m2 = self.fc2_mu(m1)
        v2 = F.softplus(self.fc2_var(v1))
        return GaussianDistribution(mu=m2, var=v2)
