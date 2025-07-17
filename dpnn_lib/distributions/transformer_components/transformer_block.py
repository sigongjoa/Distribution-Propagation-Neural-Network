import torch.nn as nn
from dpnn_lib.distributions.gaussian import GaussianDistribution
from .attention import DistAttention
from .feed_forward import DistFeedForward

class DistTransformerBlock(nn.Module):
    """
    분포를 처리하는 트랜스포머 블록입니다.

    Args:
        d_model (int): 모델 차원.
        num_heads (int): 어텐션 헤드 수.
        d_ff (int): 피드포워드 신경망의 내부 차원.
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = DistAttention(d_model, num_heads)
        self.ffn  = DistFeedForward(d_model, d_ff)
        self.norm1= nn.LayerNorm(d_model)
        self.norm2= nn.LayerNorm(d_model)

    def forward(self, dist_in: GaussianDistribution):
        """
        입력 분포에 대해 셀프 어텐션과 피드포워드 네트워크를 적용합니다.

        Args:
            dist_in (GaussianDistribution): 입력 Gaussian 분포.

        Returns:
            GaussianDistribution: 처리된 Gaussian 분포.
        """
        # Self-Attn
        attn_out = self.attn(dist_in)
        # Residual combine = 분포 가중합
        res1 = GaussianDistribution(
            mu = dist_in.mu  + attn_out.mu,
            var= dist_in.var + attn_out.var
        )
        d1  = GaussianDistribution(
            mu  = self.norm1(res1.mu),
            var = self.norm1(res1.var)
        )
        # FFN
        ffn_out = self.ffn(d1)
        # Residual 2
        res2 = GaussianDistribution(
            mu = d1.mu  + ffn_out.mu,
            var= d1.var + ffn_out.var
        )
        return GaussianDistribution(
            mu  = self.norm2(res2.mu),
            var = self.norm2(res2.var)
        )