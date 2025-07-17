
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dpnn_lib.distributions.gaussian import GaussianDistribution

class DistAttention(nn.Module):
    """
    분포에 대한 어텐션 메커니즘입니다.

    Args:
        d_model (int): 모델 차원.
        num_heads (int): 어텐션 헤드 수.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        # Q/K/V는 분포 파라미터 선형 변환
        self.q_mu   = nn.Linear(d_model, d_model)
        self.q_var  = nn.Linear(d_model, d_model)
        self.k_mu   = nn.Linear(d_model, d_model)
        self.k_var  = nn.Linear(d_model, d_model)
        self.v_mu   = nn.Linear(d_model, d_model)
        self.v_var  = nn.Linear(d_model, d_model)
        self.out_mu = nn.Linear(d_model, d_model)
        self.out_var= nn.Linear(d_model, d_model)

    def forward(self, dist: GaussianDistribution):
        """
        입력 분포에 대해 어텐션을 수행합니다.

        Args:
            dist (GaussianDistribution): 입력 Gaussian 분포.

        Returns:
            GaussianDistribution: 어텐션이 적용된 Gaussian 분포.
        """
        # Handle 2D inputs by adding a dummy sequence dimension
        is_2d = dist.mu.dim() == 2
        if is_2d:
            dist = GaussianDistribution(mu=dist.mu.unsqueeze(1), var=dist.var.unsqueeze(1))
        # 1) 프로젝트: μ, var 각각
        Q_mu  = self.q_mu(dist.mu)       # (B,L,d)
        Q_var = F.softplus(self.q_var(dist.var))  # 양수 보장
        # K, V 동일
        K_mu, K_var = self.k_mu(dist.mu), F.softplus(self.k_var(dist.var))
        V_mu, V_var = self.v_mu(dist.mu), F.softplus(self.v_var(dist.var))

        # 2) 헤드 분할
        def split_heads(x):
            B,L,D = x.shape
            x = x.view(B, L, self.num_heads, D//self.num_heads)
            return x.transpose(1,2)  # (B, H, L, D/H)
        Qm, Qv = split_heads(Q_mu), split_heads(Q_var)
        Km, Kv = split_heads(K_mu), split_heads(K_var)
        Vm, Vv = split_heads(V_mu), split_heads(V_var)

        # 3) Attention weights via 분포 거리 (KL approx)
        # KL(Gauss(q)||Gauss(k)) ≈ ...
        # 여기선 간단히 (μ_q-μ_k)^2/(2σ_k) + 0.5*(σ_q/σ_k - log(σ_q/σ_k)-1)
        # 계산 후 softmax
        # 결과: attn (B, H, L, L)
        diff = (Qm.unsqueeze(3) - Km.unsqueeze(2))**2  # (B,H,L,L,Dh)
        kl   = diff / (2*Kv.unsqueeze(2)) + 0.5*(Qv.unsqueeze(3)/Kv.unsqueeze(2) - torch.log(Qv.unsqueeze(3)/Kv.unsqueeze(2)) - 1)
        attn = F.softmax(-kl.sum(-1) / math.sqrt(self.d_k), dim=-1)

        # 4) 분포 가중합: μ_out = Σ attn * μ_v,  var_out = Σ attn^2 * var_v
        out_mu  = torch.matmul(attn, Vm)  # (B,H,L,Dh)
        out_var = torch.matmul(attn**2, Vv)

        # 5) 헤드 합치기
        def combine_heads(x):
            B,H,L,Dh = x.shape
            x = x.transpose(1,2).contiguous().view(B, L, H*Dh)
            return x
        O_mu = combine_heads(out_mu)   # (B,L,d)
        O_var= combine_heads(out_var)

        # 6) 최종 선형
        dist_out = GaussianDistribution(
            mu  = self.out_mu(O_mu),
            var = F.softplus(self.out_var(O_var))
        )

        if is_2d:
            dist_out = GaussianDistribution(mu=dist_out.mu.squeeze(1), var=dist_out.var.squeeze(1))

        return dist_out
