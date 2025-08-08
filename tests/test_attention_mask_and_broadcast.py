import torch
import torch.nn.functional as F
from dpnn.dists.gaussian import GaussianDiag
from dpnn.layers.attention import bilinear_moment
from dpnn.core.moments import softmax_logit_normal

def make_gauss(shape):
    x = torch.randn(*shape)
    d = GaussianDiag.from_tensor(x, init_std=0.1)
    return d

@torch.no_grad()
def test_bilinear_moment_broadcast_shapes():
    # Q, K: (B,H,L,Dh) → scores S: (B,H,L,L)
    B, H, L, Dh = 2, 3, 5, 7
    Q = make_gauss((B, H, L, Dh))
    K = make_gauss((B, H, L, Dh))
    S = bilinear_moment(Q, K, cfg=None)  # cfg 현재 unused 베이스라인
    assert S.mean().shape == (B, H, L, L)
    assert S.var().shape == (B, H, L, L)  # S.var() should be (B, H, L, L) for proper broadcasting
    # 분산은 음수가 아니어야 함
    assert torch.all(S.var() >= 0)

@torch.no_grad()
def test_softmax_logit_normal_exclude_masking():
    # logits 분포: (B,H,L)
    B, H, L = 2, 2, 10
    mu = torch.randn(B, H, L)
    logits = GaussianDiag.from_tensor(mu, init_std=0.05)  # 작은 분산 → α0 비교적 큼

    # exclude: 각 (B,H)에서 임의의 K_top 인덱스를 골라 마스킹
    K_top = 3
    topk_idx = logits.mean().topk(K_top, dim=-1).indices  # (B,H,K_top)

    dirichlet = softmax_logit_normal(logits, exclude=topk_idx)
    p_mean = dirichlet.mean()  # α / α0

    # 마스킹된 위치의 평균확률이 충분히 작아졌는지 검사
    # (완전 0은 아님: Dirichlet α=eps → mean≈eps/α0 이므로 매우 작아야 함)
    gathered = p_mean.gather(dim=-1, index=topk_idx)  # (B,H,K_top)
    assert torch.all(gathered < 1e-3), f"Masked probs not small enough: max={gathered.max().item()}"
