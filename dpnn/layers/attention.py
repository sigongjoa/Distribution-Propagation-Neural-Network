import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.distribution import BaseDistribution
from ..core.config import DistConfig, DistMode
from ..core.moments import softmax_logit_normal #, bilinear_moment
from ..core.sampling import propagate_ukf #, sigma_points
from ..dists.gaussian import GaussianDiag # 임시

# Placeholder functions (to be implemented later)
def bilinear_moment(Q: BaseDistribution, K: BaseDistribution, cfg: DistConfig):
    """
    Q와 K 분포의 쌍선형 모멘트를 근사합니다. (E[QK^T], Var[QK^T])
    """
    # TODO: Implement proper bilinear moment approximation
    # For now, a simple placeholder assuming Q and K are GaussianDiag
    if isinstance(Q, GaussianDiag) and isinstance(K, GaussianDiag):
        # Simplified: just multiply means for score, variance is more complex
        S_mu = Q.loc @ K.loc.transpose(-1,-2)
        
        # Variance propagation for S = QK^T
        # Var(S_ij) = sum_k (E[Q_ik]^2 Var(K_jk) + E[K_jk]^2 Var(Q_ik) + Var(Q_ik)Var(K_jk))
        # Q.loc: (B,H,L_q,Dh), Q.var(): (B,H,L_q,Dh)
        # K.loc: (B,H,L_k,Dh), K.var(): (B,H,L_k,Dh)
        
        Q_loc_sq = Q.loc**2
        K_loc_sq = K.loc**2
        Q_var = Q.var()
        K_var = K.var()

        # Term 1: E[Q_ik]^2 * Var(K_jk)
        # (B,H,L_q,Dh) @ (B,H,Dh,L_k) -> (B,H,L_q,L_k)
        term1 = torch.matmul(Q_loc_sq, K_var.transpose(-1,-2))

        # Term 2: E[K_jk]^2 * Var(Q_ik)
        # (B,H,L_q,Dh) @ (B,H,Dh,L_k) -> (B,H,L_q,L_k)
        term2 = torch.matmul(Q_var, K_loc_sq.transpose(-1,-2))

        # Term 3: Var(Q_ik) * Var(K_jk)
        # (B,H,L_q,Dh) @ (B,H,Dh,L_k) -> (B,H,L_q,L_k)
        term3 = torch.matmul(Q_var, K_var.transpose(-1,-2))

        S_var = term1 + term2 + term3

        return GaussianDiag(S_mu, 0.5*torch.log(S_var + 1e-6))
    else:
        raise NotImplementedError("bilinear_moment only implemented for GaussianDiag for now.")

def _softmax_fn(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(x, dim=dim)

def softmax_via_samples(
    logits_dist: BaseDistribution,
    topk_idx: torch.Tensor | None,
    cfg: DistConfig
) -> BaseDistribution:
    """
    샘플링 기반 (MC/UKF)으로 소프트맥스 확률을 정확화합니다.
    """
    mu = logits_dist.mean()          # (B,H,Lq,Lk)
    var = logits_dist.var().clamp_min(1e-9)
    std = var.sqrt()

    # 샘플 개수
    k = max(cfg.mc_k_min, min(cfg.mc_k_max, 8)) if cfg.mc_k_max > 0 else 0

    if cfg.mode == DistMode.UKF or (cfg.mode == DistMode.AUTO and cfg.use_sigma_points):
        # 배치 펼치기: (B*H*Lq, Lk)로 reshape 후 UKF 반복
        B,H,Lq,Lk = mu.shape
        mu_flat = mu.reshape(B*H*Lq, Lk)
        var_flat = var.reshape(B*H*Lq, Lk)
        
        outs = []
        for i in range(mu_flat.size(0)):
            g = GaussianDiag(mu_flat[i], 0.5*torch.log(var_flat[i]+1e-9)) # Reconstruct GaussianDiag
            y = propagate_ukf(g, lambda z: _softmax_fn(z, dim=-1))
            outs.append(y.mean())  # cov도 원하면 y.cov()에서 diag 꺼내 tiny var로 사용
        p = torch.stack(outs, dim=0).reshape(B,H,Lq,Lk)
        return GaussianDiag(p, torch.log(torch.full_like(p, 1e-6)))

    elif k == 0:
        # fallback: 그냥 mean-softmax
        p = F.softmax(mu, dim=-1)
        return GaussianDiag(p, torch.log(torch.full_like(p, 1e-6)))

    # MC 샘플링
    eps = torch.randn((k,)+mu.shape, device=mu.device, dtype=mu.dtype)
    samples = mu + std * eps
    p_samples = F.softmax(samples, dim=-1)          # (k,B,H,Lq,Lk)
    p_mean = p_samples.mean(dim=0)                  # (B,H,Lq,Lk)

    if topk_idx is None:
        return GaussianDiag(p_mean, torch.log(torch.full_like(p_mean, 1e-6)))

    # rest는 Dirichlet 근사로 보정해서 섞기
    from ..core.moments import softmax_logit_normal
    rest_dir = softmax_logit_normal(logits_dist, exclude=topk_idx)
    rest_mean = rest_dir.mean()                     # (B,H,Lq,Lk)

    # Top-k 위치는 p_mean, 나머지는 rest_mean → 후 renorm
    # 만드는 마스크: one-hot/one for topk indices along last dim
    mask = torch.zeros_like(p_mean, dtype=torch.bool)
    # scatter True into mask with topk_idx
    updates = torch.ones_like(topk_idx, dtype=torch.bool)
    mask = mask.scatter(-1, topk_idx, updates)

    out = torch.where(mask, p_mean, rest_mean)
    out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    return GaussianDiag(out, torch.log(torch.full_like(out, 1e-6)))

def combine_top_rest(attn_weights_top: BaseDistribution, attn_weights_rest: BaseDistribution) -> BaseDistribution:
    """
    Top-k 로짓과 나머지 로짓의 어텐션 가중치를 결합합니다.
    """
    # TODO: Implement proper combination logic
    # For now, a simple placeholder (e.g., element-wise addition, assuming shapes match)
    combined_mu = attn_weights_top.mean() + attn_weights_rest.mean()
    combined_var = attn_weights_top.var() + attn_weights_rest.var()
    return GaussianDiag(combined_mu, torch.log(torch.sqrt(combined_var + 1e-6)))

def linear_combination(V: BaseDistribution, attn: BaseDistribution, cfg: DistConfig) -> BaseDistribution:
    """
    값 분포 V와 어텐션 가중치 attn의 선형 결합을 수행합니다.
    """
    # TODO: Implement proper linear combination for distributions
    # For now, a simple placeholder (e.g., mean-based multiplication)
    Y_mu = attn.mean() @ V.loc
    Y_var = (attn.mean()**2) @ V.var() # Simplified
    return GaussianDiag(Y_mu, 0.5*torch.log(Y_var + 1e-6))


class DistSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, k_top: int = 8, learnable_tau: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k_top = k_top
        self.learnable_tau = learnable_tau

        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        if learnable_tau:
            self.tau = nn.Parameter(torch.tensor(1.0)) # Initial temperature
        else:
            self.register_buffer('tau', torch.tensor(1.0))

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, n_heads, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q_dist: BaseDistribution, K_dist: BaseDistribution, V_dist: BaseDistribution, cfg: DistConfig) -> BaseDistribution:
        # 1. Project Q, K, V distributions
        # Assuming Q_dist, K_dist, V_dist are GaussianDiag for now
        # This part needs to be generalized for BaseDistribution
        Q_proj_dist = Q_dist.affine(self.q_proj.weight, self.q_proj.bias)
        K_proj_dist = K_dist.affine(self.k_proj.weight, self.k_proj.bias)
        V_proj_dist = V_dist.affine(self.v_proj.weight, self.v_proj.bias)

        # Split heads for projected distributions
        # For now, assume GaussianDiag and split loc and log_scale
        Q_proj_dist_split = GaussianDiag(
            self._split_heads(Q_proj_dist.loc),
            torch.log(self._split_heads(Q_proj_dist.scale) + 1e-6)
        )
        K_proj_dist_split = GaussianDiag(
            self._split_heads(K_proj_dist.loc),
            torch.log(self._split_heads(K_proj_dist.scale) + 1e-6)
        )
        V_proj_dist_split = GaussianDiag(
            self._split_heads(V_proj_dist.loc),
            torch.log(self._split_heads(V_proj_dist.scale) + 1e-6)
        )

        # 2. Calculate scores S = QK^T / sqrt(d_k)
        S = bilinear_moment(Q_proj_dist_split, K_proj_dist_split, cfg) # E[S], Var[S]
        
        # Apply scaling factor
        S_scaled_mu = S.mean() / (self.head_dim**0.5)
        S_scaled_var = S.var() / (self.head_dim) # Variance scales by square of factor
        S_scaled = GaussianDiag(S_scaled_mu, torch.log(torch.sqrt(S_scaled_var + 1e-6)))

        # 3. Apply temperature and get logits
        tau = self.tau if self.learnable_tau else cfg.tau_init
        logits_mu = S_scaled.mean() / tau
        logits_var = S_scaled.var() / (tau**2)
        logits_dist = GaussianDiag(logits_mu, torch.log(torch.sqrt(logits_var + 1e-6)))

        # 4. Softmax approximation strategy
        logits_mu_for_topk = logits_dist.mean()  # (B,H,L_q,L_k)
        
        # k_top 방어 로직: 시퀀스 길이(L_k)가 k_top보다 작으면 k_top을 시퀀스 길이로 제한
        Lk = logits_mu_for_topk.size(-1)
        k = min(self.k_top, Lk)
        topk_idx = logits_mu_for_topk.topk(k, dim=-1).indices  # (..., K)

        # Top-k logits: MC/UKF
        attn_weights_top = softmax_via_samples(logits_dist, topk_idx, cfg)
        
        # Remaining logits: logit-normal approximation
        attn_weights_rest = softmax_logit_normal(logits_dist, exclude=topk_idx)

        # Combine attention weights
        attn = combine_top_rest(attn_weights_top, attn_weights_rest)

        # 5. Value propagation: Y = Attn * V
        Y_dist_split = linear_combination(V_proj_dist_split, attn, cfg)

        # Combine heads for output distribution
        Y_dist = GaussianDiag(
            self._combine_heads(Y_dist_split.loc),
            torch.log(self._combine_heads(Y_dist_split.scale) + 1e-6)
        )

        # 6. Final linear projection
        output_dist = Y_dist.affine(self.out_proj.weight, self.out_proj.bias)

        return output_dist
