import torch
import torch.nn as nn
from math import sqrt
from typing import List

from dpnn_lib.distributions.gaussian import GaussianDistribution
from dpnn_lib.distributions.base import BaseDistribution
from dpnn_lib.diffusion.base_diffusion import BaseDiffusionProcess
from dpnn_lib.distributions.transformer_components.transformer_block import DistTransformerBlock

class GaussianDiffusion(BaseDiffusionProcess):
    """
    Gaussian 분포에 대한 확산 프로세스를 구현합니다.

    Args:
        denoise_model (torch.nn.Module): 디노이징을 수행하는 모델 (예: DistDiffusionBlock).
        timesteps (int): 총 타임스텝 수.
        betas (torch.Tensor): 각 타임스텝에 대한 노이즈 스케줄 (베타 값).
    """
    def __init__(self, denoise_model: nn.Module, timesteps: int, betas: torch.Tensor):
        super().__init__(timesteps)
        self.denoise_model = denoise_model # This will be DistDiffusionBlock
        self.betas = betas
        self.alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def forward_step(self, dist_prev: BaseDistribution, t: int) -> BaseDistribution:
        """
        Gaussian 분포에 한 타임스텝만큼 노이즈를 더한 후 변화된 분포를 반환합니다.

        Args:
            dist_prev (BaseDistribution): 이전 타임스텝의 Gaussian 분포.
            t (int): 현재 타임스텝.

        Returns:
            BaseDistribution: 노이즈가 추가된 Gaussian 분포.
        """
        if not isinstance(dist_prev, GaussianDistribution):
            raise NotImplementedError("GaussianDiffusion only supports GaussianDistribution for now.")
        
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]

        # 1) 분포 스케일링: μ_t = sqrt(alpha)*μ_{t-1}
        mu_t = sqrt(alpha_t) * dist_prev.mu
        # 2) 분산 업데이트: var_t = alpha*var_{t-1} + beta
        var_t = alpha_t * dist_prev.var + beta_t
        return GaussianDistribution(mu=mu_t, var=var_t)

    def denoise_block(self, noisy_dist: BaseDistribution) -> BaseDistribution:
        """
        노이즈가 더해진 Gaussian 분포로부터 깨끗한 분포를 복원(denoise)합니다.

        Args:
            noisy_dist (BaseDistribution): 노이즈가 추가된 Gaussian 분포.

        Returns:
            BaseDistribution: 디노이즈된 Gaussian 분포.
        """
        if not isinstance(noisy_dist, GaussianDistribution):
            raise NotImplementedError("GaussianDiffusion denoise_block only supports GaussianDistribution for now.")
        
        # The denoise_model (DistDiffusionBlock) takes a distribution and a timestep
        # and returns a denoised distribution.
        # Note: The timestep 't' is implicitly handled by the denoise_model if it needs it.
        return self.denoise_model(noisy_dist, noisy_dist.params['t'] if 't' in noisy_dist.params else None) # Pass t if available

    def loss(self, clean: BaseDistribution, noisy: BaseDistribution, denoised: BaseDistribution, t: int) -> torch.Tensor:
        """
        Gaussian 분포에 대한 손실을 계산합니다.

        Args:
            clean (BaseDistribution): 원본 깨끗한 Gaussian 분포.
            noisy (BaseDistribution): 노이즈가 추가된 Gaussian 분포.
            denoised (BaseDistribution): 디노이즈된 Gaussian 분포.
            t (int): 현재 타임스텝.

        Returns:
            torch.Tensor: 계산된 손실 (MSE).
        """
        if not isinstance(clean, GaussianDistribution) or \
           not isinstance(noisy, GaussianDistribution) or \
           not isinstance(denoised, GaussianDistribution):
            raise NotImplementedError("GaussianDiffusion loss only supports GaussianDistribution for now.")
        
        # Assuming the denoised distribution is a prediction of the clean distribution
        # We can use KL divergence as a loss function
        # KL(P || Q) = 0.5 * (log(var2/var1) + (var1 + (mu1-mu2)^2)/var2 - 1)
        
        # Simplified loss: MSE between predicted clean mu and true clean mu
        # and MSE between predicted clean var and true clean var
        mu_loss = torch.mean((denoised.mu - clean.mu)**2)
        var_loss = torch.mean((denoised.var - clean.var)**2)
        
        # You might want to weight these losses differently or use KL divergence directly
        return mu_loss + var_loss

# DistDiffusionBlock (from denoise.py) will be used as the denoise_model
class DistDiffusionBlock(nn.Module):
    """
    확산 프로세스에서 디노이징을 수행하는 블록입니다.
    트랜스포머 블록을 사용하여 분포를 처리합니다.

    Args:
        d_model (int): 모델 차원.
        num_heads (int): 어텐션 헤드 수.
        d_ff (int): 피드포워드 신경망의 내부 차원.
    """
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.transformer_block = DistTransformerBlock(d_model, num_heads, d_ff)

    def forward(self, dist_t: GaussianDistribution, t: int):
        """
        주어진 분포를 디노이징합니다.

        Args:
            dist_t (GaussianDistribution): 노이즈가 추가된 Gaussian 분포.
            t (int): 현재 타임스텝 (API 일관성을 위해 유지).

        Returns:
            GaussianDistribution: 디노이즈된 Gaussian 분포.
        """
        # The time step `t` is not used in this implementation, but is kept for API consistency.
        return self.transformer_block(dist_t)