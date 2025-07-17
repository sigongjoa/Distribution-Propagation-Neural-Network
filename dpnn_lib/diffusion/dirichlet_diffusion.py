# dpnn_lib/diffusion/dirichlet_diffusion.py

import torch
import torch.nn as nn
from dpnn_lib.distributions.dirichlet import DirichletDistribution
from dpnn_lib.diffusion.base_diffusion import BaseDiffusionProcess

class DirichletDiffusion(BaseDiffusionProcess):
    """
    Dirichlet 분포에 대한 확산 프로세스를 구현합니다.

    Args:
        model (torch.nn.Module): 디노이징을 수행하는 모델 (예: DirichletDenoiser).
        betas (torch.Tensor): 각 타임스텝에 대한 노이즈 스케줄 (베타 값).
        prior_alpha (torch.Tensor): t=0의 Dirichlet alpha prior.
        timesteps (int): 총 타임스텝 수.
    """
    def __init__(self, model, betas, prior_alpha, timesteps):
        super().__init__(timesteps)
        self.model = model                  # denoise 네트워크
        self.betas = betas
        self.prior_alpha = prior_alpha      # t=0의 Dirichlet alpha prior

    def forward_step(self, dist: DirichletDistribution, t: int) -> DirichletDistribution:
        """
        Dirichlet 분포에 한 타임스텝만큼 노이즈를 더한 후 변화된 분포를 반환합니다.

        Args:
            dist (DirichletDistribution): 이전 타임스텝의 Dirichlet 분포.
            t (int): 현재 타임스텝.

        Returns:
            DirichletDistribution: 노이즈가 추가된 Dirichlet 분포.
        """
        alpha_prev = dist.alpha_vector
        beta = self.betas[t]
        # 예시: 지수 감쇠 + 고정 prior
        alpha_t = (1 - beta) * alpha_prev + beta * self.prior_alpha
        return DirichletDistribution(alpha_t)

    def denoise_block(self, noisy_dist: DirichletDistribution) -> DirichletDistribution:
        """
        노이즈가 더해진 Dirichlet 분포로부터 깨끗한 분포를 복원(denoise)합니다.

        Args:
            noisy_dist (DirichletDistribution): 노이즈가 추가된 Dirichlet 분포.

        Returns:
            DirichletDistribution: 디노이즈된 Dirichlet 분포.
        """
        # 모델은 현재 alpha와 timestep t를 입력받아 다음 단계 alpha를 예측
        # alpha_vector는 텐서이므로, 모델 입력에 맞게 조정
        # t는 모델에 전달할 때는 텐서로 변환
        alpha_hat = self.model(noisy_dist.alpha_vector, noisy_dist.params['t'] if 't' in noisy_dist.params else None) # Pass t if available
        return DirichletDistribution(alpha_hat)

    def loss(self, x0: DirichletDistribution, xt: DirichletDistribution, x0_hat: DirichletDistribution, t: int) -> torch.Tensor:
        """
        Dirichlet 분포에 대한 손실을 계산합니다.

        Args:
            x0 (DirichletDistribution): 원본 깨끗한 Dirichlet 분포.
            xt (DirichletDistribution): 노이즈가 추가된 Dirichlet 분포.
            x0_hat (DirichletDistribution): 디노이즈된 Dirichlet 분포.
            t (int): 현재 타임스텝.

        Returns:
            torch.Tensor: 계산된 손실 (MSE).
        """
        # KL divergence between true posterior and model prediction
        # For Dirichlet, KL(P || Q) = sum( (alpha_P - alpha_Q) * digamma(alpha_P) ) + log(Beta(alpha_Q)) - log(Beta(alpha_P))
        # Simplified loss: MSE between predicted clean alpha and true clean alpha
        return torch.mean((x0_hat.alpha_vector.squeeze(-1) - x0.alpha_vector)**2)