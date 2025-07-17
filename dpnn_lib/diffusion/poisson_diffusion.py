# dpnn_lib/diffusion/poisson_diffusion.py

from dpnn_lib.distributions.poisson import PoissonDistribution
from dpnn_lib.diffusion.base_diffusion import BaseDiffusionProcess
import torch

class PoissonDiffusion(BaseDiffusionProcess):
    """
    Poisson 분포에 대한 확산 프로세스를 구현합니다.

    Args:
        model (torch.nn.Module): 디노이징을 수행하는 모델 (예: PoissonDenoiser).
        betas (torch.Tensor): 각 타임스텝에 대한 노이즈 스케줄 (베타 값).
        prior_rate (torch.Tensor): t=0의 Poisson rate prior.
        timesteps (int): 총 타임스텝 수.
    """
    def __init__(self, model, betas, prior_rate, timesteps):
        super().__init__(timesteps)
        self.model = model                  # denoise 네트워크
        self.betas = betas
        self.prior_rate = prior_rate        # t=0의 Poisson rate prior

    def forward_step(self, dist: PoissonDistribution, t: int) -> PoissonDistribution:
        """
        Poisson 분포에 한 타임스텝만큼 노이즈를 더한 후 변화된 분포를 반환합니다.

        Args:
            dist (PoissonDistribution): 이전 타임스텝의 Poisson 분포.
            t (int): 현재 타임스텝.

        Returns:
            PoissonDistribution: 노이즈가 추가된 Poisson 분포.
        """
        λ_prev = dist.lambda_param
        β = self.betas[t]
        # 예시: 지수 감쇠 + 고정 prior
        λ_t = (1 - β) * λ_prev + β * self.prior_rate
        return PoissonDistribution(λ_t)

    def denoise_block(self, noisy_dist: PoissonDistribution) -> PoissonDistribution:
        """
        노이즈가 더해진 Poisson 분포로부터 깨끗한 분포를 복원(denoise)합니다.

        Args:
            noisy_dist (PoissonDistribution): 노이즈가 추가된 Poisson 분포.

        Returns:
            PoissonDistribution: 디노이즈된 Poisson 분포.
        """
        # 모델은 현재 rate와 timestep t를 입력받아 다음 단계 rate를 예측
        # 예: simple MLP(model)로 noisy_rate → rate_hat 예측
        # t를 모델에 전달할 때는 텐서로 변환
        rate_hat = self.model(noisy_dist.lambda_param.unsqueeze(-1), noisy_dist.params['t'] if 't' in noisy_dist.params else None) # Pass t if available
        return PoissonDistribution(rate_hat.squeeze(-1))

    def loss(self, x0: PoissonDistribution, xt: PoissonDistribution, x0_hat: PoissonDistribution, t: int) -> torch.Tensor:
        """
        Poisson 분포에 대한 손실을 계산합니다.

        Args:
            x0 (PoissonDistribution): 원본 깨끗한 Poisson 분포.
            xt (PoissonDistribution): 노이즈가 추가된 Poisson 분포.
            x0_hat (PoissonDistribution): 디노이즈된 Poisson 분포.
            t (int): 현재 타임스텝.

        Returns:
            torch.Tensor: 계산된 손실 (Negative log-likelihood).
        """
        # Negative log-likelihood: − ln p(x0 | rate_hat)
        # x0.sample()은 스칼라 값을 반환하므로, 배치 처리를 위해 x0.lambda_param을 사용
        # torch.distributions.Poisson은 rate 파라미터를 기대
        return -torch.distributions.Poisson(x0_hat.lambda_param.squeeze(-1)).log_prob(x0.lambda_param.round()).mean()