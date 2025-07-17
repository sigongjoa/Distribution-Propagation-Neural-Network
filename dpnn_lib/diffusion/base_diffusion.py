from abc import ABC, abstractmethod
from typing import List
import torch

from dpnn_lib.distributions.base import BaseDistribution

class BaseDiffusionProcess(ABC):
    """
    모든 확산 프로세스의 추상 베이스 클래스.

    Args:
        timesteps (int): 총 타임스텝 수.
    """
    def __init__(self, timesteps: int):
        self.timesteps = timesteps

    @abstractmethod
    def forward_step(self, dist: BaseDistribution, t: int) -> BaseDistribution:
        """
        한 타임스텝만큼 노이즈를 더한 후 변화된 분포를 반환합니다.

        Args:
            dist (BaseDistribution): 이전 타임스텝의 분포.
            t (int): 현재 타임스텝.

        Returns:
            BaseDistribution: 노이즈가 추가된 분포.
        """
        pass

    @abstractmethod
    def denoise_block(self, noisy_dist: BaseDistribution) -> BaseDistribution:
        """
        노이즈가 더해진 분포로부터 깨끗한 분포를 복원(denoise)합니다.

        Args:
            noisy_dist (BaseDistribution): 노이즈가 추가된 분포.

        Returns:
            BaseDistribution: 디노이즈된 분포.
        """
        pass

    @abstractmethod
    def loss(self, clean: BaseDistribution, noisy: BaseDistribution, denoised: BaseDistribution, t: int) -> torch.Tensor:
        """
        한 스텝에 대한 손실을 계산합니다.

        Args:
            clean (BaseDistribution): 원본 깨끗한 분포.
            noisy (BaseDistribution): 노이즈가 추가된 분포.
            denoised (BaseDistribution): 디노이즈된 분포.
            t (int): 현재 타임스텝.

        Returns:
            torch.Tensor: 계산된 손실.
        """
        pass

    def run_forward(self, x0: BaseDistribution) -> List[BaseDistribution]:
        """
        0부터 T 타임까지 forward diffusion 전체 궤적을 생성합니다.

        Args:
            x0 (BaseDistribution): 초기 깨끗한 분포.

        Returns:
            List[BaseDistribution]: 각 타임스텝에서의 분포 리스트.
        """
        xs = [x0]
        for t in range(self.timesteps):
            xs.append(self.forward_step(xs[-1], t))
        return xs

    def run_reverse(self, xt: BaseDistribution) -> BaseDistribution:
        """
        T부터 0까지 reverse denoising (샘플링)을 수행합니다.

        Args:
            xt (BaseDistribution): 노이즈가 추가된 최종 분포.

        Returns:
            BaseDistribution: 재구성된 깨끗한 분포.
        """
        x = xt
        for t in reversed(range(1, self.timesteps + 1)): # Corrected range for reversed loop
            x = self.denoise_block(x)
        return x