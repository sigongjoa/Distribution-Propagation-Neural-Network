import numpy as np
from scipy.special import factorial
from .base import BaseDistribution

class PoissonDistribution(BaseDistribution):
    """
    Poisson 분포를 나타내는 클래스.

    Args:
        lambda_param (float): 분포의 평균 λ (rate parameter)
    """
    def __init__(self, lambda_param: float):
        super().__init__({'lambda': lambda_param})
        self.lambda_param = lambda_param

    def sample(self):
        """
        Poisson 분포에서 샘플을 생성합니다.

        Returns:
            int: 샘플링된 값
        """
        return np.random.poisson(self.lambda_param)

    def log_prob(self, x):
        """
        주어진 값에 대한 로그 확률을 계산합니다.

        Args:
            x (int): 관측값

        Returns:
            float: log P(x)
        """
        # Implement log probability calculation
        if not isinstance(x, int) or x < 0:
            raise ValueError("x must be a non-negative integer for Poisson log_prob.")
        if self.lambda_param <= 0:
            raise ValueError("Lambda must be positive for Poisson log_prob.")
        return x * np.log(self.lambda_param) - self.lambda_param - np.log(factorial(x))

    def combine(self, other: 'PoissonDistribution'):
        """
        두 Poisson 분포를 결합합니다.

        Args:
            other (PoissonDistribution): 결합할 다른 Poisson 분포

        Returns:
            PoissonDistribution: 결합된 Poisson 분포
        """
        # Implement combination logic (e.g., averaging lambda)
        new_lambda = (self.lambda_param + other.lambda_param) / 2
        return PoissonDistribution(lambda_param=new_lambda)