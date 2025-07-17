from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    """
    모든 확률 분포의 추상 베이스 클래스입니다.
    """
    def __init__(self, params: dict):
        """
        BaseDistribution의 생성자입니다.

        Args:
            params (dict): 분포의 파라미터를 담고 있는 딕셔너리.
        """
        self.params = params

    @abstractmethod
    def sample(self):
        """
        분포에서 샘플을 생성하는 추상 메소드입니다.
        하위 클래스에서 구현해야 합니다.
        """
        pass

    @abstractmethod
    def log_prob(self, x):
        """
        주어진 값에 대한 로그 확률을 계산하는 추상 메소드입니다.
        하위 클래스에서 구현해야 합니다.

        Args:
            x: 로그 확률을 계산할 값.
        """
        pass

    @abstractmethod
    def combine(self, other):
        """
        다른 분포와 결합하는 추상 메소드입니다.
        하위 클래스에서 구현해야 합니다.

        Args:
            other (BaseDistribution): 결합할 다른 분포.
        """
        pass
