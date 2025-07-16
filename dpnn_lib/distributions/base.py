from abc import ABC, abstractmethod

class BaseDistribution(ABC):
    def __init__(self, params: dict):
        self.params = params

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def combine(self, other):
        pass
