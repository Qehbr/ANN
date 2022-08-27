from abc import ABC, abstractmethod

class CostFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute(self, activation, expected):
        pass

    @abstractmethod
    def derivative(self, activation, expected):
        pass

