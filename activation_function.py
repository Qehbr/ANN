from abc import ABC, abstractmethod

class ActivationFunction(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def compute(self, layer):
        pass

    @abstractmethod
    def derivative(self, output):
        pass