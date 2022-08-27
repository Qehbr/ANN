import numpy as np
from activation_function import ActivationFunction


class ReLU(ActivationFunction):
    def compute(self, layer):
        return (layer > 0) * layer

    def derivative(self, output):
        return output > 0