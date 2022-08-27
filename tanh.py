import numpy as np
from activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def compute(self, layer):
        return np.tanh(layer)

    def derivative(self, output):
        return 1 - (output ** 2)