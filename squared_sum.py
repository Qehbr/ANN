import numpy as np
from cost_function import CostFunction


class SquaredSum(CostFunction):
    def compute(self, activation, expected):
        return np.sum((activation - expected) ** 2)

    def derivative(self, activation, expected):
        return activation - expected


