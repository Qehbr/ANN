import numpy as np
from cost_function import CostFunction


class CrossEntropy(CostFunction):
    def compute(self, activation, expected):
        return np.sum(np.nan_to_num(-expected*np.log(activation)-(1-expected)*np.log(1-activation)))

    def derivative(self, activation, expected):
        return activation - expected


