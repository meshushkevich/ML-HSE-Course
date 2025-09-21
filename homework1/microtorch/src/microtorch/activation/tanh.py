from typing import Optional
from microtorch.activation.interface import Activation

import numpy as np


class Tanh(Activation):
    def __init__(self):
        super().__init__()
        self.output : Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (1 - self.output**2)
