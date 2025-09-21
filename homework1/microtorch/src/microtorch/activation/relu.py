from typing import Optional
from microtorch.activation.interface import Activation

import numpy as np


class ReLU(Activation):
    def __init__(self):
        super().__init__()
        self.mask : Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask
