from typing import Optional
from microtorch.module.module import Module
from microtorch.utils import Parameter
import numpy as np


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, weight_init_std: float = 0.1):
        super().__init__()
        self._in_features  = in_features
        self._out_features = out_features
        
        self._W = Parameter(
            name  = 'W',
            value = np.random.randn(out_features, in_features).astype(np.float32) * weight_init_std,
            grad  = np.zeros((out_features, in_features), dtype=np.float32),
        )
        self._b = Parameter(
            name  = 'b',
            value = np.zeros(out_features, dtype=np.float32),
            grad  = np.zeros(out_features, dtype=np.float32),
        )
        
        self.register_parameter(self._W)
        self.register_parameter(self._b)

        self._input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._input = x
        return x.dot(self._W.value.T) + self._b.value

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Note: grad_output is assumed to already include any reduction scaling
        # (e.g., CrossEntropyWithLogits.backward() returns (probs - one_hot) / N).
        # Therefore, do NOT divide by batch size here to avoid double-scaling.

        # dL/dW = grad_output^T @ input
        self._W.grad = grad_output.T.dot(self._input)
        # dL/db = sum over batch of grad_output (since grad_output already scaled)
        self._b.grad = grad_output.sum(axis=0)

        # dL/dx = grad_output @ W
        return grad_output.dot(self._W.value)
