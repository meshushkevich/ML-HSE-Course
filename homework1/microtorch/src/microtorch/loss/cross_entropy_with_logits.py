from typing import Optional
from ..module.module import Module

import numpy as np


class CrossEntropyWithLogits(Module):
    def __init__(self):
        self._probs  : Optional[np.ndarray] = None
        self._labels : Optional[np.ndarray] = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        # logits: (N, C), labels: (N,) integers
        shift = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shift)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
    
        self._probs = probs
        self._labels = labels
        
        N = logits.shape[0]
        loss = -np.log(probs[np.arange(N), labels] + 1e-12).mean()
        return loss

    def backward(self):
        N = self._probs.shape[0]
        grad = self._probs.copy()
        grad[np.arange(N), self._labels] -= 1
        grad /= N
        return grad
