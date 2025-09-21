from ..module.module import Module

import numpy as np


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules : list[Module] = list(modules)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for module in self.modules:
            output = module.forward(output)
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for module in reversed(self.modules):
            grad = module.backward(grad)
        return grad

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params
