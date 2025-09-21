from abc import ABC, abstractmethod
import numpy as np

from microtorch.utils.parameter import Parameter


class Module(ABC):
    def __init__(self):
        self._parameters : list[Parameter] = []
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    @abstractmethod 
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
    
    def parameters(self) -> list[Parameter]:
        return self._parameters

    def register_parameter(self, param: Parameter):
        self._parameters.append(param)
