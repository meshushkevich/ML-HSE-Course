from abc import ABC, abstractmethod

class Optimizer(ABC):
    @abstractmethod
    def step(self):
        pass
    
    @abstractmethod
    def zero_grad(self):
        pass

