from microtorch.module.module import Module
from microtorch.optimizer.interface import Optimizer

import numpy as np

# AdamW
class AdamOptimizer(Optimizer):
    def __init__(self, model: Module, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 1e-2):
        self.model = model
        self.lr = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.weight_decay = weight_decay
        self.params_list = list(model.parameters())
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params_list):
            grad = param.grad
            if grad is None:
                continue
                
            # Use parameter index as key for stability
            m = self.m.get(i, np.zeros_like(grad))
            v = self.v.get(i, np.zeros_like(grad))
            
            # Update biased first and second moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            
            # Bias correction
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            
            # AdamW: Apply weight decay directly to parameters (decoupled weight decay)
            param.value = param.value * (1 - self.lr * self.weight_decay)
            
            # Parameter update with Adam step
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param.value -= update
            
            # Save state using parameter index as key
            self.m[i] = m
            self.v[i] = v

    def zero_grad(self):
        for param in self.params_list:
            if param.grad is not None:
                param.grad = np.zeros_like(param.grad)
            else:
                param.grad = np.zeros_like(param.value)
