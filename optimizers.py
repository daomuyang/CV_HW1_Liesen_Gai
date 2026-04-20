import numpy as np

class SGDOptimizer:
    def __init__(self, momentum=0.9):
        self.momentum = momentum
        self.velocity = {}
    
    def _init_velocity(self, model):
        if not self.velocity:
            for key in model.params:
                self.velocity[key] = np.zeros_like(model.params[key])
    
    def step(self, model, grads, lr):
        self._init_velocity(model)
        for key in model.params:
            self.velocity[key] = self.momentum * self.velocity[key] - lr * grads[key]
            model.params[key] += self.velocity[key]