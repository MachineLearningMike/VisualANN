import numpy as np

class Optimizer:
    pass

class DummyOptimizer(Optimizer):
    def __init__(self, alpha=0.5, learningRate=1e-5) -> None:
        self.alpha = alpha
        self.learningRate = learningRate
        self.momentum = 0

    def step(self, weights, wGradients, epoch):
        self.momentum = wGradients * self.learningRate * (1-self.alpha) \
         + self.alpha * self.momentum
        weights = weights - self.momentum
        return weights
        