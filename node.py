import numpy as np


class Node():
    def __init__(self, num_inputs=None):
        self.weights = np.random.rand(num_inputs)
        self.bias = 0.0

    def print(self):
        print("Node with weights: ", self.weights)
