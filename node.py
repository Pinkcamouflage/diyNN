import numpy as np


class Node():
    def __init__(self, num_inputs=None):
        self.weights = np.random.rand(num_inputs)

    def activate(self, inputs):
        # Berechne skalarprodukt von Weights Vektor mit inputs
        print(
            f"running  activate function with inmputs: {inputs} and weights: {self.weights}")
        total = np.dot(self.weights, inputs)
        print(f"this results to total: {total}")
        return self._activationFunction(total)

    def _activationFunction(self, x):
        return max(0, x)

    def print(self):
        print("Node with weights: ", self.weights)
