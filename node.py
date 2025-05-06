import numpy as np

class Node():
    def __init__(self, weights = None):
        self.weights = weights
    
    def activationFunction(self, x):
        return max(0, x)
    
    def print(self):
        print("Node with weights: ", self.weights)