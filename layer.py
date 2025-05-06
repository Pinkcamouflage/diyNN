

from node import Node


class Layer():
    def __init__(self, amountOfNodes, weights=None):
        self.nodes = [Node() for _ in range(amountOfNodes)]
        print ( f"weights {weights}")
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [Node() for _ in range(amountOfNodes)]
            
    def forward(self, inputData):
        weightedSum = sum([preNode.activationFunction(inputData) for preNode in self.weights])
        return weightedSum
    
    def print(self):
        for node in self.nodes:
            node.print()