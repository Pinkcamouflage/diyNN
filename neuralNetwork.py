

from layer import Layer
import numpy as np 


class NN():
    def __init__(self):
        self.layers: list[Layer] = []
    
    def addInputLayer(self, amountOfNodes):
        self.layers.append(Layer(amountOfNodes))
        
    def addHiddenLayer(self, amountOfNodes):
        self.layers.append(Layer(amountOfNodes, weights=[np.random.randint(0, 2**16) / 2**16 for _ in range(len(self.layers[-1].nodes))]))
        
    def addOutputLayer(self, amountOfNodes):
        self.layers.append(Layer(amountOfNodes, weights=[np.random.randint(0, 2**16) / 2**16 for _ in range(len(self.layers[-1].nodes))]))
        
    def forward(self, inputData):
        for layer in self.layers:
            inputData = layer.forward(inputData)
        return inputData
    
    def print(self):
        for layer in self.layers:
            layer.print()
            