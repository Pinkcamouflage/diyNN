

from layer import Layer
import numpy as np


class NN():
    def __init__(self):
        self.layers: list[Layer] = []

    def addInputLayer(self, amountOfNodes):
        self.input_size = amountOfNodes
        self.layers.append(None)

    def addHiddenLayer(self, amountOfNodes):
        num_inputs = self.input_size if len(
            self.layers) == 1 else len(self.layers[-1].nodes)

        self.layers.append(Layer(amountOfNodes, num_inputs))

    def addOutputLayer(self, amountOfNodes):
        num_inputs = len(self.layers[-1].nodes)
        self.layers.append(Layer(amountOfNodes, num_inputs))

    def forward(self, inputData):
        for layer in self.layers[1:]:
            inputData = layer.forward(inputData)
        return inputData
    
    def train(self, inputData: np.ndarray, targetData: np.ndarray, epochs=1):
        for epoch in range(epochs):
            outputs=np.array([self.forward(x) for x in inputData]).flatten()

            loss = np.mean((np.array(outputs) - np.array(targetData)) ** 2)
            
            gradients = []
            for layer in reversed(self.layers[1:]):
                gradients.append([outputs @ (2 * (outputs[i] - targetData[i]) / targetData.shape) for i in range(len(outputs))])
                
            print(gradients)
                

                    

    def print(self):
        for i, layer in enumerate(self.layers[1:], start=1):
            print(f"\nLayer {i}:")
            layer.print()
