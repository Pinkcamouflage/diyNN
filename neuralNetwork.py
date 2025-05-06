

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

    def print(self):
        for i, layer in enumerate(self.layers[1:], start=1):
            print(f"\nLayer {i}:")
            layer.print()
