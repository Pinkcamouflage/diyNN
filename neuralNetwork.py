

from layer import Layer
import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

class NN():
    def __init__(self):
        self.layers: list[Layer] = []
        self.trainingHistory = []

    def addInputLayer(self, amountOfNodes):
        self.input_size = amountOfNodes

    def addHiddenLayer(self, amountOfNodes):
        self.layers.append(Layer(amountOfNodes, self.input_size))
        self.input_size = amountOfNodes

    def addOutputLayer(self, amountOfNodes):
        num_inputs = len(self.layers[-1].nodes)
        self.layers.append(Layer(amountOfNodes, num_inputs))
        
    def forward(self, inputs: np.ndarray):
        activations = [inputs]
        current_input = inputs
        
        for i, layer in enumerate(self.layers):
            layer_output = []

            for node in layer.nodes:
                z = np.dot(current_input, node.weights) + node.bias
                layer_output.append(z)

            z_out = np.stack(layer_output, axis=1)

            if i < len(self.layers) - 1:
                a_out = relu(z_out)
            else:
                a_out = z_out

            activations.append(a_out)
            current_input = a_out

        return activations
            
    
    def backward(self, targets: np.ndarray, activations: list[np.ndarray]):
        lr = 0.01
        batch_size = targets.shape[0]

        output = activations[-1]
        delta = 2 * (output - targets) / batch_size

        for layer_index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[layer_index]
            prev_activation = activations[layer_index]

            new_delta = np.zeros_like(prev_activation)

            for j, node in enumerate(layer.nodes):
                z_j = np.dot(prev_activation, node.weights) + node.bias
                if layer_index < len(self.layers) - 1:
                    delta_j = delta[:, j] * relu_derivative(z_j)
                else:
                    delta_j = delta[:, j]

                dw = np.dot(prev_activation.T, delta_j) / batch_size
                db = np.mean(delta_j)

                node.weights -= lr * dw
                node.bias -= lr * db

                new_delta += np.outer(delta_j, node.weights)

            delta = new_delta
            

    def train(self, inputs, targets, epochs=1):
        for _ in range(epochs):
            result = self.forward(inputs)
                
            loss = sum([(pred - target) ** 2 for pred, target in zip(result[-1], targets)]) / len(inputs)
            
            self.trainingHistory.append(self.forward([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])[-1])
            
            self.backward(targets, result)
            
            print(f"Loss: {loss}")


    def print(self):
        for i, layer in enumerate(self.layers[1:], start=1):
            print(f"\nLayer {i}:")
            layer.print()
