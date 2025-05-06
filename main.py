
import numpy as np
from neuralNetwork import NN
import pprint


if __name__ == "__main__":
    nn = NN()
    nn.addInputLayer(1)
    nn.addHiddenLayer(2)
    nn.addHiddenLayer(2)
    nn.addOutputLayer(1)

    print("Neural Network created with 3 input nodes, 2 hidden nodes, and 2 output nodes.")
    
    nn.train(np.array([1,2,3,4]), np.array([1,2,3,4]), epochs=1)

    
