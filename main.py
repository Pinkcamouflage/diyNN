

from neuralNetwork import NN
import pprint


if __name__ == "__main__":
    nn = NN()
    nn.addInputLayer(3)
    nn.addHiddenLayer(2)
    nn.addOutputLayer(2)
    
    print("Neural Network created with 3 input nodes, 2 hidden nodes, and 2 output nodes.")
    pprint.pprint(nn.print())