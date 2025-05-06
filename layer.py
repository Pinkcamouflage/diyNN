

from node import Node


class Layer():
    def __init__(self, amountOfNodes, num_inputs_per_node):
        self.nodes = [Node(num_inputs_per_node) for _ in range(amountOfNodes)]

    def forward(self, inputs):
        return [node.activate(inputs) for node in self.nodes]

    def print(self):
        for node in self.nodes:
            node.print()
