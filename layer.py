

from node import Node


class Layer():
    def __init__(self, amountOfNodes, num_inputs_per_node):
        self.nodes = [Node(num_inputs_per_node) for _ in range(amountOfNodes)]


    def print(self):
        for node in self.nodes:
            node.print()
