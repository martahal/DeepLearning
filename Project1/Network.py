import numpy as np
from Projects.Project1 import DataGeneration

class Network:

    def __init__(self, layer_specs, cost_function='cross_entropy'):
        self.layer_specs = layer_specs
        self.cost_function = cost_function
        self.layers = []  # Array of layer objects

    def gen_layers(self):

        pass
    def gen_network(self):
        pass

    def train(self):
        pass

    def _forward_pass(self):
        pass

    def _backward_pass(self):
        pass


def main():
    simple_network = Network(())


if __name__ == '__main__':
    main()
