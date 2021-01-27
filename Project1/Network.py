import numpy as np
from Projects.Project1.Layer import Layer


class Network:

    def __init__(self, layer_specs, cost_function='cross_entropy'):
        self.layer_specs = layer_specs  # Array of tuples in the form [(nodes, w_range, act_func, lr...)...] TODO Should be a dict
        self.cost_function = cost_function
        self.layers = []  # Array of layer objects
        self.weight_array = []  # Array of weight matrices

    def gen_network(self):
        """Initializes a network of weights and biases according to specifications from the configuration file
        parsed into the layer_specs argument.\n
        Calls functions to generate layers and initialize weight matrices and bias vectors"""

        '''Generating layers'''
        for spec in self.layer_specs:
            new_layer = self._gen_layer(spec) # Layer(spec['nodes'], spec['act_func'], )
            self.layers.append(new_layer)

        '''Initializing weights'''
        for i in range(1, len(self.layers)):
            new_weights = self._gen_weights(self.layers[i - 1], self.layers[i])
            self.weight_array.append(new_weights)
        #
        #   if spec['type'] != 'input':  # Note that the input layer MUST have type specified as input
        #       self._gen_weights()

        '''Initializing biases'''
        pass

    def train(self):
        pass

    def _gen_layer(self, l_dict):
        """Takes a dictionary describing the layer from layer_spec as input.\n
        Parses the dictionary to valid arguments for layer constructor.\n
        Returns a layer object"""
        # TODO Handle variable layer configuration
        new_layer = Layer(l_dict['size'], l_dict['act_func'])
        return new_layer

    def _gen_weights(self, layer_i, layer_j):
        """Returns a weight matrix of dimensions layer_i.nodes x layer_j.nodes"""
        # Randomly generated weights within weight range of layer j
        W = np.random.uniform(layer_j.w_range[0], layer_j.w_range[1], (layer_i.size, layer_j.size))  # np.zeros((layer_i.size, layer_j.size))
        # TODO handle other weight initialization algorithms
        return np.array(W)

    def _gen_biases(self):
        pass

    def _forward_pass(self):
        pass

    def _backward_pass(self):
        pass


def main():
    layer_specs1 = [{'size': 3, 'act_func': 'sigmoid', 'lr': None, 'type': 'input'},
                    {'size': 2, 'act_func': 'relu', 'lr': None, 'type': 'hidden'},
                    {'size': 2, 'act_func': 'relu', 'lr': None, 'type': 'hidden'},
                    {'size': 4, 'act_func': 'softmax', 'lr': None, 'type': 'output'}]

    new_network = Network(layer_specs1)
    new_network.gen_network()
    print(new_network.layers)
    print(new_network.weight_array)
    print(type(new_network.weight_array[2]))


if __name__ == '__main__':
    main()
