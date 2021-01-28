import numpy as np
from Projects.Project1.Layer import Layer
from Projects.Project1.DataGeneration import DataGeneration


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
            new_layer = self._gen_layer(spec)
            self.layers.append(new_layer)

        '''Initializing weights'''
        for i in range(1, len(self.layers)-1):  # Input and  Softmax layer does not take in weights--> skipping first and last layers
            self.layers[i].gen_weights(self.layers[i-1].size)

        '''Initializing biases'''
        pass

    def train(self, image_input):
        """Traings the network according to user parameters implemented later"""
        '''Forward pass'''
        self._forward_pass(image_input)

    def _gen_layer(self, l_dict):
        """Takes a dictionary describing the layer from layer_spec as input.\n
        Parses the dictionary to valid arguments for layer constructor.\n
        Returns a layer object"""
        # TODO Handle variable layer configuration
        new_layer = Layer(l_dict['size'], l_dict['act_func'], l_dict['type'])
        return new_layer


    def _gen_biases(self):
        pass

    def _forward_pass(self, image_input):
        x = image_input
        for layer in self.layers:
            x = layer.forward_pass(x)  # calls the forward pass method in each layer object. Not a recursive call
            print(x)

    def _backward_pass(self):
        pass


def main():
    layer_specs1 = [{'size': 3, 'act_func': 'linear', 'lr': None, 'type': 'input'},
                    {'size': 2, 'act_func': 'relu', 'lr': None, 'type': 'hidden'},
                    {'size': 4, 'act_func': 'relu', 'lr': None, 'type': 'hidden'},
                    {'size': 4, 'act_func': 'softmax', 'lr': None, 'type': 'output'}]

    new_network = Network(layer_specs1)
    new_network.gen_network()
    data = DataGeneration(flatten= True)  # TODO make this work for 2d images as well
    data.gen_dataset()
    img_input = [1,1,1]
    new_network.train(img_input)  # data.testSet[-1][-1]

if __name__ == '__main__':
    main()
