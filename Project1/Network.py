import numpy as np
from Projects.Project1.Layer import Layer
from Projects.Project1.DataGeneration import DataGeneration


class Network:

    def __init__(self, layer_specs, cost_function='cross_entropy'):
        self.layer_specs = layer_specs
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

        '''Initializing weights and biases'''
        for i in range(1, len(self.layers)-1):
            # Input and  Softmax layer does not have associated weights nor bias--> skipping first and last layers
            self.layers[i].gen_weights(self.layers[i-1].size) # Passes size of previous layer as argument
            self.layers[i].gen_bias()
        pass

    def train(self, training_set):
        """Trains the network with forward and backward propagation\n
        Takes a training set as input
        returns an array of training loss"""
        # TODO: enable types of training e.g. minibatch or SGD
        '''Forward pass'''
        self._forward_pass(training_set[0]) #Tentatively testing with only one training case


    def _gen_layer(self, l_dict):
        """Takes a dictionary describing the layer from layer_spec as input.\n
        Parses the dictionary to valid arguments for layer constructor.\n
        Returns a layer object"""
        # TODO Handle variable layer configuration
        new_layer = Layer(l_dict['size'], l_dict['act_func'], l_dict['type'])
        return new_layer


    def _forward_pass(self, image_input):
        """Feeds a single training-, validation- or test case forward in the network"""
        x = image_input['flat_image'] #tentatively only using flattened images
        # TODO make this work for 2d images as well
        for layer in self.layers:
            x = layer.forward_pass(x)  # calls the forward pass method in each layer object. Not a recursive call
            print(x)
        #after the loop, x is the output of the network
        '''Calculating loss'''
        one_hot_classification = image_input['one_hot']
        error = self._calculate_error(x, one_hot_classification)
        return error

    def _calculate_error(self, output, target):
        raise NotImplementedError('No error calculated')
        pass

    def _backward_pass(self):
        pass



def main():
    layer_specs1 = [{'size': 100, 'act_func': 'linear', 'lr': None, 'type': 'input'},
                    {'size': 2, 'act_func': 'relu', 'lr': None, 'type': 'hidden'},
                    {'size': 4, 'act_func': 'relu', 'lr': None, 'type': 'hidden'},
                    {'size': 4, 'act_func': 'softmax', 'lr': None, 'type': 'output'}]

    new_network = Network(layer_specs1)
    new_network.gen_network()
    data = DataGeneration(flatten= True)
    data.gen_dataset()
    dummy_dataset = [{'class': 'bars', 'image': [1,1,1], 'flat_image': [1,1,1]}]
    new_network.train(data.trainSet) #dummy_dataset)  # data.testSet[-1][-1]

if __name__ == '__main__':
    main()
