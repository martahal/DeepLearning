import numpy as np
from Projects.Project2.ConvLayer2D import ConvLayer2D
from Projects.Project2.FullyConnectedLayer import FullyConnectedLayer

class ConvolutionalNetwork:

    def __init__(self, layer_specs, loss_function, learning_rate=0.01, verbose=False):
        self.layer_specs = layer_specs
        self.loss_function = loss_function
        self.lr = learning_rate
        self.verbose = verbose
        self.fully_connected_layers = []  # Array of FullyConnectedLayer objects
        self.convolutoinal_layers = []   # Array of ConvLayer2D objects

    def gen_network(self):
        for spec in self.layer_specs:
            if spec['type'] == 'conv2d':
                new_conv_layer = self._gen_conv_layer(spec)
                new_conv_layer.gen_kernels()  # Input and output channels is known by the object upon construction
                self.convolutoinal_layers.append(new_conv_layer)
            else:
                new_fc_layer = self._gen_fc_layer(spec)
                new_fc_layer.gen_weights(spec['input_size'])
                self.fully_connected_layers.append(new_fc_layer)


    def _gen_conv_layer(self, spec):
        new_conv_layer = ConvLayer2D(spec['input_channels'],
                                     spec['output_channels'],
                                     spec['kernel_size'],
                                     spec['stride'],
                                     spec['mode'],
                                     spec['act_func'],
                                     spec['lr'])
        return new_conv_layer

    def _gen_fc_layer(self, spec):
        new_fc_layer = FullyConnectedLayer(spec['output_size'],
                                           spec['act_func'],
                                           spec['type'],
                                           spec,
                                           self.lr)
        return new_fc_layer

    def train(self):
        pass

    def _forward_pass(self):
        pass

    def _backward_pass(self):
        pass

    def _calculate_error_and_accuracy(self):
        pass




def main():
    image_size = 10
    specs = [{'input_size': image_size**2, 'output_size': 4, 'act_func': 'linear', 'type': 'input'},
             {'input_channels': 1, 'output_channels': 4,'kernel_size': (2,2),
              'stride': 1, 'mode': 'same', 'act_func': 'relu', 'lr': 0.01, 'type': 'conv2d'},
                    {'input_size': 4, 'output_size': 4, 'act_func': 'sigmoid', 'type': 'output'},
                    {'input_size': 4, 'output_size': 4, 'act_func': 'softmax', 'type': 'softmax'}]

    convnet = ConvolutionalNetwork(specs, loss_function='cross_entropy')
    convnet.gen_network()

if __name__ == '__main__':
    main()