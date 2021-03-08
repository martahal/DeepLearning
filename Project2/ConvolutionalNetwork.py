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
        # TODO make construction of layer simpler
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

    def _forward_pass(self, minibatch):
        """

        :param minibatch: shape((batch_size), image_channels, (image_width, image_height))
        :return: prediction, loss, (accuracy)
        """
        predictions = []
        losses = []
        for i in range(len(minibatch)):
            '''Forward pass through convolutional layers'''
            x = minibatch[i]['image'] # TODO extend so that it forwards a minibatch of images
            for conv_layer in self.convolutoinal_layers:
                x = conv_layer.forward_pass(x)
            '''flattening output to fully connected layers'''
            (n_filters, output_width, output_height) = x.shape
            flattened_output = x.reshape((1, n_filters * output_width * output_height))
            #TODO x.reshape((BATCH_SIZE, n_filters * output_width * output_height))
            y = flattened_output
            '''Forward pass through fully connected layers'''
            for fc_layer in self.fully_connected_layers:
                y = fc_layer.forward_pass(y)
                print(f"Layer: {fc_layer.l_type}, output: {y}")

            '''Calculating loss (and accuracy)'''
            targets = minibatch[i]['one_hot']
            loss = self._calculate_error_and_accuracy(y, targets)

            predictions.extend(y)
            losses.append(loss)
        return np.array(predictions), np.array(losses)

    def _backward_pass(self):
        pass

    def _calculate_error_and_accuracy(self, output, target):
        if self.loss_function == 'cross_entropy':
            error = self._cross_entropy(output, target)
        elif self.loss_function == 'SSE':
            error = self._sum_of_squared_errors(output, target)
        elif self.loss_function == 'MSE':
            error = self._mean_squared_errors(output, target)
        else:
            raise NotImplementedError("You have either misspelled the loss function, "
                                      "or this loss function is not implemented")
        return error

    '''Each of the error functions return an array of errors with error for each case'''
    def _cross_entropy(self, output, target):
        error = -np.sum(target * np.log2(output) + 1e-15)
        return error

    def _sum_of_squared_errors(self, output, target):
        return[(output - target) ** 2]

    def _mean_squared_errors(self, output, target):
        errors = np.zeros(shape=output.shape)
        for i in range(len(output)):
            errors[i] = (1/len(output[0]) * (output[i] - target[i])**2)
        return np.sum(errors, axis=1)

    def _loss_derivative(self, output, targets):
        if self.loss_function == 'cross_entropy':
            d_loss = output - targets
            return d_loss
        elif self.loss_function == 'SSE':
            pass
        elif self.loss_function == 'MSE':
            d_loss = np.zeros(shape=output.shape)
            for i in range(len(output)):
                d_loss[i] = 2/len(output[0]) * (output[i] - targets[i])
            return d_loss
        else:
            pass



def main():
    image_size = 10
    num_filters = 4
    specs = [{'input_channels': 1, 'output_channels': num_filters,'kernel_size': (2,2),
            'stride': 1, 'mode': 'same', 'act_func': 'relu', 'lr': 0.01, 'type': 'conv2d'},
            {'input_size': 64, 'output_size': 4, 'act_func': 'sigmoid', 'type': 'output'},
            {'input_size': 4, 'output_size': 4, 'act_func': 'softmax', 'type': 'softmax'}]
    #specs = [{'input_size': image_size**2, 'output_size': 4, 'act_func': 'linear', 'type': 'input'},
    #         {'input_channels': 1, 'output_channels': 4,'kernel_size': (2,2),
    #          'stride': 1, 'mode': 'same', 'act_func': 'relu', 'lr': 0.01, 'type': 'conv2d'},
    #                {'input_size': 4, 'output_size': 4, 'act_func': 'sigmoid', 'type': 'output'},
    #                {'input_size': 4, 'output_size': 4, 'act_func': 'softmax', 'type': 'softmax'}]
#
    convnet = ConvolutionalNetwork(specs, loss_function='cross_entropy')
    convnet.gen_network()


    raw_image = np.zeros((5,5))
    raw_image[0] = np.array([1,1,1,1,1])
    test_image = np.array([raw_image])
    print(test_image)
    dummy_dataset = [{'class': 'bars','one_hot': [0,1,0,0], 'image': test_image, 'flat_image': [1,1,1]},
                     {'class': 'cross','one_hot': [1,0,0,0], 'image': test_image, 'flat_image': [1,1,1]}]
    feature_map = convnet._forward_pass(dummy_dataset)
    print(feature_map)

if __name__ == '__main__':
    main()