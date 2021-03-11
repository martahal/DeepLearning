import numpy as np
import matplotlib.pyplot as plt
from Projects.Project2.ConvLayer2D import ConvLayer2D
from Projects.Project2.DenseLayer import DenseLayer
from Projects.Project2.FullyConnectedLayer import FullyConnectedLayer
from Projects.Project2.DataGeneration2 import DataGeneration2

class ConvolutionalNetwork:

    def __init__(self, layer_specs, loss_function, verbose,visualize_kernels, optionals):
        self.layer_specs = layer_specs
        self.loss_function = loss_function
        self.lr = optionals['lr'] if 'lr' in optionals else 0.01
        self.verbose = verbose
        self.visualize_kernels = visualize_kernels
        self.dense_layers = []  # Array of FullyConnectedLayer object
        self.convolutional_layers = []   # Array of ConvLayer2D objects
        self.fully_connected_layer = None

    def gen_network(self):
        for spec in self.layer_specs:
            if spec['type'] == 'conv2d':
                new_conv_layer = self._gen_conv_layer(spec)
                new_conv_layer.gen_kernels()  # Input and output channels is known by the object upon construction
                self.convolutional_layers.append(new_conv_layer)
            elif spec['type']== 'fully_connected':
                new_fc_layer = self._gen_fc_layer(spec)
                new_fc_layer.gen_weights(self.convolutional_layers[-1].output_dimensions)
                self.fully_connected_layer =new_fc_layer
            else:
                new_dense_layer = self._gen_dense_layer(spec)
                new_dense_layer.gen_weights(spec['input_size'])
                self.dense_layers.append(new_dense_layer)


    def _gen_conv_layer(self, spec):
        # TODO make construction of layer simpler
        new_conv_layer = ConvLayer2D(spec['spatial_dimensions'],
                                     spec['input_channels'],
                                     spec['output_channels'],
                                     spec['kernel_size'],
                                     spec['stride'],
                                     spec['mode'],
                                     spec['act_func'],
                                     spec['lr'])
        return new_conv_layer

    def _gen_dense_layer(self, spec):
        new_dense_layer = DenseLayer(spec['output_size'],
                                  spec['act_func'],
                                  spec['type'],
                                  spec,
                                  self.lr)
        return new_dense_layer
    def _gen_fc_layer(self, spec):
        new_fc_layer = FullyConnectedLayer(spec['output_size'],
                                           spec['act_func'],
                                           spec['type'],
                                           spec,
                                           self.lr)
        return new_fc_layer

    def train(self, training_set,validation_set, test_set, batch_size):
        """
        Forward passes a batch of training cases and calculate the networks prediction and loss
        Backward passes the loss gradient through the network
        Calculates validation and test error, as well as plotting the learning progress

        :param training_set: The whole dataset to train the network.
                            A list of dictionaries with keys 'class', 'one_hot' 'image' and 'flat'image'
                            shape:(len(training_set), 1)
        :param batch_size: Number of training cases to feed the network simultaneously
        """
        training_loss = []
        validation_loss = []
        test_loss = []
        batch_num = 1
        for i in range(0, len(training_set), batch_size):
            '''Forward pass'''
            predictions, train_errors = self._forward_pass(training_set[i: i + batch_size])
            training_loss.append(np.average(train_errors))
            print('Batch number: ', batch_num, 'of', round(len(training_set) / batch_size))
            '''Backward pass'''
            self._backward_pass(predictions, training_set[i:i + batch_size])
            '''Updating weights and biases in each dense layer'''
            for dense_layer in self.dense_layers:
                dense_layer.update_weights_and_bias()

            '''Updating weights for fully connected layer'''
            self.fully_connected_layer.update_weights()

            '''Updating weights for convolutional layers'''
            for conv_layer in self.convolutional_layers:
                conv_layer.update_filters()

            '''Validating training progress'''
            val_output, validation_error = self._forward_pass(validation_set)
            validation_loss.append(np.average(validation_error))
            batch_num += 1
        test_output, test_error = self._forward_pass(test_set)
        test_loss.append(np.average(test_error))
        if self.verbose:
            for i in range(len(test_set)):
                print('\nInput images: \n', test_set[i]['image'][0],
                      '\nImage class: ', test_set[i]['class'],
                      '\nOutputs: ', test_output[i],
                      '\nTarget vectors: ', test_set[i]['one_hot'],
                      '\nError: ', test_error[i] )

        print('Test loss: ', np.average(test_error))
        self._plot_learning_progress_and_kernels(training_loss, validation_loss, test_loss, batch_num)

    def _plot_learning_progress_and_kernels(self, training_loss, validation_loss, test_loss, batch_num):
        # Plotting learning progress:
        x1 = np.linspace(0, batch_num, len(training_loss))
        plt.plot(x1, training_loss, 'b', label='Training loss')
        x2 = np.linspace(0, batch_num, len(validation_loss))
        plt.plot(x2, validation_loss, 'y', label='Validation loss')

        plt.hlines(np.average(test_loss),xmin=batch_num, xmax=batch_num+ 0.1*batch_num, label= 'Test loss', colors='r')

        plt.xlabel('Minibatch')
        if self.loss_function == 'cross_entropy':
            plt.ylabel('Cross Entropy Loss')
            plt.ylim(bottom=0,top=2.1)
            indicator = np.ones_like(training_loss) * 2
            plt.plot(x1, indicator, 'g--', label='Pure guessing')
        elif self.loss_function == 'MSE':
            plt.ylabel('Mean Squared Error Loss')
            plt.ylim(bottom=0, top=0.2)
            indicator = np.ones_like(training_loss) * 0.1875
            plt.plot(x1, indicator, 'g--', label='Pure guessing')
        plt.title('Learning progress')
        plt.legend()
        plt.show()

        if self.visualize_kernels:
            for conv_layer in self.convolutional_layers:
                conv_layer.visualize_kernels()

    def _forward_pass(self, minibatch):
        """

        :param minibatch: shape((batch_size), image_channels, (image_width, image_height))
        :return: prediction, loss, (accuracy)
        """
        activations = []
        for i in range(len(minibatch)):
            '''Forward pass through convolutional layers'''
            x = minibatch[i]['image'] 
            for conv_layer in self.convolutional_layers:
                x = conv_layer.forward_pass(x)

            '''Forward pass through fully connected layer'''
            x = self.fully_connected_layer.forward_pass(x)
            activations.append(x)
        '''Forward pass through dense layers'''
        activations = np.array(activations)
        # Doing this for the whole batch since that was the way it was done in the prev. project
        for dense_layer in self.dense_layers:
            activations = dense_layer.forward_pass(activations)
            #print(f"Layer: {dense_layer.l_type}, output: {activations}")

        '''Calculating loss (and accuracy)'''
        targets = [minibatch[i]['one_hot'] for i in range(len(minibatch))]
        loss = self._calculate_error_and_accuracy(activations, targets)

        predictions = activations
        return np.array(predictions), np.array(loss)

    def _backward_pass(self, output, minibatch):
        """
        Calculates the gradients for each fully connected and convolutional layer


        :param output: The predicted output of the network after forward pass.
                        shape:(batch_size, 4)
        :param minibatch: The minibatch of training cases. A list of dictionaries
                        shape: (batch_size, 1)
        """

        '''Backward pass for dense layers'''
        targets = [minibatch[i]['one_hot'] for i in range(len(minibatch))]

        if self.dense_layers[-1].l_type == 'softmax':
            # Creating the initial jacobian to be used for the initial delta calculations when output is softmaxed
            last_layer = self.dense_layers[-2]
            num_layers = len(self.dense_layers) - 1
            initial_jacobian = []
            for i in range(len(minibatch)):
                # Initial Jacobian = d_loss_funciton * J_soft
                temp_loss_deriv = self._loss_derivative(output, targets)[i]
                temp_Jsoft = self.dense_layers[-1].derivation()[i]
                initial_jacobian.append(np.dot(temp_loss_deriv, temp_Jsoft))
        else:
            # Creating initial delta for non-softmaxed output layer:
            last_layer = self.dense_layers[-1]
            num_layers = len(self.dense_layers)
            initial_jacobian = self._loss_derivative(output, targets)

        layer_derivatives = last_layer.derivation()

        deltas = []
        for i in range(len(minibatch)):
            deltas.append(initial_jacobian[i] * layer_derivatives[i])
        deltas = np.array(deltas)
        # weight and bias gradient for output(not softmax) layer
        if num_layers - 1 == 0:
            prev_layer_activation = self.fully_connected_layer.cached_activation
        else:
            prev_layer_activation = self.dense_layers[num_layers - 1].cached_activation
        weight_gradients = []
        for i in range(len(minibatch)):
            weight_gradients.append(np.einsum('i,j->ji', deltas[i], prev_layer_activation[i]))
        # weight gradient is averaged over all training cases in the minibatch
        self.dense_layers[num_layers - 1].weight_gradient = np.average(weight_gradients, axis=0)

        for i in range(num_layers - 2, 0, -1):  # TODO: would this break if there is only one fc_layer before softmax? probably yes
            # In this case the first fc_layer is not an input layer.
            # However, the weight gradient for the first fc_layer doesn't work here since prev_layer is convlayer
            layer_derivatives = self.dense_layers[i].derivation()
            prev_layer_activation = self.dense_layers[i - 1].cached_activation
            weight_gradients = []
            new_deltas = []
            for j in range(len(minibatch)):
                new_deltas.append(np.dot(self.dense_layers[i + 1].weights, deltas[j]) * layer_derivatives[j])
                weight_gradients.append(np.einsum('i,j->ji', new_deltas[j], prev_layer_activation[j]))
            self.dense_layers[i].weight_gradient = np.average(weight_gradients, axis=0)
            deltas = new_deltas

        '''Backward pass linking dense layer and convolution layer'''
        # First we need to backward pass the  fully connected layer with the activation from the last convolutional layer
        layer_derivatives = self.fully_connected_layer.derivation()
        prev_layer_activation = self.convolutional_layers[-1].cached_activation #THIS IS CORRECT, but figure out if this should be flattened
        weight_gradients = []
        fc_deltas = []
        output_jacobians = []
        for j in range(len(minibatch)):
            fc_deltas.append(np.dot(self.dense_layers[0].weights, deltas[j]) * layer_derivatives[j])

            weight_gradients.append(np.einsum('l,kij->kij', fc_deltas[j], prev_layer_activation[j]))

            output_jacobians.append(np.einsum('l,ijkl ->ijk', fc_deltas[j], self.fully_connected_layer.weights))
        self.fully_connected_layer.weight_gradient = np.average(weight_gradients, axis=0)



        delta_jacobians = np.array(output_jacobians)
        # Delta jacobian has shape: (batch_size, number of filters, kernel_height, kernel_width)
        # Only passing one delta jacobian at a time to the backward pass function in the conv layer

        ''' Backward pass for convolutional layers'''
        for i in range(len(self.convolutional_layers)-1, -1, -1):

            if i == 0:
                # in order to adjust filters in first convolutional layer
                upstream_feature_map = [minibatch[k]['image'] for k in range(len(minibatch))]
            else:
                upstream_feature_map = self.convolutional_layers[i - 1].cached_activation
            updated_jacobians = []
            filter_gradients = []
            # Only passing one delta jacobian at a time to the backward pass function in the conv layer
            for j in range(len(delta_jacobians)):
                # Backward convolution is handled in each conv layer
                updated_delta_jacobian, filter_gradient = self.convolutional_layers[i].\
                    backward_pass(
                        delta_jacobians[j],
                        upstream_feature_map[j]
                    )
                # TODO resolve
                derived_jacobian = self.convolutional_layers[i].derivation(updated_delta_jacobian)
                updated_jacobians.append(derived_jacobian)
                #updated_jacobians.append(updated_delta_jacobian)
                filter_gradients.append(filter_gradient)
            delta_jacobians = np.array(updated_jacobians)
            self.convolutional_layers[i].filter_gradient = np.average(filter_gradients,axis=0)


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
        error = [-np.sum(target[i] * np.log2(output[i]) + 1e-15) for i in range(len(output))]
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

    raw_image = np.zeros((5,5))
    raw_image[0] = np.array([1, 1, 1, 1, 1])
    test_image = np.array([raw_image])
    print(test_image)

    dummy_dataset = [{'class': 'bars', 'one_hot': [0, 1, 0, 0], 'image': test_image, 'flat_image': [1, 1, 1]},
                     {'class': 'cross', 'one_hot': [1, 0, 0, 0], 'image': test_image, 'flat_image': [1, 1, 1]},
                     {'class': 'bars', 'one_hot': [0, 1, 0, 0], 'image': test_image, 'flat_image': [1, 1, 1]},
                     {'class': 'cross', 'one_hot': [1, 0, 0, 0], 'image': test_image, 'flat_image': [1, 1, 1]}
                     ]

    #image_size = 13
    num_filters = 3
    # real_dataset = DataGeneration2(noise=0.0, img_size=image_size, set_size=2000,
    #                             flatten=True, fig_centered=True,
    #                             train_val_test=(0.9, 0.05, 0.05), draw=False)
    # real_dataset.gen_dataset()

    specs = [
        {'spatial_dimensions':(5,5), 'input_channels': 1, 'output_channels': num_filters,'kernel_size': (3,3),
            'stride': 2, 'mode': 'same', 'act_func': 'selu', 'lr': 0.01, 'type': 'conv2d'},
        {'spatial_dimensions':(6,6),'input_channels': num_filters, 'output_channels': num_filters*2,'kernel_size': (3,3),
            'stride': 1, 'mode': 'same', 'act_func': 'selu', 'lr': 0.01, 'type': 'conv2d'},
        {'input_size': 4 * 4 *num_filters*2, 'output_size': 8, 'act_func': 'sigmoid', 'type': 'fully_connected'},
        {'input_size': 8, 'output_size': 4, 'act_func': 'sigmoid', 'type': 'output'},
        # NOTE Cannot remove intermediate dense layer yet
        {'input_size': 4, 'output_size': 4, 'act_func': 'softmax', 'type': 'softmax'}]

    convnet = ConvolutionalNetwork(specs, loss_function='cross_entropy', verbose=True)
    convnet.gen_network()

    convnet.train(
        dummy_dataset,
        dummy_dataset,
        dummy_dataset,
        batch_size=1)

    #feature_map = convnet.train(dummy_dataset, 2)


if __name__ == '__main__':
    main()