import numpy as np
import matplotlib.pyplot as plt
from Projects.Project1.Layer import Layer
from Projects.Project1.DataGeneration import DataGeneration


class Network:

    def __init__(self, layer_specs, loss_function, verbose, optionals):# lr=0.01, wreg=0.001, reg_option=None, verbose=False):
        self.layer_specs = layer_specs
        self.loss_function = loss_function
        self.lr = optionals['lr'] if 'lr' in optionals else 0.01
        self.w_reg = optionals['w_reg'] if 'w_reg' in optionals else 0.001
        self.reg_option = optionals['reg_option'] if 'reg_option' in optionals else None
        self.verbose = verbose
        self.layers = []  # Array of layer objects

    def gen_network(self):
        """Initializes a network of weights and biases according to specifications from the configuration file
        parsed into the layer_specs argument.\n
        Calls functions to generate layers and initialize weight matrices and bias vectors"""
        '''Generating layers'''
        for spec in self.layer_specs:
            new_layer = self._gen_layer(spec)
            self.layers.append(new_layer)

        '''Initializing weights and biases'''
        for i in range(1, len(self.layers)):
            # Input  does not have associated weights nor bias--> skipping first layer
            self.layers[i].gen_weights(self.layers[i-1].size) # Passes size of previous layer as argument
            self.layers[i].gen_bias()

    def _gen_layer(self, l_dict):
        """Takes a dictionary describing the layer from layer_spec as input.\n
        Parses the dictionary to valid arguments for layer constructor.\n
        Returns a layer object"""
        # TODO Handle variable layer configuration
        new_layer = Layer(l_dict['size'], l_dict['act_func'], l_dict['type'], l_dict, self.lr)
        return new_layer

    def train(self, training_set, validation_set, test_set, batch_size):
        """Trains the network with forward and backward propagation\n
        Takes a training set as input
        returns an array of training loss"""
        # For the number of epochs
        # calculate the activation of each minibatch (a collection of training cases)
        # Compute the training error for the minibatch (sum the errors of each training case)
        # Compute the gradients of the network using backpropagation
        # Update the weights and biases based on the gradient

        training_loss = []
        validation_loss = []
        batch_num = 1
        for i in range(0, len(training_set), batch_size):
            '''Forward pass'''
            train_error, output = self._forward_pass(training_set[i: i + batch_size])
            training_loss.append(np.average(train_error))
            print('Batch number: ', batch_num, 'of', round(len(training_set) / batch_size))
            '''Backward pass'''
            self._backward_pass(output, training_set[i:i + batch_size])
            '''Updating weights and biases in each layer'''
            for layer in self.layers:
                layer.update_weights_and_bias()
            '''Validating training progress'''
            validation_error, val_output = self._forward_pass(validation_set)
            validation_loss.append(np.average(validation_error))
            batch_num += 1
        test_error, test_output = self._forward_pass(test_set)
        if self.verbose:
            for i in range(len(test_set)):
                print('\nInput images: \n', test_set[i]['image'],
                      '\nImage class: ', test_set[i]['class'],
                      '\nOutputs: ', test_output[i],
                      '\nTarget vectors: ', test_set[i]['one_hot'],
                      '\nError: ', test_error[i] )
        self._plot_learning_progress(training_loss, validation_loss, test_error, batch_num)

    def _plot_learning_progress(self, training_loss, validation_loss, test_loss, batch_num):
        # Plotting learning progress:
        x1 = np.linspace(0, batch_num, len(training_loss))
        plt.plot(x1, training_loss, 'b', label='Training loss')
        x2 = np.linspace(0, batch_num, len(validation_loss))
        plt.plot(x2, validation_loss, 'y', label='Validation loss')
        x3 = np.linspace(batch_num, batch_num + len(test_loss), len(test_loss))

        plt.plot(x3, test_loss, 'r', label='Test loss')
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

    def _forward_pass(self, minibatch):
        """Feeds a batch of training, validation or training cases through the network"""
        image_inputs = np.array([minibatch[i]['flat_image'] for i in range(len(minibatch))]) # ['flat_image']  # tentatively only using flattened images
        x = image_inputs
        # TODO make this work for 2d images as well
        for layer in self.layers:
            x = layer.forward_pass(x)  # calls the forward pass method in each layer object. Not a recursive call
        # After the loop, x is the (softmaxed) output of the network
        '''Calculating loss'''
        one_hot_classifications = np.array([minibatch[i]['one_hot'] for i in range(len(minibatch))])
        error = self._calculate_error(x, one_hot_classifications)  # "How wrong is the output"
        return error, x

    def _backward_pass(self, output, minibatch):
        """Takes the output error, and the minibatch of training cases as input,
        updates the weight and bias gradient for each layer"""
        targets = [minibatch[i]['one_hot'] for i in range(len(minibatch))]

        if self.layers[-1].l_type == 'softmax':
            #Creating the initial jacobian to be used for the initial delta calculations when output is softmaxed
            last_layer = self.layers[-2]
            num_layers = len(self.layers) - 1
            initial_jacobian = []
            for i in range(len(minibatch)):
                # Initial Jacobian = d_loss_funciton * J_soft
                temp_loss_deriv = self._loss_derivative(output, targets)[i]
                temp_Jsoft = self.layers[-1].derivation()[i]
                initial_jacobian.append(np.dot(temp_loss_deriv, temp_Jsoft))
        else:
            # Creating initial delta for non-softmaxed output layer:
            last_layer = self.layers[-1]
            num_layers = len(self.layers)
            initial_jacobian = self._loss_derivative(output, targets)

        layer_derivatives = last_layer.derivation()

        deltas = []
        for i in range(len(minibatch)):
            deltas.append(initial_jacobian[i] * layer_derivatives[i])
        deltas = np.array(deltas)
        #weight and bias gradient for output(not softmax) layer
        prev_layer_activation = self.layers[num_layers - 2].cached_activation
        weight_gradients = []
        for i in range(len(minibatch)):
            weight_gradients.append(np.einsum('i,j->ji', deltas[i], prev_layer_activation[i]))
        # weight gradient is averaged over all training cases in the minibatch
        self.layers[num_layers - 1].weight_gradient = np.average(weight_gradients, axis=0)

        for i in range(num_layers - 2, 0, -1):
            layer_derivatives = self.layers[i].derivation()
            prev_layer_activation = self.layers[i-1].cached_activation
            weight_gradients = []
            new_deltas=[]
            for j in range(len(minibatch)):
                new_deltas.append(np.dot(self.layers[i+1].weights, deltas[j]) * layer_derivatives[j])
                weight_gradients.append(np.einsum('i,j->ji', new_deltas[j], prev_layer_activation[j]))
            self.layers[i].weight_gradient = np.average(weight_gradients, axis=0)
            deltas = new_deltas

    def _calculate_error(self, output, target):
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
    image_size = 5
    layer_specs1 = [{'size': image_size**2, 'act_func': 'linear', 'type': 'input'},
                    #{'size': 4, 'act_func': 'relu', 'type': 'hidden'},
                    #{'size': 32, 'act_func': 'relu', 'type': 'hidden'},
                    #{'size': 16, 'act_func': 'relu', 'type': 'hidden'},
                    {'size': 4, 'act_func': 'sigmoid', 'type': 'output'},
                    {'size': 4, 'act_func': 'softmax', 'type': 'softmax'}]

    global_parameters = {'loss': 'cross_entropy', 'verbose': False, 'lr': 0.03, 'w_reg': 0.003, 'reg_option': 'L2'}
    new_network = Network(layer_specs1, global_parameters['loss'], global_parameters['verbose'], global_parameters)
    new_network.gen_network()
    #data = DataGeneration(noise=0.0, img_size=image_size, set_size=1000,
     #                     flatten=True, fig_centered=True,
     #                     train_val_test=(0.9, 0.05, 0.05), draw=False)
    #data.gen_dataset()
    raw_image = np.zeros((5, 5))
    raw_image[0] = np.array([1, 1, 1, 1, 1])
    test_image = np.array([raw_image])
    flat_image = test_image.flatten()
    print(test_image)
    dummy_dataset = [{'class': 'bars', 'one_hot': [0, 1, 0, 0], 'image': test_image, 'flat_image': flat_image},
                     {'class': 'cross', 'one_hot': [1, 0, 0, 0], 'image': test_image, 'flat_image': flat_image}]

    #new_network.train(data.train_set, data.val_set, data.test_set, batch_size=20)
    new_network.train(dummy_dataset, dummy_dataset, dummy_dataset, batch_size=1)

if __name__ == '__main__':
    main()
