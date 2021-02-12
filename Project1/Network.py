import numpy as np
from Projects.Project1.Layer import Layer
from Projects.Project1.DataGeneration import DataGeneration


class Network:

    def __init__(self, layer_specs, cost_function='SSE'):
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
        for i in range(1, len(self.layers)):
            # Input  does not have associated weights nor bias--> skipping first layer
            self.layers[i].gen_weights(self.layers[i-1].size) # Passes size of previous layer as argument
            self.layers[i].gen_bias()


    def train(self, training_set, batch_size):
        """Trains the network with forward and backward propagation\n
        Takes a training set as input
        returns an array of training loss"""
        # For the number of epochs
        # calculate the activation of each minibatch (a collection of training cases)
        # Compute the training error for the minibatch (sum the errors of each training case)
        # Compute the gradients of the network using backpropagation
        # Update the weights and biases based on the gradient

        error = 0
        output = []
        gradient = []
        for i in range(0, len(training_set), batch_size):
            '''Forward pass'''
            error, output = self._forward_pass(training_set[i: i + batch_size])
            print('Error: ', error, "Output: ", output)
            '''Backward pass'''
            self._backward_pass(error, output , training_set[i:i + batch_size])
            '''Updating weights and biases in each layer'''
            for layer in self.layers:
                layer.update_weights_and_bias()


    def _gen_layer(self, l_dict):
        """Takes a dictionary describing the layer from layer_spec as input.\n
        Parses the dictionary to valid arguments for layer constructor.\n
        Returns a layer object"""
        # TODO Handle variable layer configuration
        new_layer = Layer(l_dict['size'], l_dict['act_func'], l_dict['type'])
        return new_layer


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

    def _backward_pass(self, error, output, minibatch):
        """Takes the output error, and the minibatch of training cases as input,
        updates the weight and bias gradient for each layer"""
        # Backward pass across softmax
        targets = np.array([minibatch[i]['one_hot'] for i in range(len(minibatch))])
        loss_derivatives = self._loss_derivative(output, targets)
        j_soft_matrices = self.layers[-1].derivation()
        initial_jacobians = []
        for loss_derivative, j_soft in zip(loss_derivatives, j_soft_matrices):
            initial_jacobian = np.dot(loss_derivative, j_soft)
            initial_jacobians.append(initial_jacobian)
        initial_delta = []




















        '''
        loss_jacobians = self._loss_derivative(output, targets)
        output_jacobians = self.layers[-1].derivation()
        initial_jacobians = []
        for i in range(len(minibatch)):
            # Create the initial jacobian for each of the respective training cases in the minibatch
            initial_jacobian = np.dot(loss_jacobians[i], output_jacobians[i])
            #each jacobian is a m x 1 vector, where m is the size of the output jacobian matrix (m x m)
            initial_jacobians.append(initial_jacobian)
        initial_jacobians = np.array(initial_jacobians)
        print("initial_jacobians:\n ", initial_jacobians)
        # Compute initial delta for last layer (not softmax layer)
        derivatives = self.layers[-2].derivation()
        deltas = []
        for initial_jacobian, derivative in zip(initial_jacobians, derivatives):
            # Taking the outer product since derivatives is supposed to be a matrix with only diagonal entries,
            # but im simplifying it as a vector
            deltas.append(np.outer(initial_jacobian, derivative))
        deltas = np.array(deltas)
        
        # Use initial delta to calculate weight and bias gradients for the hidden layers
        for i in range(len(self.layers)-2, 0, -1): # Propagating backwards
            derivatives = self.layers[i].derivation()
            prev_layer_activations = self.layers[i - 1].cached_inputs
            for j in range(len(deltas)):
                self.layers[i].weight_gradient.append(np.outer(prev_layer_activations[j], deltas[j]))
                deltas[j] = np.dot(np.dot(self.layers[i-1].weights.transpose(), deltas[j]), derivatives[j])
                #self.layers[i].weight_gradient.append(np.outer(deltas[j].transpose(), np.dot(prev_layer_activations[j], derivatives[j]))) # weight gradient is now a list of weight gradients with the same size as the weight matrix
                #self.layers[i].bias_gradient = deltas  # TODO Check correctness
                #deltas[j] = np.dot(derivatives[j], self.layers[i].weights.transpose()) #* derivatives[j]  #TODO Check correctness
        '''


        #for layer m =  (n-1 down to layer 1(first hidden layer) )
            # delta m = np.dot(incoming weights to layer m, previous delta) * layer_m.d_activation
            # layer_m.bias graidient = delta_m
            # layer_m.weight gradient = np.dot( delta_m, activations at layer_m-1)

        # Now all layers has (an array of) weight and bias gradients
        # Each layer can now update incoming weights and biases

        pass

    def _calculate_error(self, output, target):
        if self.cost_function == 'cross_entropy':
            error = self._cross_entropy(output, target)
        elif self.cost_function == 'SSE':
            error = self._sum_of_squared_errors(output, target)
        else:
            raise NotImplementedError("You have either misspelled the loss function, "
                                      "or this loss function is not implemented")
        return error

    def _cross_entropy(self, output, target):
        return -np.sum([target * np.log2(output)])
        # return -np.sum([target[i] * np.log2(output[i]) for i in range(len(output))])  # TODO Check if log2 is correct
        # TODO Maybe add a small number to prevent taking log of 0

    def _sum_of_squared_errors(self, output, target):
        return np.sum([(output - target) ** 2])

    def _mean_squared_errors(self, output, target):
        return 1/len(output[1]) * np.sum([(output - target) ** 2])

    def _loss_derivative(self, output, targets):
        if self.cost_function == 'cross_entropy':
            initial_delta = output - targets
            return initial_delta
        elif self.cost_function == 'SSE':
            pass
        elif self.cost_function == 'MSE':
            pass
        else:
            pass



def main():
    layer_specs1 = [{'size': 100, 'act_func': 'linear', 'lr': None, 'type': 'input'},
                    {'size': 4, 'act_func': 'sigmoid', 'lr': None, 'type': 'hidden'},
                    {'size': 4, 'act_func': 'sigmoid', 'lr': None, 'type': 'output'},
                    {'size': 4, 'act_func': 'softmax', 'lr': None, 'type': 'softmax'}]

    new_network = Network(layer_specs1, cost_function='cross_entropy')
    new_network.gen_network()
    data = DataGeneration(flatten= True)
    data.gen_dataset()
    dummy_dataset = [{'class': 'bars', 'image': [1,1,1], 'flat_image': [1,1,1]}]
    new_network.train(data.trainSet, batch_size=8) #dummy_dataset)  # data.testSet[-1][-1]

if __name__ == '__main__':
    main()
