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
        # Creating initial delta for not-softmaxed output layer:
        if self.layers[-1].l_type != 'softmax':
            last_layer = self.layers[-1]
            num_layers = len(self.layers)
        else:
            last_layer = self.layers[-2]
            num_layers = len(self.layers) - 1

        targets = [minibatch[i]['one_hot'] for i in range(len(minibatch))]
        layer_derivatives = last_layer.derivation()
        loss_derivatives = self._loss_derivative(output, targets)
        deltas = []
        for i in range(len(minibatch)):
            deltas.append(loss_derivatives[i] * layer_derivatives[i])
        deltas = np.array(deltas)
        #weight and bias gradient for output(not softmax) layer
        prev_layer_activation = self.layers[num_layers - 2].cached_activation
        weight_gradients = []
        for i in range(len(minibatch)):
            # TODO What to do with softmax here?
            weight_gradients.append(np.einsum('i,j->ij', deltas[i], prev_layer_activation[i])) #TODO Check if correct atleast they're the correct dimension
        # weight gradient is averaged over all training cases in the minibatch
        self.layers[num_layers - 1].weight_gradient = np.average(weight_gradients, axis=0)

        for i in range(num_layers - 2, 0, -1):
            layer_derivatives = self.layers[i].derivation()
            prev_layer_activation = self.layers[i-1].cached_activation
            weight_gradients = []
            new_deltas=[]
            for j in range(len(minibatch)):
                new_deltas.append(np.dot(self.layers[i+1].weights, deltas[j]) * layer_derivatives[j]) #Each delta shoud be a 4x1 array
                weight_gradients.append(np.einsum('i,j->ij', new_deltas[j], prev_layer_activation[j]))    # new weight gradients should be 100X4 matrix (prev layer activation = 100x1)
            deltas = new_deltas
            print('hello')




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
                    {'size': 6, 'act_func': 'sigmoid', 'lr': None, 'type': 'hidden'},
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
