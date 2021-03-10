import numpy as np
from Projects.Project1 import DataGeneration


class FullyConnectedLayer:

    def __init__(self, size, activation_func, l_type, optionals, global_lr):
        self.size = int(size)
        self.activation_func = activation_func
        self.w_range = optionals['w_range'] if 'w_range' in optionals else (-0.1, 0.1)
        self.lr = optionals['lr'] if 'lr' in optionals else global_lr
        self.l_type = l_type
        self.weights = None
        self.bias_vector = None
        self.weight_gradient = None # an array that will be filled up with a gradient for each delta. later summed and averaged to update weights
        self.bias_gradient = None
        self.bias_node = 1
        self.cached_activation = []  # caching activation vector to simplify calculation of derivative when backpropping.


    def activation(self, z):
        """Computes the activation of the input vector z with the activation function defined for this particular layer.
        The resulting vector will be cached so that it can be used to easily calculate the derivatives during backprop"""
        if self.activation_func == 'sigmoid':
            return self._sigmoid(z)
        elif self.activation_func == 'tanh':
            return self._tanh(z)
        elif self.activation_func == 'relu':
            return self._relu(z)
        elif self.activation_func == 'linear':
            return self._linear(z)
        else:
            raise NotImplementedError("You have either misspelled the activation function, "
                                      "or this activation function is not implemented")

    def derivation(self):
        """Inserts the cached activation vector in the derivative function defined for this layer"""
        if self.activation_func == 'sigmoid':
            return self._d_sigmoid(self.cached_activation)
        elif self.activation_func == 'tanh':
            return self._d_tanh(self.cached_activation)
        elif self.activation_func == 'relu':
            return self._d_relu(self.cached_activation)
        elif self.activation_func == 'linear':
            return self._d_linear(self.cached_activation)
        else:
            raise NotImplementedError("You have either misspelled the activation function, "
                                      "or the derivative of this activation function is not implemented")

    def gen_weights(self, init_conv_layer_dim):
        """Initializes the weight matrix that feeds into this layer object\n
        Takes size of previous layer as argument"""
        (n_channels, fm_width, fm_height) = init_conv_layer_dim[0].shape # wow, such hack
        W = np.random.uniform(self.w_range[0], self.w_range[1], (n_channels, fm_width, fm_height, self.size))
        self.weights = W

    def gen_bias(self):
        """Initializes a bias vector with all values equal to 0"""
        self.bias_vector = np.zeros(self.size)

    def forward_pass(self, inputs):
        z = np.einsum('ijk,ijkl->l', inputs, self.weights)
        x = self.activation(z)
        self.cached_activation.append(x)
        return x

    def update_weights(self):
        """Sums and averages the array of weight and bias gradients into one weight gradient and one bias gradient
        updates the weights and biases for this layer"""
        new_weights = []
        for weight in self.weights.transpose():
            weight = weight.transpose()
            new_weights.append(weight - self.lr * self.weight_gradient)
            # TODO Figure out how to do this

    def _sigmoid(self, z):
        activation = np.array([1 / (1 + np.exp(-z_i)) for z_i in z])
        return activation

    def _d_sigmoid(self, x):
        return np.array([x_i * (1- x_i) for x_i in x])

    def _tanh(self, z):
        activation = np.array([np.tanh(z_i) for z_i in z])
        return activation

    def _d_tanh(self, x):
        return np.array([1 - x_i**2 for x_i in x])

    def _relu(self, z):
        activation = np.copy(z)
        for i in range(len(z)):
            for j in range(len(z[i])):
                activation[i][j] = (max(0, z[i][j]))
        return np.array(activation)

    def _d_relu(self, x):
        derivative = np.copy(x)
        derivative[derivative <= 0] = 0
        derivative[derivative > 0] = 1
        return derivative

    def _linear(self, z):
        activation = np.copy(z)
        return activation

    def _d_linear(self, x):
        return np.ones_like(x)




def main():
    l1 = FullyConnectedLayer(size=4, activation_func='relu', l_type='fully_connected', optionals={}, global_lr=0.1)
    z = np.array([[1,1,2,1,1],
         [1,1,2,-1,-1]])
    # test_softmax = Layer._softmax(z)

    test_softmax = l1.activation(z)
    print(test_softmax)
    test_d_softmax = l1.derivation()
    print(test_d_softmax)


if __name__ == '__main__':
    main()