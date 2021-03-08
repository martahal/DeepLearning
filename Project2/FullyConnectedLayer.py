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
        elif self.activation_func == 'softmax':
            return self._softmax(z)
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
        elif self.activation_func == 'softmax':
            return self._d_softmax(self.cached_activation)
        else:
            raise NotImplementedError("You have either misspelled the activation function, "
                                      "or the derivative of this activation function is not implemented")

    def gen_weights(self, prev_layer_size):
        """Initializes the weight matrix that feeds into this layer object\n
        Takes size of previous layer as argument"""
        if self.activation_func == 'softmax':  # Softmax layer does not take any weights
            W = None
        else:
            #W = np.ones(shape=(prev_layer_size, self.size))
            W = np.random.uniform(self.w_range[0], self.w_range[1], (prev_layer_size, self.size))
        self.weights = W

    def gen_bias(self):
        """Initializes a bias vector with all values equal to 0"""
        self.bias_vector = np.zeros(self.size)

    def forward_pass(self, inputs):
        if self.l_type == 'input' or self.activation_func == 'softmax':
            x = self.activation(inputs)
        else:
            #z = np.add(np.einsum('ij,jk->ik', inputs, self.weights), np.dot(self.bias_vector, self.bias_node))
            z = np.einsum('ij,jk->ik', inputs, self.weights)
            x = self.activation(z)
        self.cached_activation = x
        return x

    def update_weights_and_bias(self):
        """Sums and averages the array of weight and bias gradients into one weight gradient and one bias gradient
        updates the weights and biases for this layer"""
        if self.l_type == 'input' or self.l_type == 'softmax':
            pass
        else:
            self.weights = self.weights - self.lr * self.weight_gradient

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

    def _softmax(self, z_array):
        # Writing out all steps to not get this calculation wrong
        z_sums = []
        for z in z_array:
            z_sums.append(np.sum(np.exp([z_i for z_i in z])))
        softmaxed = []
        for i in range(len(z_sums)):
            softmaxed.append(np.exp(z_array[i])/z_sums[i])
        softmaxed = np.array(softmaxed)
        return softmaxed

    def _d_softmax(self, activations):
        '''Creates one jacobian matrix (J_SN) for each of the activation vectors in the passed argument'''
        """The argument is an array of activation vectors of size m
        Returns an array of m x m jacobian matrices"""
        j_soft_matrices = []
        for x in activations:
            j_soft = np.zeros(shape=(len(x), len(x)))
            for i in range(len(x)):
                for j in range(len(x)):
                    j_soft[i][j] = (x[i] - x[j] ** 2) if i == j else -(x[i] * x[j])
            j_soft_matrices.append(j_soft)
        j_soft_matrices = np.array(j_soft_matrices)
        return j_soft_matrices



def main():
    l1 = FullyConnectedLayer(size=4, activation_func='softmax', l_type='softmax', optionals={}, global_lr=0.1)
    z = np.array([[1,1,2,1,1],
         [1,1,2,-1,-1]])
    # test_softmax = Layer._softmax(z)

    test_softmax = l1.activation(z)
    print(test_softmax)
    test_d_softmax = l1.derivation()
    print(test_d_softmax)


if __name__ == '__main__':
    main()