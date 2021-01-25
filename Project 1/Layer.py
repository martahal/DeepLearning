import numpy as np


class Layer:

    def __init__(self, nodes, activation_func='sigmoid'):
        self.nodes = nodes
        self.activation_func = activation_func
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
        else:
            raise NotImplementedError("You have either misspelled the activation function, "
                                      "or this activation function is not implemented")

    def _sigmoid(self, z):
        activation = np.array([1 / (1 + np.exp(-z_i)) for z_i in z])
        self.cached_activation = activation
        return activation

    def _d_sigmoid(self, x):
        '''Inserts the cached activation to the expression of the derivative of the activation function'''
        return np.array([x_i * (1- x_i) for x_i in x])

    def _tanh(self, z):
        activation = np.array([np.tanh(z_i) for z_i in z])
        self.cached_activation = activation
        return activation

    def _d_tanh(self, x):
        return np.array([1 - x_i**2 for x_i in x])

    def _relu(self, z):
        activation = np.array([np.max(0, z_i) for z_i in z])
        self.cached_activation = activation
        return activation

    def _d_relu(self, x):
        derivative  = []
        for x_i in x:
            if x_i > 0:
                derivative.append(1)
            else:
                derivative.append(0)
        return np.array(derivative)

    def _linear(self, z):
        activation = np.array([z_i for z_i in z])
        self.cached_activation = activation
        return activation

    def _d_linear(self, x):
        return np.ones_like(x)

    def _softmax(self, z):
        pass


def main():
    l1 = Layer(nodes=2, activation_func='sigmoid')
    z = [1, 1, 1]
    x = l1.activation(z)
    print(x)
    something = l1.derivation()
    print(something)


if __name__ == '__main__':
    main()