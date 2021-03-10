import numpy as np

class ConvLayer2D:

    def __init__(self, spatial_dimensions, input_channels, output_channels, kernel_size, stride, mode, activation_function, lr):
        self.spatial_dimensions = spatial_dimensions
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.activation_function = activation_function
        self.lr = lr

        self.l_type = 'conv2d'
        self.kernels = np.zeros((self.output_channels, self.input_channels, kernel_size[0], kernel_size[1]))  # TODO
        self.filter_gradient = None  # TODO
        self.cached_activation = []
        #calculating dimensions of output
        output_dimensions = (output_channels, int(((spatial_dimensions[0] - kernel_size[0])/stride) + 1),
                             int(((spatial_dimensions[1] - kernel_size[1])/stride) + 1)) # Eventual padding and mode
        self.output_dimensions = [np.zeros(output_dimensions)]

    def gen_kernels(self):
        for i in range(self.output_channels):
            scale = 1.0
            standard_deviation = scale/np.sqrt(np.prod(self.kernel_size))
            self.kernels[i] = (np.random.normal(loc=0, scale=standard_deviation, size=self.kernel_size))
        # TODO: implement kernel generation

    def activation(self, feature_map):
        if self.activation_function == 'sigmoid':
            return self._sigmoid(feature_map)
        elif self.activation_function == 'tanh':
            return self._tanh(feature_map)
        elif self.activation_function == 'relu':
            return self._relu(feature_map)
        elif self.activation_function == 'elu':
            return self._elu(feature_map)
        elif self.activation_function == 'selu':
            return self._selu(feature_map)
        elif self.activation_function == 'linear':
            return self._linear(feature_map)
        else:
            raise NotImplementedError("You have either misspelled the activation function, "
                                      "or this activation function is not implemented")
        # TODO: implement activation after convolution
        pass

    def derivation(self):
        """Inserts the cached activation vector in the derivative function defined for this layer"""
        if self.activation_function == 'sigmoid':
            return self._d_sigmoid(self.cached_activation)
        elif self.activation_function == 'tanh':
            return self._d_tanh(self.cached_activation)
        elif self.activation_function == 'relu':
            return self._d_relu(self.cached_activation)
        elif self.activation_function == 'elu':
            return self._d_elu(self.cached_activation)
        elif self.activation_function == 'selu':
            return self._d_selu(self.cached_activation)
        elif self.activation_function == 'linear':
            return self._d_linear(self.cached_activation)

        else:
            raise NotImplementedError("You have either misspelled the activation function, "
                                      "or the derivative of this activation function is not implemented")


    def forward_pass(self, input):
        """
        :param input: shape= ((batch_size), input_channels, input_width, input_height)
        :return: activation_function(feature map), shape= ((batch_size), output_channels, output_width, output_height)
        """
        # TODO apply padding and different modes

        # TODO: implement convolution
        n_filters, n_kernels_filter, filter_size, filter_height = self.kernels.shape
        n_kernels, input_width, input_height = input.shape

        output_width = int((input_width - filter_size)/self.stride) + 1
        output_height = int((input_height - filter_size)/self.stride) + 1

        feature_map = np.zeros((n_filters, output_width, output_height))

        # TODO extend so that this works for batch_size > 1
        for current_filter_idx in range(n_filters):
            y_idx = out_y_idx = 0
            # vertical filter movement
            while y_idx +filter_size <= input_height:
                x_idx = out_x_idx = 0
                # horizontal filter movement
                while x_idx + filter_size <= input_width:
                    # Convolution operation happens here:
                    feature_map[current_filter_idx, out_y_idx, out_x_idx] = \
                        np.sum(
                        self.kernels[current_filter_idx] *
                        input[:, y_idx:y_idx + filter_size, x_idx: x_idx + filter_size]
                        )
                    # TODO cache this convolution operation in a lookup-table for easy backprop
                    x_idx += self.stride
                    out_x_idx += 1
                y_idx += self.stride
                out_y_idx += 1
        feature_map = self.activation(feature_map)
        self.cached_activation.append(feature_map) # Appending actual activation
        # Mad hack, not very pretty
        return feature_map

    def backward_pass(self, delta_jacobian, upstream_feature_map):
        # TODO: decide responsibility for backward pass method
        # TODO: implement backward pass for convolutional layer
        '''Absolutely freestyling this. Lets see if it works'''
        n_filters, n_kernels_filter, filter_width, filter_height = self.kernels.shape
        input_channels, fm_width, fm_height = upstream_feature_map.shape
#
        #Initializing derivatives
        new_jacobian = np.zeros(upstream_feature_map.shape)
        kernel_gradient = np.zeros(self.kernels.shape)
         #TODO this breaks somehow
        for current_filter_idx in range(n_filters):
            y_idx = out_y_idx = 0
            while y_idx + filter_height <= fm_height:
                x_idx = out_x_idx = 0
                while x_idx + filter_width <= fm_width: #TODO Comment on what im doing here
                    kernel_gradient[current_filter_idx] += \
                        delta_jacobian[current_filter_idx, out_y_idx, out_x_idx] * \
                        upstream_feature_map[:, y_idx: y_idx + filter_height, x_idx: x_idx+ filter_width]
##
                    new_jacobian[:, y_idx:y_idx + filter_height, x_idx: x_idx + filter_width] += \
                        delta_jacobian[current_filter_idx, out_y_idx, out_x_idx] * \
                        self.kernels[current_filter_idx]
                    x_idx += self.stride
                    out_x_idx += 1
                y_idx += self.stride
                out_y_idx += 1

        return new_jacobian, kernel_gradient # TODO New jacobian must be derived

    def update_filters(self):
        self.kernels = self.kernels - self.lr * self.filter_gradient

    def visualize_kernels(self):
        # TODO: implement visualization of kernels
        pass

    def _sigmoid(self, feature_map):
        activation = np.vectorize(lambda fm: 1 / (1 + np.exp(-fm)))(feature_map)
        return activation

    def _d_sigmoid(self, activation):
        d_activation = np.vectorize(lambda a: a * (1 - a))(activation)
        return d_activation

    def _tanh(self, feature_map):
        activation = np.vectorize(np.tanh)(feature_map)
        return activation

    def _d_tanh(self, activation):
        d_activation = np.vectorize(lambda a: 1 - a**2)(activation)

    def _relu(self, feature_map):
        activation = np.copy(feature_map)
        activation[activation < 0] = 0
        return activation

    def _d_relu(self, activation):
        d_activation = np.copy(activation)
        d_activation[d_activation <= 0] = 0
        d_activation[d_activation > 0] = 1
        return d_activation

    def _elu(self, feature_map):
        alpha = 1.0
        activation = np.vectorize(lambda fm: alpha*(np.exp(fm) - 1) if fm <= 0 else fm)(feature_map)
        return activation

    def _d_elu(self, activation):
        alpha = 1.0
        d_activation = np.vectorize(lambda a: alpha * (np.exp(a)) if a < 0 else 1)(activation)
        return d_activation

    def _selu(self, feature_map):
        alpha = 1.6732632423543772848170429916717
        lmb = 1.0507009873554804934193349852946
        activation = np.vectorize(lambda fm: lmb * (alpha * np.exp(fm) - alpha) if fm <= 0 else lmb * fm)(feature_map)
        return activation

    def _d_selu(self,activation):
        alpha = 1.6732632423543772848170429916717
        lmb = 1.0507009873554804934193349852946
        d_activation = np.vectorize(lambda a: lmb * (alpha * np.exp(a)) if a <=0 else lmb)(activation)
        return d_activation


    def _linear(self, feature_map):
        return feature_map

    def _d_linear(self,activation):
        return np.ones_like(activation)




def main():
    raw_image = np.zeros((7, 7))
    spec = {'spatial_dimensions':raw_image.shape,'input_channels': 1, 'output_channels': 4,'kernel_size': (3, 3),
              'stride': 1, 'mode': 'same', 'act_func': 'relu', 'lr': 0.01, 'type': 'conv2d'}

    test_layer = ConvLayer2D(
        spec['spatial_dimensions'],
        spec['input_channels'],
        spec['output_channels'],
        spec['kernel_size'],
        spec['stride'],
        spec['mode'],
        spec['act_func'],
        spec['lr'])

    test_layer.gen_kernels()

    #raw_image = np.zeros((5,5))
    raw_image[0] = np.ones_like(raw_image[0])
    test_image = np.array([raw_image])
    print(test_image)

    feature_map = test_layer.forward_pass(test_image)
    print(feature_map)



if __name__ == '__main__':
    main()