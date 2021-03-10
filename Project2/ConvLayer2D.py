import numpy as np

class ConvLayer2D:

    def __init__(self, input_channels, output_channels, kernel_size, stride, mode, activation_function, lr):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.activation_function = activation_function
        self.lr = lr

        self.l_type = 'conv2d'
        self.kernels = np.zeros((self.output_channels, self.input_channels, kernel_size[0], kernel_size[1]))  # TODO
        self.weight_gradients = None  # TODO
        self.cached_activation = None

    def gen_kernels(self):
        for i in range(self.output_channels):
            scale = 1.0
            standard_deviation = scale/np.sqrt(np.prod(self.kernel_size))
            self.kernels[i] = (np.random.normal(loc=0, scale=standard_deviation, size=self.kernel_size))
        # TODO: implement kernel generation

    def activation(self, feature_map):
        # TODO: implement activation after convolution
        pass

    def derivation(self):
        # TODO: implement derivation for backprop
        pass

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
        # TODO feature_map = activation(feature_map)
        self.cached_activation = feature_map
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
#
        return new_jacobian, kernel_gradient # TODO New jacobian must be derived


    def visualize_kernels(self):
        # TODO: implement visualization of kernels
        pass

def main():

    spec = {'input_channels': 1, 'output_channels': 4,'kernel_size': (3, 3),
              'stride': 1, 'mode': 'same', 'act_func': 'relu', 'lr': 0.01, 'type': 'conv2d'}

    test_layer = ConvLayer2D(spec['input_channels'],
                                     spec['output_channels'],
                                     spec['kernel_size'],
                                     spec['stride'],
                                     spec['mode'],
                                     spec['act_func'],
                                     spec['lr'])
    test_layer.gen_kernels()

    raw_image = np.zeros((5,5))
    raw_image[0] = np.array([1,1,1,1,1])
    test_image = np.array([raw_image])
    print(test_image)

    feature_map = test_layer.forward_pass(test_image)
    print(feature_map)



if __name__ == '__main__':
    main()