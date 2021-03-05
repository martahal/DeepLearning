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

        self.kernel_weights = None  # TODO
        self.weight_gradients = None  # TODO
        self.cached_activation = None

    def gen_kernels(self):
        self.kernel_weights = 'Something'
        # TODO: implement kernel generation
        pass

    def activation(self):
        # TODO: implement activation after convolution
        pass

    def derivation(self):
        # TODO: implement derivation for backprop
        pass

    def forward_pass(self):
        # TODO: implement convolution
        pass

    def backward_pass(self):
        # TODO: decide responsibility for backward pass method
        # TODO: implement backward pass for convolutional layer
        pass

    def visualize_kernels(self):
        # TODO: implement visualization of kernels
        pass

