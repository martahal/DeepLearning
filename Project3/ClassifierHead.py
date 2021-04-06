from torch import nn


class ClassifierHead(nn.Module):

    def __init__(self,
                 input_size,
                 output_size):

        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(in_features=input_size,out_features=output_size),
            # Why does adding relu at the end give such a bad classification?
        )

    def forward(self, x):
        """
        Performs the forward pass of the classifier head
        :param x:  tensor, input of classifier head, shape [batch size, latent vector size]
        :return output: the latent vector of the encoder
        """
        x = self.model(x)
        output = x
        self._test_correct_output(output)
        return output

    def _test_correct_output(self, output):
        batch_size = output.shape[0]
        expected_shape = (batch_size, self.output_size)
        assert output.shape == (batch_size, self.output_size), \
            f"Expected output of forward pass to be: {expected_shape}, but got: {output.shape}"