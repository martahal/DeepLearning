from torch import nn
from Projects.Project3 import Trainer


class SCN(nn.Module):

    def __init__(self,
                 encoder,
                 classifier_head,
                 num_classes):
        """
        Constructs the Supervised Classifier Network
        :param encoder: nn.Module, the encoder of the semi-supervised network (not trained)
        :param classifier_head: nn.Module, the classifier head used for classifying images
        :param num_classes: int, number of classes to identify
        """
        super().__init__()

        self.encoder = encoder
        self.classifier_head = classifier_head

        self.num_classes = num_classes

    def forward(self, x):
        """
        Performs the forward pass through the SCN

        :param x: Input image, shape: [batch_size, image channels, width, height]
        :return:  output of network, latent vector.
        """
        latent_vector = self.encoder(x)
        output = self.classifier_head(latent_vector)

        self._test_correct_output(output)

        return output, latent_vector

    def _test_correct_output(self, output):
        batch_size = output.shape[0]
        expected_shape = (batch_size, self.num_classes)
        assert output.shape == (batch_size, self.num_classes), \
            f"Expected output of forward pass to be: {expected_shape}, but got: {output.shape}"
