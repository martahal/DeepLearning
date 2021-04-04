from torch import nn
from Projects.Project3 import Trainer

from Projects.Project3.Encoder import Encoder
from Projects.Project3.ClassifierHead import ClassifierHead
from Projects.Project3.Trainer import Trainer
from Projects.Project3.Dataloader import load_fashion_mnist


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

def main():
    epochs = 50
    batch_size = 16
    learning_rate = 0.0002
    loss_function = 'cross_entropy'
    optimizer = 'SGD'

    #Loading FashionMNIST
    num_classes = 10
    dataloaders = load_fashion_mnist(batch_size, D1_fraction=0.8, validation_fraction=0.1, test_fraction=0.1)
    input_size = (1,28,28)
    latent_vector_size = 256

    encoder = Encoder(input_size=input_size, num_filters=4, latent_vector_size=latent_vector_size)
    classifier_head = ClassifierHead(latent_vector_size, num_classes)
    SCN_model = SCN(encoder, classifier_head, num_classes)

    SCN_trainer = Trainer(batch_size=batch_size,
                          lr=learning_rate,
                          epochs= epochs,
                          model= SCN_model,
                          dataloaders=dataloaders,
                          loss_function=loss_function,
                          optimizer=optimizer)
    SCN_trainer.do_classifier_train()

if __name__ == '__main__':
    main()