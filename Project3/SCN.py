from Projects.Project3.Autoencoder import Autoencoder
from Projects.Project3.Classifier import Classifier
from Projects.Project3.Encoder import Encoder
from Projects.Project3.Decoder import Decoder
from Projects.Project3.Trainer import Trainer
from Projects.Project3.ClassifierHead import ClassifierHead
from Projects.Project3 import visualisations

import matplotlib.pyplot as plt
from torch import nn

class SCN:

    def __init__(
            self,
            dataloaders,
            learning_rate: float,
            loss_function: str,
            optimizer: str,
            latent_vector_size: int,
            epochs: int,
            batch_size: int,
            image_dimensions: tuple,
            num_classes: int
    ):
        self.dataloaders = dataloaders

        self.encoder = Encoder(
            input_shape=image_dimensions,
            num_filters=16,
            last_layer_dim=(32, 10, 10),
            latent_vector_size=latent_vector_size)

        self.classifier_head = ClassifierHead(
            latent_vector_size,
            num_classes)

        self.classifier = Classifier(
            self.encoder,
            self.classifier_head,
            num_classes)

        self.SCN_trainer = Trainer(
            batch_size=batch_size,
            lr=learning_rate,
            epochs=epochs,
            model=self.classifier,
            dataloaders=dataloaders,
            loss_function=loss_function,
            optimizer=optimizer)

    def run_SCN_training_routine(self):
        self.SCN_trainer.do_classifier_train()
        #return self.SCN_trainer
