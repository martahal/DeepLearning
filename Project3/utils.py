from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from Projects.Project3.Autoencoder import Autoencoder
from Projects.Project3.Classifier import Classifier
from Projects.Project3.Encoder import Encoder
from Projects.Project3.Decoder import Decoder
from Projects.Project3.ClassifierHead import ClassifierHead
from Projects.Project3.Trainer import Trainer
from Projects.Project3.Dataloader import load_fashion_mnist, load_mnist, load_cifar10, load_kmnist
from Projects.Project3 import visualisations


def get_and_split_dataset(
        dataset_name,
        D1_fraction: float,
        D2_train_val_test_fraction: tuple,
        batch_size
):
    if dataset_name == 'FashionMNIST':
        #Loading FashionMNIST
        num_classes = 10
        dataloaders = load_fashion_mnist(
            batch_size=batch_size,
            D1_fraction=D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[0]
        )
        image_dimensions = (1, 28, 28)

    elif dataset_name == 'MNIST':
        # Loading MNIST
        num_classes = 10
        image_dimensions = (1, 28, 28)

        dataloaders = load_mnist(
            batch_size=batch_size,
            D1_fraction= D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[1]
        )
    elif dataset_name == 'CIFAR10':
        # Loading CIFAR10
        num_classes = 10
        image_dimensions = (3, 28, 28)

        dataloaders = load_cifar10(
            batch_size=batch_size,
            D1_fraction=D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[1]
        )
    elif dataset_name == 'KMNIST':
        num_classes = 10
        image_dimensions = (1, 28, 28)

        dataloaders = load_kmnist(
            batch_size=batch_size,
            D1_fraction=D1_fraction,
            validation_fraction=D2_train_val_test_fraction[0],
            test_fraction=D2_train_val_test_fraction[1]
        )

    else:
        raise NotImplementedError('Dataset not implemented yet')

    return dataloaders, image_dimensions,  num_classes

def compare_SSN_and_SCN(SSN_trainer, SCN_trainer):

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title('SCN and SSN loss')
    visualisations.plot_metric(SSN_trainer.train_history['loss'], label='SSN Training loss', averaged_plot=True)
    visualisations.plot_metric(SSN_trainer.validation_history['loss'], label='SSN Validation loss', averaged_plot=False)
    visualisations.plot_metric(SCN_trainer.train_history['loss'], label='SCN Training loss', averaged_plot=True)
    visualisations.plot_metric(SCN_trainer.validation_history['loss'], label=' SCN Validation loss', averaged_plot=False)
    plt.legend()
    plt.ylim(bottom=0, top=2.5)

    plt.subplot(1, 2, 2)
    plt.title('SCN and SSN accuracy')
    visualisations.plot_metric(SSN_trainer.train_history['accuracy'], label='SSN Training accuracy', averaged_plot=True)
    visualisations.plot_metric(SSN_trainer.validation_history['accuracy'], label='SSN Validation accuracy', averaged_plot=False)
    visualisations.plot_metric(SCN_trainer.train_history['accuracy'], label='SCN Training accuracy', averaged_plot=True)
    visualisations.plot_metric(SCN_trainer.validation_history['accuracy'], label='SCN Validation accuracy', averaged_plot=False)
    plt.legend()

    plt.ylim(bottom=0, top=1)
    plt.show()
