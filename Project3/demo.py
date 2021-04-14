from Projects.Project3.SSN import SSN
from Projects.Project3.SCN import SCN
from Projects.Project3.utils import get_and_split_dataset, compare_SSN_and_SCN

def main():
    # Global parameters for both SSN and SCN
    dataset_name = 'MNIST'  # AVAILABLE  #'CIFAR10' # 'KMNIST' #'FashionMNIST' #
    D1_fraction = 0.8
    D2_train_val_test_fraction = (0.1, 0.1)

    latent_vector_size = 64
    batch_size = 16

    # Parameters for Autoencoder training
    autoencoder_learning_rate = 0.2
    autoencoder_loss_function = 'binary_cross_entropy'  # AVAILABLE 'binary_cross_entropy'
    autoencoder_optimizer = 'SGD'  # AVAILABLE 'SGD' #'adam' #
    autoencoder_epochs = 3 # Optimal for MNIST: 3

    freeze_encoder_weights = False
    plot_t_sne = True
    n_reconstructions_to_display = 10

    # Parameters for classifier training
    classifier_learning_rate = 0.002
    classifier_loss_function = 'cross_entropy'  # AVAILABLE 'cross_entropy'
    classifier_optimizer = 'SGD'  # AVAILABLE 'SGD' #'adam'
    classifier_epochs = 10

    dataloaders, \
        image_dimensions,\
        num_classes \
        = get_and_split_dataset(dataset_name, D1_fraction, D2_train_val_test_fraction, batch_size)

    SSN_approach = SSN(
        dataloaders,
        autoencoder_learning_rate,
        autoencoder_loss_function,
        autoencoder_optimizer,
        autoencoder_epochs,

        classifier_learning_rate,
        classifier_loss_function,
        classifier_optimizer,
        classifier_epochs,

        latent_vector_size,
        batch_size,
        num_classes,
        image_dimensions,

        freeze_encoder_weights,
        n_reconstructions_to_display,
        plot_t_sne
    )

    #SSN_approach.run_SSN_training_regime()

    SCN_approach = SCN(
        dataloaders,
        classifier_learning_rate,
        classifier_loss_function,
        classifier_optimizer,
        latent_vector_size,
        classifier_epochs,
        batch_size,
        image_dimensions,
        num_classes
    )

    SCN_approach.run_SCN_training_regime()

    #compare_SSN_and_SCN(SSN_approach.SSN_trainer, SCN_approach.SCN_trainer)

if __name__ == '__main__':
    main()