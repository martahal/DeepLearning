from Project4.VAE import VAE
from Project4.VAE_Encoder import Encoder
from Project4.VAE_Decoder import Decoder
from Project4.verification_net import VerificationNet
from Project3 import visualisations
from Trainer import Trainer
from Project4.stacked_mnist import StackedMNISTData, DataMode
import utils
from var_ae_routine import VAE_Routine
import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    torch.manual_seed(1)
    """ GENERATIVE VAE ROUTINE"""
    batch_size = 16
    data_object = StackedMNISTData(
        mode=DataMode.MONO_FLOAT_COMPLETE,
        default_batch_size=batch_size)
    net = VerificationNet(force_learn=False)
    net.train(
        generator=data_object,
        epochs=5)  # gen=data_object, makes sure we test on the same type of data as the model was trained on
    verification_tolerance = 0.8 if data_object.channels == 1 else 0.5

    learning_rate = 1.0e-2
    loss_function = 'elbo'
    optimizer = 'adam'
    epochs = 1

    latent_vector_size = 128
    num_samples = 2000
    gen_name = 'Test_demo_gen_VAE'
    gen_vae_save_path = f'checkpoints/gen_VAE/{gen_name}'
    vae_routine = VAE_Routine(
        data_object,
        learning_rate,
        loss_function,
        optimizer,
        epochs,
        #
        latent_vector_size,
        batch_size,
        num_samples,
        gen_vae_save_path
    )
    #vae_routine.train_vae()
    #images, reconstructions, labels = vae_routine.reconstruct_test_data(load_model_path=gen_vae_save_path)
    vae_routine.anomaly_detection(10, load_model_path=gen_vae_save_path)
if __name__ == '__main__':
    main()