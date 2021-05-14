# DeepLearning
This repository contains four projects that correspond to assignments in the course Deep Learning at NTNU. (course code left out on purpose)
The code is structured to solve these assignments and not necessarily be ready to run out of the box although demonstrations can be done without too much plundering.

Below is a short description of each project. 

### Fully Connected Neural Network
This project is an implementation of a fully connected neural network from scratch using numpy. In addition to this we generate our own (image) data wich the network is trained on.

**To run:** Run the `Configurator.py` script with either of the `demo.txt` files. The `demo.txt` files are configuration files where network architecture and hyperparameters can be configured

### Convolutional Neural Network
This project is an implementation of a convolutional neural network from scracth using numpy. We use the same data generation technique as in the FC neural network

**To run:** Same as for the FC neural network project.

### Semi-supervised Learning
This is an implementation of an autoencoder and a semi supervised training routine using PyTorch. We compare a semi supervised learning approach to a purely supervised approach.
The comparison can be done with four different common datasets; CIFAR10, MNIST, FashionMNIST and KMNIST. 

**To run:** Configure which dataset to run and hyperparameters in `demo.py` and run it. Note that this is a bit computationally heavy, and is not configured to run on GPUs so keep the number of epochs low unless you have a really powerful computer. You should be able to obtain ~96% accuracy without melting your computer.

### Generative modelling
Here is an implementation of a regular autoencoder and a variational autoencoder both as generative models and anomaly detectors using PyTorch trained on MNIST. The models are configured to be able to run on a GPU if you have one available. 

**To run:** Easiest way is to run either of the demo notebooks. VAE stands for variational autoencoder and AE for Autoencoder. Note that this should be done on a GPU to achieve some ok results. 
