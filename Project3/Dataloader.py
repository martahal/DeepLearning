from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
np.random.seed(0)




def load_fashion_mnist(batch_size :int, D1_fraction: float = 0.8, validation_fraction: float = 0.1, test_fraction: float = 0.1)\
        -> typing.List[torch.utils.data.DataLoader]:
    """
    Loads the MNIST dataset and partitions it into two training sets D1, and D2 and a test set, and the respective
    train and validation set for both D1 and D2.
    Also enable data transformation.
    :param batch_size: int, batch size
    :param D1_fraction: float, fraction of dataset wich will be used to train autoencoder (unlabelled training)
    :param validation_fraction: float, fraction of training data which will be used for validation. Applies for both D1 and D2
    :return: List of torch.utils.data.DataLoader, D1_train_dataloader, D2_train_dataloader, D2_val_dataloader, D2_test_dataloader
    """
    # TODO resolve whether this returns list or tuple

    fashion_mnist_mean = (0.5,)
    fashion_mnist_std = (0.25,)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(fashion_mnist_mean, fashion_mnist_std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(fashion_mnist_mean, fashion_mnist_std)
    ])

    data_train = datasets.FashionMNIST('data',
                                train=True,
                                download=True,
                                transform=transform_train)

    data_test = datasets.FashionMNIST('data',
                               train=False,
                               download=True,
                               transform=transform_test)

    d1_train_dataloader,\
    d2_train_dataloader,\
    d2_val_dataloader,\
    d2_test_dataloader = split_dataset(
        data_train,
        data_test,
        batch_size,
        D1_fraction,
        validation_fraction,
        test_fraction
    )
    return d1_train_dataloader, d2_train_dataloader, d2_val_dataloader, d2_test_dataloader


def load_mnist(batch_size :int, D1_fraction: float = 0.8, validation_fraction: float = 0.1, test_fraction: float = 0.1)\
        -> typing.List[torch.utils.data.DataLoader]:
    mnist_mean = (0.5,)
    mnist_std = (0.25,)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])
    data_train = datasets.MNIST('data',
                                train=True,
                                download=True,
                                transform=transform_train)

    data_test = datasets.MNIST('data',
                               train=False,
                               download=True,
                               transform=transform_test)

    d1_train_dataloader,\
    d2_train_dataloader,\
    d2_val_dataloader,\
    d2_test_dataloader = split_dataset(
        data_train,
        data_test,
        batch_size,
        D1_fraction,
        validation_fraction,
        test_fraction
    )
    return d1_train_dataloader, d2_train_dataloader, d2_val_dataloader, d2_test_dataloader


def load_cifar10(batch_size :int, D1_fraction: float = 0.8, validation_fraction: float = 0.1, test_fraction: float = 0.1)\
        -> typing.List[torch.utils.data.DataLoader]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.247, 0.243, 0.261)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        #transforms.RandomAffine(degrees=15, scale=(0.5,2))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)
    d1_train_dataloader, \
    d2_train_dataloader, \
    d2_val_dataloader, \
    d2_test_dataloader = split_dataset(
        data_train,
        data_test,
        batch_size,
        D1_fraction,
        validation_fraction,
        test_fraction
    )
    return d1_train_dataloader, d2_train_dataloader, d2_val_dataloader, d2_test_dataloader

def load_kmnist(batch_size :int, D1_fraction: float = 0.8, validation_fraction: float = 0.1, test_fraction: float = 0.1)\
        -> typing.List[torch.utils.data.DataLoader]:
    kmnist_mean = (0.5,)
    kmnist_std = (0.25,)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(kmnist_mean, kmnist_std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(kmnist_mean, kmnist_std)
    ])
    data_train = datasets.KMNIST('data',
                                train=True,
                                download=True,
                                transform=transform_train)

    data_test = datasets.KMNIST('data',
                               train=False,
                               download=True,
                               transform=transform_test)

    d1_train_dataloader, \
    d2_train_dataloader, \
    d2_val_dataloader, \
    d2_test_dataloader = split_dataset(
        data_train,
        data_test,
        batch_size,
        D1_fraction,
        validation_fraction,
        test_fraction
    )
    return d1_train_dataloader, d2_train_dataloader, d2_val_dataloader, d2_test_dataloader

def split_dataset(data_train, data_test, batch_size, D1_fraction: float = 0.8, validation_fraction: float = 0.1, test_fraction: float = 0.1):
    # Split dataset into D1 and D2
    indices = list(range(len(data_train)))
    d1_split_idx = int(np.floor(D1_fraction * len(data_train)))
    d1_indices = np.random.choice(indices, size=d1_split_idx, replace=False)
    d2_indices = list(set(indices) - set(d1_indices))

    # Split D2 into training, validation and test set
    d2_val_split_idx = int(np.floor(validation_fraction * len(d2_indices)))
    d2_test_split_idx= int(np.floor(test_fraction * len(d2_indices)))
    d2_val_indices = np.random.choice(d2_indices, size=d2_val_split_idx, replace=False)
    d2_train_indices = list(set(d2_indices) - set(d2_val_indices))
    d2_test_indices = np.random.choice(d2_train_indices, size=d2_test_split_idx, replace=False)
    d2_train_indices = list(set(d2_train_indices) - set(d2_test_indices))

    # quickfix to take test data from test data partition provided by torchvision.datasets
    d2_test_indices = np.array([x for x in range(len(d2_test_indices))])

    d1_train_sampler = SubsetRandomSampler(d1_indices)
    d2_train_sampler = SubsetRandomSampler(d2_train_indices)
    d2_val_sampler = SubsetRandomSampler(d2_val_indices)
    d2_test_sampler = SubsetRandomSampler(d2_test_indices)

    d1_train_dataloader = torch.utils.data.DataLoader(data_train,
                                                      sampler= d1_train_sampler,
                                                      batch_size=batch_size,
                                                      drop_last=True)
    d2_train_dataloader = torch.utils.data.DataLoader(data_train,
                                                      sampler=d2_train_sampler,
                                                      batch_size=batch_size,
                                                      drop_last=True)
    d2_val_dataloader = torch.utils.data.DataLoader(data_train,
                                                    sampler=d2_val_sampler,
                                                    batch_size=batch_size,
                                                    drop_last=True)
    d2_test_dataloader = torch.utils.data.DataLoader(data_test,
                                                     sampler=d2_test_sampler,
                                                     batch_size=batch_size,
                                                     shuffle=False)

    return d1_train_dataloader, d2_train_dataloader, d2_val_dataloader, d2_test_dataloader





