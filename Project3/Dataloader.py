from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
np.random.seed(0)

mean = (0.5, 0.5, 0.5)
std = (.25, .25, .25)


def load_mnist(batch_size :int, D1_fraction: float = 0.8, validation_fraction: float = 0.1):
    """
    Loads the MNIST dataset and partitions it into two training sets D1, and D2 and a test set, and the respective
    train and validation set for both D1 and D2.
    Also enable data transformation.
    :param batch_size: int, batch size
    :param D1_fraction: float, fraction of dataset wich will be used to train autoencoder (unlabelled training)
    :param validation_fraction: float, fraction of training data which will be used for validation. Applies for both D1 and D2
    :return: tuple of torch.utils.data.DataLoader, D1_train_dataloader, D1_val_dataloader, D2_train_dataloader, D2_val_dataloader, test_dataloader
    """

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    data_train = datasets.MNIST('data/MNIST',
                              train=True,
                              download=True,
                              transforms=transform_train)

    data_test = datasets.MNIST('data/MNIST',
                              train=False,
                              download=True,
                              transforms=transform_test)

    indices = list(range(len(data_train)))
    d1_split_idx = int(np.floor(D1_fraction * len(data_train)))
    d1_indices = np.random.choice(indices, size=d1_split_idx, replace=False)
    d2_indices = list(set(indices) - set(d1_indices))

    d1_val_split_idx = int(np.floor(validation_fraction * len(d1_indices)))
    d1_val_indices = np.random.choice(d1_indices, size=d1_val_split_idx, replace=False)
    d1_train_indices = list(set(d1_indices) - set(d1_val_indices))

    d2_val_split_idx = int(np.floor(validation_fraction * len(d2_indices)))
    d2_val_indices = np.random.choice(d2_indices, size=d2_val_split_idx, replace=False)
    d2_train_indices = list(set(d2_indices) - set(d2_val_indices))

    d1_train_sampler = SubsetRandomSampler(d1_train_indices)
    d1_val_sampler = SubsetRandomSampler(d1_val_indices)
    d2_train_sampler = SubsetRandomSampler(d2_train_indices)
    d2_val_sampler = SubsetRandomSampler(d2_val_indices)

    d1_train_dataloader = torch.utils.data.DataLoader(data_train,
                                           sampler= d1_train_sampler,
                                           batch_size=batch_size,
                                           drop_last=True)
    d1_val_dataloader = torch.utils.data.DataLoader(data_train,
                                         sampler=d1_val_sampler,
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
    test_dataloader = torch.utils.data.DataLoader(data_test,
                                       batch_size=batch_size,
                                       shuffle=False)

    return d1_train_dataloader, d1_val_dataloader, d2_train_dataloader, d2_val_dataloader, test_dataloader


def load_cifar10(batch_size: int, validation_fraction: float = 0.1
                 ): # TODO remove  #-> typing.List[torch.utils.data.DataLoader]:
    # Note that transform train will apply the same transform for
    # validation!
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

    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_val, dataloader_test
