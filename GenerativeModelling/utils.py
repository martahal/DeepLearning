from SemiSupervisedLearning import visualisations
import matplotlib.pyplot as plt
import numpy as np
import torch
import pathlib




def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements

def get_data_to_tensors(data, batch_size):
    train_data_subsample, test_data = [], []
    for (images, classes) in data.batch_generator(training=True, batch_size=batch_size):
        images = normalize_images(images)
        images, classes = torch.from_numpy(images).float(), torch.from_numpy(classes).float()
        images = images.permute(0, 3, 1, 2)  # change axis from NHWC to NCHW
        batch = (images, classes)
        train_data_subsample.append(batch)

    for (images, classes) in data.batch_generator(training=False, batch_size=batch_size):
        images = normalize_images(images)
        images, classes = torch.from_numpy(images).float(), torch.from_numpy(classes).float()
        images = images.permute(0, 3, 1, 2)  # change axis from NHWC to NCHW
        batch = (images, classes)
        test_data.append(batch)
    return (train_data_subsample, test_data)

def normalize_images(images):
    # Assuming pixel values are more or less the same for all images
    # We pick the first image of the batch
    image = images[0]
    pixels = np.asarray(image)
    means = pixels.mean(axis=(0, 1), dtype='float64')
    stds = pixels.std(axis=(0, 1), dtype='float64')
    pixels = (pixels - means) / stds
    # Apply normalization to all images in the batch
    norm_images = []
    for i in range(len(images)):
        norm_images.append((images[i] - means) / stds)
    norm_images = np.array(norm_images)
    return norm_images


def make_reconstructions(autoencoder, vis_data, num_images, batch_size, image_dimensions, title):
    # Extremely inefficient way of doing this
    # Forward all images, then selecting the ones i want to visualize
    images = []
    reconstructions = []
    labels = []
    for image_batch, label in vis_data:
        #Make reconstruction
        image_batch = to_cuda(image_batch)
        reconstruction_batch, aux = autoencoder(image_batch)
        # Convert from tensor to numpy
        image_batch = image_batch.reshape(
            image_batch.shape[0],
            image_batch.shape[2],
            image_batch.shape[3],
            image_batch.shape[1]
        )
        image_batch = image_batch.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        reconstruction_batch = reconstruction_batch.reshape(
            reconstruction_batch.shape[0],
            reconstruction_batch.shape[2],
            reconstruction_batch.shape[3],
            reconstruction_batch.shape[1]
        )
        reconstruction_batch = reconstruction_batch.cpu().detach().numpy()
        images.extend(image_batch)
        labels.extend(label)
        reconstructions.extend(reconstruction_batch)
    vis_images = images[1000: 1000 + num_images]
    vis_reconstructions = reconstructions[1000: 1000 +num_images]
    vis_labels = labels[1000: 1000 + num_images]
#
    visualisations.show_images_and_reconstructions(np.array(vis_images), title, vis_labels)
    visualisations.show_images_and_reconstructions(np.array(vis_reconstructions),
                                                   f'{title}_reconstructions', vis_labels)
    return np.array(images), np.array(reconstructions), np.array(labels)


def make_vae_reconstructions(vae, vis_data, num_images, batch_size, image_dimensions, title):
    images = []
    reconstructions = []
    labels = []
    for image_batch, label in vis_data:
        # Make reconstruction
        image_batch = to_cuda(image_batch)
        reconstruction_batch, aux1, aux2, aux_3 = vae(image_batch)
        # Convert from tensor to numpy
        image_batch = image_batch.reshape(
            image_batch.shape[0],
            image_batch.shape[2],
            image_batch.shape[3],
            image_batch.shape[1]
        )
        image_batch = image_batch.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        reconstruction_batch = reconstruction_batch.reshape(
            reconstruction_batch.shape[0],
            reconstruction_batch.shape[2],
            reconstruction_batch.shape[3],
            reconstruction_batch.shape[1]
        )
        reconstruction_batch = reconstruction_batch.cpu().detach().numpy()
        images.extend(image_batch)
        labels.extend(label)
        reconstructions.extend(reconstruction_batch)
    vis_images = images[1000: 1000 +num_images]
    vis_reconstructions = reconstructions[1000: 1000 +num_images]
    vis_labels = labels[1000: 1000 +num_images]

    visualisations.show_images_and_reconstructions(np.array(vis_images), title, vis_labels)
    visualisations.show_images_and_reconstructions(np.array(vis_reconstructions),
                                                   f'{title}_reconstructions', vis_labels)
    return np.array(images), np.array(reconstructions), np.array(labels)

def generate_images_from_Z(Z, decoder, image_dimensions, title):
    #Z = torch.from_numpy(Z).float()
    # Transfer to GPU if available
    Z = to_cuda(Z)
    decoder = to_cuda(decoder)
    # generate fake images
    generated_images = decoder(Z)
    generated_images = generated_images.view(
        Z.shape[0],
        image_dimensions[1],
        image_dimensions[2],
        image_dimensions[0],
    )
    generated_images = generated_images.cpu().detach().numpy()
    labels = None
    number_to_vis = 25
    visualisations.show_vae_generated_img(generated_images[:number_to_vis], title=title)
    return generated_images

