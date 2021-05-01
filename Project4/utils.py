from Project3 import visualisations
import matplotlib.pyplot as plt
import numpy as np
import torch




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


def make_reconstructions_figure(autoencoder, vis_data, num_images, batch_size, image_dimensions):
    # Extremely inefficient way of doing this
    # Forward all images, then selecting the ones i want to visualize
    images = []
    reconstructions = []
    labels = []
    for image, label in vis_data:
        #Make reconstruction
        reconstruction_batch, aux = autoencoder(image)
        # Convert from tensor to numpy
        image = image.view(batch_size, image_dimensions[1], image_dimensions[2], image_dimensions[0])
        image = image.detach().numpy()
        label = label.detach().numpy()
        reconstruction_batch = reconstruction_batch.view(batch_size, image_dimensions[1], image_dimensions[2], image_dimensions[0])
        reconstruction_batch = reconstruction_batch.detach().numpy()
        images.extend(image)
        labels.extend(int(label))
        reconstructions.extend(reconstruction_batch)
    images = images[:num_images]
    reconstructions = reconstructions[:num_images]
    labels = labels[:num_images]
#
    visualisations.show_images_and_reconstructions(np.array(images), labels)
    visualisations.show_images_and_reconstructions(np.array(reconstructions), labels)
    plt.show()

def generate_images_from_Z(Z, decoder, image_dimensions):
    Z = torch.from_numpy(Z).float()
    generated_images = decoder(Z)
    generated_images = generated_images.view(
        Z.shape[0],
        image_dimensions[1],
        image_dimensions[2],
        image_dimensions[0],
    )
    generated_images = generated_images.detach().numpy()
    labels = None
    visualisations.show_images_and_reconstructions(generated_images, labels)
    plt.show()


