from Projects.Project3 import visualisations
import matplotlib.pyplot as plt


def make_reconstructions_figure(autoencoder, vis_data, num_images, batch_size, image_dimensions):
    # Extremely inefficient way of doing this
    # Forward all images, then selecting the ones i want to visualize
    images = []
    reconstructions = []
    labels = []
    for X_batch, Y_batch in vis_data:
        images.extend(X_batch)
        labels.extend(Y_batch)
        reconstruction_batch, aux = autoencoder(X_batch)
        reconstruction_batch = reconstruction_batch.view(batch_size, image_dimensions[0], image_dimensions[1],
                                                         image_dimensions[2])
        reconstruction_batch = reconstruction_batch.detach().numpy()
        reconstructions.extend(reconstruction_batch)
    images = images[:num_images]
    reconstructions = reconstructions[:num_images]
    labels = labels[:num_images]

    visualisations.show_images_and_reconstructions(images, reconstructions, labels)
    plt.show()