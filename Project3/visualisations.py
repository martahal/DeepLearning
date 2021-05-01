import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd


def plot_metric(metric_dict, label, averaged_plot=True, n=8):
    global_steps = list(metric_dict.keys())
    metric = list(metric_dict.values())

    if not averaged_plot:
        plt.plot(global_steps, np.array(metric), label=label)
    else:
        num_points = len(metric) // n
        mean_values = []
        std_values = []
        steps = []
        for i in range(num_points):
            values = metric[i * n: (i+1) * n]
            step = global_steps[i * n + n // 2]
            mean_values.append(np.mean([float(value) for value in values])) # ugly fix because metric is list of ugly tensor objects or whatever
            std_values.append(np.std([float(value) for value in values]))
            steps.append(step)
        mean_values = np.array(mean_values)
        std_values = np.array(std_values)
        plt.plot(steps, mean_values, label=f'{label} averaged over {n} points')
        plt.fill_between(steps, mean_values - std_values,
                         mean_values + std_values,
                         alpha=0.3, label= f'{label} variance over {n} points')


def show_images_and_reconstructions(images, labels, title):
    """
    Plot data in RGB (3-channel data) or monochrome (one-channel data).
    If data is submitted, we need to generate an example.
    If there are many images, do a subplot-thing.
    """

    # Just a little hacky check to not make large modifications
    if isinstance(images, list):
        no_images = len(images)
        no_channels = images[0].shape[0]
    else:
        no_images = images.shape[0]
        no_channels = images.shape[-1]

    # Do the plotting
    plt.Figure()
    no_rows = np.ceil(np.sqrt(no_images))
    no_cols = np.ceil(no_images / no_rows)
    for img_idx in range(no_images):
        plt.subplot(int(no_rows), int(no_cols), int(img_idx + 1))
        if no_channels == 1:
            plt.imshow(images[img_idx, :, :, 0], cmap="binary")
        else:
            plt.imshow(images[img_idx, :, :, :].astype(np.float))
        plt.xticks([])
        plt.yticks([])
        if labels is not None:
            plt.title(f"Class is {str(int(labels[img_idx])).zfill(no_channels)}")
    plt.savefig(f'figures/{title}')
    # Show the thing ...
    #plt.show()


def plot_t_sne(latent_vectors_and_classes: tuple):
    latent_vectors = latent_vectors_and_classes[0]
    classes = latent_vectors_and_classes[1]
    latent_vectors_embedded = TSNE(perplexity=50, learning_rate=100).fit_transform(latent_vectors)

    # create the 'data table' to show in the scatterplot
    d = {'x': latent_vectors_embedded[:, 0], 'y': latent_vectors_embedded[:, 1], 'classes': classes}
    df = pd.DataFrame(data=d)
    sns.scatterplot(data=df, x='x', y='y', hue='classes', palette='deep')

def main():
    values = np.random.uniform(0,2, 6000)
    test_dict = {}
    for i in range(len(values)):
        test_dict[i] = values[i]

    plot_metric(test_dict, 'Test_plot')


if __name__ == '__main__':
    main()
