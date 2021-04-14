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


def show_images_and_reconstructions(images, reconstructions, labels):
    # helper function
    num_images = len(images)
    def imshow(image):
        # unnormalize image
        image = image / 2 + 0.5
        #convert from Tensor image
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)
    '''
    n_rows = np.ceil
    '''

    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(num_images/2)), figsize=(num_images, 4))#, sharex=True, sharey=True)#, figsize=(num_images + 4, 4))
    fig.tight_layout()
    for idx in np.arange(num_images):
        ax = fig.add_subplot(2, int(np.ceil(num_images/2)), idx+1, xticks=[], yticks= [], frame_on= False)
        imshow(reconstructions[idx])
        #ax.set_title(labels[idx])

    fig, axes = plt.subplots(nrows=2, ncols=int(np.ceil(num_images/2)), figsize=(num_images, 4))#, sharex=True, sharey=True)#, figsize=(num_images + 4, 4))
    fig.tight_layout()
    for idx in np.arange(num_images):
        ax = fig.add_subplot(2, int(np.ceil(num_images/2)), idx+1, xticks=[], yticks= [], frame_on= False)
        imshow(images[idx])
        #ax.set_title(labels[idx])



def plot_t_sne(latent_vectors_and_classes: tuple):
    latent_vectors = latent_vectors_and_classes[0]
    classes = latent_vectors_and_classes[1]
    latent_vectors_embedded = TSNE().fit_transform(latent_vectors)

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
