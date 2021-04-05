import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

def plot_metric(metric_dict, label, averaged_plot=True, n=4):
    global_steps = list(metric_dict.keys())
    metric = list(metric_dict.values())

    if not averaged_plot:
        plt.plot(global_steps, metric, label=label)
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
                         alpha=0.1, label= f'{label} variance over {n} points')

def plot_t_sne(model, visualisation_data):
    """
    Creates a t-SNE plot of the output of a model, by feeding the data to be visualised forward through the model
    :param model: nn.Module, Will use the encoder of a semi supervised classifier. May be trained or not.
    :param visualisation_data:
    :return:
    """
    latent_vectors = []
    classes = []
    for X_batch, Y_batch in visualisation_data:
        latent_vectors_batch = model(X_batch)
        latent_vectors.extend(latent_vectors_batch.detach().numpy())
        classes.extend(Y_batch.detach().numpy())
    latent_vectors_embedded = TSNE().fit_transform(latent_vectors)

    #create the 'data table' to show in the scatterplot
    d = {'x': latent_vectors_embedded[:,0], 'y':latent_vectors_embedded[:,1], 'classes': classes}
    df = pd.DataFrame(data=d)
    sns.scatterplot(data = df, x = 'x', y = 'y', hue='classes', palette='deep')
    plt.show()


def main():
    values = np.random.uniform(0,2, 6000)
    test_dict = {}
    for i in range(len(values)):
        test_dict[i] = values[i]

    plot_metric(test_dict, 'Test_plot')

if __name__ == '__main__':
    main()