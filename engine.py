import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.cm import register_cmap
from utilities import vectorize


plt.rcParams['ytick.labelcolor'] = '#96a7b0'


def gradient(colors=[], ranges=[], name=""):
    cmap = LinearSegmentedColormap.from_list(name, list(zip(ranges, colors)))
    register_cmap(name=name, cmap=cmap)
    return cmap


def colors(colors=[], name=""):
    cmap = ListedColormap(colors, name=name)
    register_cmap(name=name, cmap=cmap)
    return cmap


gradient(
    name="islands",
    colors=["#2B3A67", "#0E79B2", "#8F754F", "#41521F", "#256D1B"],
    ranges=[0.0, 0.45, 0.5, 0.6, 1.0],
)


gradient(
    name="mountains",
    colors=[
        "#2B3A67", "#0E79B2",
        "#8F754F", "#41521F",
        "#256D1B", "#E1C16E",
        "#CD7F32", "#EADDCA",
    ],
    ranges=[0.0, 0.35, 0.40, 0.55, 0.65, 0.75, 0.85, 1.0],
)


gradient(
    name="snowpeaks",
    colors=["#2B3A67", "#0E79B2", "#302B27", "#E0E2DB", "#545454", "#302B27"],
    ranges=[0.0, 0.45, 0.5, 0.6, 0.85, 1.0],
)


gradient(
    name="frostfire",
    colors=["#96031A", "#A63C06", "#302B27", "#E0E2DB", "#1B1B1E"],
    ranges=[0.0, 0.45, 0.5, 0.55, 1.0],
)


def Heatmap(*matrix, scale=1.0, cbar=False, cmap='islands', **kwargs):
    """Create one or multiple heatmaps and arrange them into a row."""

    # Make grid
    shape = np.shape(matrix)[0]
    fig, axs = plt.subplots(ncols=shape)
    fig.dpi = 100 * scale
    fig.set_facecolor('white')
    fig.frameon = False
    if shape == 1:
        axs = [axs]

    # Render heatmaps
    for i, ax in enumerate(axs):
        ax.axis("scaled")
        sns.heatmap(
            matrix[i],
            ax=ax,
            cmap=cmap,
            yticklabels=False,
            xticklabels=False,
            cbar=cbar,
            cbar_kws={"pad": 0.04, "shrink": 1 / len(axs)},
            **kwargs
        )


def Field(*matrix, time=20, alpha=1.0, seed=0, scale=1.0, cbar=False, cmap='twilight', **kwargs):
    """Visualise matrix as a vector field.

    Trailing effects are achieved by placing random particles inside the field
    and the simulating their movements.

        time - running time of the simulation

        alpha - weak trail relevance, the lower the value the more information is preserved

        seed - allows for control of the random process
    """
    matrices = [vectorize(m, time=time, alpha=alpha, seed=seed)
                for m in matrix]
    Heatmap(*matrices, scale=scale, cbar=cbar, cmap=cmap, **kwargs)


def Noise(noise_function, **kwargs):
    """Create interactive heatmap.
    Every matched keyword argument is passed to the noise function.
    Arguments that failed to match are passed to the heatmap instead.
    """

    tokens = noise_function.__code__.co_varnames
    fargs = {k: v for k, v in kwargs.items() if k in tokens}
    hargs = dict(set(kwargs.items()) - set(fargs.items()))
    interact(lambda **fargs: Heatmap(noise_function(**fargs), **hargs), **fargs)


def WeightSum(weights, noises):
    """Weight sum of noises."""
    assert len(weights) == len(noises)
    return sum(weight * noise for weight, noise in zip(weights, noises))


def CombineNoises(weights, noises):
    return WeightSum(weights, noises)/sum(weights)
