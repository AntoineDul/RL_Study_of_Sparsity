import numpy as np
import matplotlib.pyplot as plt
from utils import moving_average


def smooth_curve(y, window=50):
    if len(y) < window:
        return np.array(y)
    return moving_average(y, window)


def plot_comparison(results, metric_key, title, ylabel, window=50, save_path=None):
    plt.figure(figsize=(8, 5))

    for label, summary in results.items():
        y = summary[metric_key]
        y_smooth = smooth_curve(y, window)
        x = np.arange(len(y_smooth))
        plt.plot(x, y_smooth, label=label)

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()