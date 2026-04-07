import matplotlib.pyplot as plt
import datetime 
import numpy as np

from utils import moving_average

def plot_step_tracker(step_tracker):
    plt.plot(step_tracker)
    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.title('Sarsa Learning Progress')
    plt.grid()
    plt.savefig(f"plot_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

# Test plot (ChatGPT generated)
def plot_single_run(success_tracker, step_tracker=None, window=50, save_path="plot_single_run.png"):
    """
    Plots smoothed success rate (and optionally steps) for one training run.

    Args:
        success_tracker (list): list of 0/1 per episode
        step_tracker (list, optional): number of steps per episode
        window (int): moving average window
        save_path (str, optional): if provided, saves the plot

    """

    episodes = np.arange(len(success_tracker))

    # --- Smooth success rate ---
    smoothed_success = moving_average(success_tracker, window)
    success_x = np.arange(len(smoothed_success))

    plt.figure()

    plt.plot(success_x, smoothed_success, label="Success Rate (smoothed)")

    # --- Optional: steps ---
    if step_tracker is not None:
        smoothed_steps = moving_average(step_tracker, window)
        steps_x = np.arange(len(smoothed_steps))

        plt.plot(steps_x, smoothed_steps, label="Steps (smoothed)", linestyle="--")

    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.title(f"Single Run Performance (window={window})")
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()