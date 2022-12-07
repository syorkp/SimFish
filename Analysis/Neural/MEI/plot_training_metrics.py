import numpy as np

import matplotlib.pyplot as plt


def plot_training_metrics_full(trial_name, model_name):
    with open(f"MEI-Models/{trial_name}/{model_name}/prediction_error.npy", "rb") as f:
        data = np.load(f)
    # plt.plot(data)
    # plt.savefig(f"MEI-Models/{trial_name}/{model_name}/prediction_error.png")

    with open(f"MEI-Models/{trial_name}/{model_name}/compiled_loss.npy", "rb") as f:
        data = np.load(f)
    data = data[1000:]
    running_mean = np.array([np.mean(data[i:i+100]) for i in range(len(data)-100)])
    plt.plot(running_mean)
    plt.savefig(f"MEI-Models/{trial_name}/{model_name}/compiled_loss.png")


if __name__ == "__main__":
    trial_name = "dqn_scaffold_18-1"
    model_name = "conv4l_1"
    plot_training_metrics_full(trial_name, model_name)


