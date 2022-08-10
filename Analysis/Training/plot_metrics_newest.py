import matplotlib.pyplot as plt
import numpy as np


from Analysis.Training.load_from_logfiles import load_all_log_data, order_metric_data


def interpolate_metric_data(data, scaffold_points):
    scaffold_points = np.array(scaffold_points)
    scaffold_points = scaffold_points[scaffold_points[:, 1].argsort()]
    previous_switch_ep = 0
    previous_switch_len = 0

    for s in scaffold_points:
        switch_ep = s[0]
        data_before_switch = (data[:, 0] < switch_ep) * (data[:, 0] >= previous_switch_ep)
        switch_len = np.sum(data_before_switch * 1)

        data[data_before_switch, 0] -= previous_switch_len
        data[data_before_switch, 0] /= switch_len
        data[data_before_switch, 0] += (s[1] - 1)

        previous_switch_ep = switch_ep
        previous_switch_len += switch_len

    data_before_switch = (data[:, 0] >= previous_switch_ep)
    switch_len = np.sum(data_before_switch * 1)

    data[data_before_switch, 0] -= previous_switch_len
    data[data_before_switch, 0] /= switch_len
    data[data_before_switch, 0] += s[1]

    return data


def compute_rolling_averages_over_data(data, window):
    data_points = data.shape[0]
    data_points_cut = data_points - window
    rolling_average = np.array([[np.mean(data[i: i+window, 1])] for i in range(data_points) if i < data_points_cut])
    steps_cut = data[:data_points_cut, 0:1]

    rolling_average = np.concatenate((steps_cut, rolling_average), axis=1)
    return rolling_average


def plot_multiple_metrics_multiple_models(model_list, metrics, window, interpolate_scaffold_points):
    """Different to previous versions in that it uses data directly from log files, and scales points between scaffold
    switching points to allow plotting between models. The resulting graph is x: config change point, y: metric.

    window is the window for computing the rolling averages for each metric.
    """

    model_data = [load_all_log_data(model) for model in model_list]
    ordered_chosen_model_data = [{metric: order_metric_data(model[metric]) for metric in metrics} for model in
                                 model_data]
    ordered_chosen_model_data_rolling_averages = [{metric: compute_rolling_averages_over_data(model[metric], window) for metric
                                                   in metrics} for model in ordered_chosen_model_data]
    if interpolate_scaffold_points:
        scaffold_switching_points = [model["Configuration change"] for model in model_data]
        ordered_chosen_model_data_rolling_averages = [{metric: interpolate_metric_data(model[metric],
                                                                                       scaffold_switching_points[i])
                                                       for metric in metrics} for i, model in
                                                      enumerate(ordered_chosen_model_data_rolling_averages)]

    num_metrics = len(metrics)
    fig, axs = plt.subplots(num_metrics, 1, figsize=(15, int(4*num_metrics)), sharex=True)
    for model in ordered_chosen_model_data_rolling_averages:
        for i, metric in enumerate(metrics):
            axs[i].plot(model[metric][:, 0], model[metric][:, 1])
            axs[i].set_ylabel(metric)
    axs[-1].set_xlabel("Scaffold Point")
    sc = np.concatenate(([np.array(s) for s in scaffold_switching_points]))
    axs[-1].set_xticks([int(t) for t in np.linspace(0, np.max(sc[:, 1]))])
    plt.grid(axis="x")
    axs[-1].set_xlim(1, np.max(sc[:, 1])+1)
    plt.show()


def plot_scaffold_durations(model_name):
    data = load_all_log_data(model_name)
    scaffold_switching_points = np.array(data["Configuration change"])
    scaffold_switching_points = scaffold_switching_points[scaffold_switching_points[:, 1].argsort()]

    config = scaffold_switching_points[:, 1]
    episode = scaffold_switching_points[:, 0]
    duration = [d - episode[i-1] if i > 0 else d for i, d in enumerate(episode)]
    plt.plot(config, duration)
    plt.show()


if __name__ == "__main__":
    models = ["ppo_scaffold_21-1", "ppo_scaffold_21-2"]
    # models = ["ppo_scaffold_22-1", "ppo_scaffold_22-2"]
    # models = ["dqn_scaffold_26-1", "dqn_scaffold_26-2", "dqn_scaffold_26-3", "dqn_scaffold_26-4"]
    # models = ["dqn_scaffold_27-1", "dqn_scaffold_27-2"]
    # models = ["dqn_scaffold_28-1", "dqn_scaffold_28-2"]

    chosen_metrics = ["prey capture index (fraction caught)", "prey capture rate (fraction caught per step)"]
    plot_multiple_metrics_multiple_models(models, chosen_metrics, window=40, interpolate_scaffold_points=True)
    plot_scaffold_durations(models[0])

