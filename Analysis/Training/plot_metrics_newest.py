import matplotlib.pyplot as plt
import numpy as np

from Analysis.Training.load_from_logfiles import load_all_log_data, order_metric_data
from Analysis.Training.tools import find_nearest


def interpolate_metric_data(data, scaffold_points):
    scaffold_points = np.array(scaffold_points)
    scaffold_points = scaffold_points[scaffold_points[:, 1].argsort()]
    previous_switch_ep = 0

    for s in scaffold_points:
        switch_ep = s[0]
        data_before_switch = (data[:, 0] < switch_ep) * (data[:, 0] >= previous_switch_ep)

        switch_ep_difference = switch_ep - previous_switch_ep
        data[data_before_switch, 0] -= previous_switch_ep
        data[data_before_switch, 0] /= switch_ep_difference
        data[data_before_switch, 0] += (s[1] - 1)

        previous_switch_ep = switch_ep

    data_before_switch = (data[:, 0] >= previous_switch_ep)
    switch_ep_difference = np.sum(data_before_switch * 1) * 2

    data[data_before_switch, 0] -= previous_switch_ep
    data[data_before_switch, 0] /= switch_ep_difference
    data[data_before_switch, 0] += s[1]

    return data


def compute_rolling_averages_over_data_scaled_window(data, max_window, scaffold_points, min_window=10):
    # Compute episode length of each scaffold point...
    scaffold_points = np.array(scaffold_points)

    # Add final window for data since last scaffold point change
    scaffold_points = np.concatenate((scaffold_points, np.array([[max(data[:, 0]), max(scaffold_points[:, 1]+1)]])))

    scaffold_durations = scaffold_points[1:, 0] - scaffold_points[:-1, 0]
    scaffold_durations = np.concatenate((scaffold_points[0:1, 0], scaffold_durations))
    max_scaffold_duration = max(scaffold_durations)
    window_scaling = max_scaffold_duration / max_window
    window_sizes = scaffold_durations / window_scaling
    window_sizes = np.ceil(window_sizes).astype(int)
    window_sizes = np.clip(window_sizes, min_window, max_window)

    window_start = 0
    rolling_average_full = []
    for w, window in enumerate(window_sizes):
        window_end = int(scaffold_points[w, 0])
        window_end_index = find_nearest(data[:, 0], window_end)

        if w == len(window_sizes) - 1:
            data_points = data.shape[0]
            data_points_cut = data_points - window
        else:
            data_points_cut = len(data[:, 1])

        rolling_average = np.array([[np.mean(data[i: i + window, 1])] for i in range(window_start, window_end_index) if i < data_points_cut])
        rolling_average_full.append(rolling_average)
        window_start = window_end_index

    rolling_average = np.concatenate((rolling_average_full), axis=0)

    steps_cut = data[:data_points_cut, 0:1]

    rolling_average = np.concatenate((steps_cut, rolling_average), axis=1)
    return rolling_average


def compute_rolling_averages_over_data(data, window):
    data_points = data.shape[0]
    data_points_cut = data_points - window
    rolling_average = np.array([[np.mean(data[i: i + window, 1])] for i in range(data_points) if i < data_points_cut])
    steps_cut = data[:data_points_cut, 0:1]

    rolling_average = np.concatenate((steps_cut, rolling_average), axis=1)
    return rolling_average


def get_metric_name(metric_label):
    if metric_label == "prey caught":
        metric_name = "Prey Caught"

    elif metric_label == "capture success rate":
        # metric_name = "Capture Success Rate"
        metric_name = "CSR"

    elif metric_label == "prey capture rate (fraction caught per step)":
        # metric_name = "Prey Capture Rate (fraction caught per step)"
        metric_name = "PCR"

    elif metric_label == "prey capture index (fraction caught)":
        # metric_name = "Prey Capture Index (fraction caught)"
        metric_name = "PCI"

    elif metric_label == "Energy Efficiency Index":
        # metric_name = "Energy Efficiency Index"
        metric_name = "EEI"

    elif metric_label == "Episode Duration":
        metric_name = "Episode Duration"

    elif metric_label == "Mean salt damage taken per step":
        # metric_name = "Salt Damage (mean per step)"
        metric_name = "Salt Damage"

    elif metric_label == "Phototaxis Index":
        metric_name = "Phototaxis Index"

    elif metric_label == "episode reward":
        metric_name = "Episode Reward"

    elif metric_label == "predator avoidance index (avoided/p_pred)":
        # metric_name = "Predator Avoidance Index"
        metric_name = "PAI"

    elif metric_label == "predators avoided":
        metric_name = "Predators Avoided"

    elif metric_label == "Exploration Quotient":
        metric_name = "EQ"

    elif metric_label == "turn chain preference":
        # metric_name = "Turn Chain Preference"
        metric_name = "TCP"

    elif metric_label == "Action Heterogeneity Score":
        metric_name = "AHI"

    else:
        metric_name = metric_label

    return metric_name


def remove_repeated_switching_points(scaffold_switching_points):
    """Removes switching points - chooses to keep those of a later episode."""

    cleaned_scaffold_switching_points = []
    for model_switching_points in scaffold_switching_points:
        new_model_switching_points = []

        config_nums = [m[1] for m in model_switching_points]
        reduced_config_nums = list(set(config_nums))

        for c in reduced_config_nums:
            switching_points = [s[0] for s in model_switching_points if s[1] == c]
            new_model_switching_points.append([max(switching_points), c])

        # to_delete = []
        # for i, c in enumerate(config_nums):
        #     if i > 0:
        #         if c == model_switching_points[i - 1][1]:
        #             to_delete.append(i - 1)
        #             print("Removing repeated")
        # for d in reversed(to_delete):
        #     del model_switching_points[d]
        cleaned_scaffold_switching_points.append(new_model_switching_points)
    return cleaned_scaffold_switching_points


def plot_multiple_metrics_multiple_models(model_list, metrics, window, interpolate_scaffold_points, figure_name,
                                          key_scaffold_points=None, scaled_window=False, show_inset=None):
    """Different to previous versions in that it uses data directly from log files, and scales points between scaffold
    switching points to allow plotting between models. The resulting graph is x: config change point, y: metric.

    window is the window for computing the rolling averages for each metric.
    """

    model_data = [load_all_log_data(model) for model in model_list]
    # Parsimonious way:
    ordered_chosen_model_data = [{metric: order_metric_data(model[metric]) for metric in metrics} for model in
                                 model_data]
    # With error handling:
    # ordered_chosen_model_data = []
    # for model in model_data:
    #     metric_data = {}
    #     for metric in metrics:
    #         try:
    #             metric_data[metric] = order_metric_data(model[metric])
    #         except KeyError:
    #             print(f"{model} - {metric} data unavailable")
    #     ordered_chosen_model_data.append(metric_data)

    #                Rolling Averages
    # Parsimonious way:
    if scaled_window:
        ordered_chosen_model_data_rolling_averages = [
            {metric: compute_rolling_averages_over_data_scaled_window(model[metric], window,
                                                        model_data[m]["Configuration change"]) for metric in metrics}
            for m, model in enumerate(ordered_chosen_model_data)]
    else:
        ordered_chosen_model_data_rolling_averages = [
            {metric: compute_rolling_averages_over_data(model[metric], window) for metric
             in metrics} for model in ordered_chosen_model_data]
    # With error handling:
    # ordered_chosen_model_data_rolling_averages = []
    # for model in ordered_chosen_model_data:
    #     metric_data = {}
    #     for metric in metrics:
    #         try:
    #             metric_data[metric] = compute_rolling_averages_over_data(model[metric], window)
    #         except KeyError:
    #             print(f"{metric} data unavailable")
    #         ordered_chosen_model_data_rolling_averages.append(metric_data)

    # episodes = np.array(ordered_chosen_model_data_rolling_averages[0]["prey caught"])[:, 0]
    # plt.plot(sorted(episodes))
    # plt.show()

    if interpolate_scaffold_points:
        scaffold_switching_points = [model["Configuration change"] for model in model_data]
        scaffold_switching_points = remove_repeated_switching_points(scaffold_switching_points)

        new_orders = [np.argsort(np.array(model)[:, 1]) for model in scaffold_switching_points]
        scaffold_switching_points = [np.array(model)[new_orders[i]] for i, model in
                                     enumerate(scaffold_switching_points)]
        # if scaled_window:
        #     ordered_chosen_model_data_rolling_averages = [{metric: interpolate_metric_data_scaled_window(model[metric],
        #                                                                                                  scaffold_switching_points[
        #                                                                                                      i], window)
        #                                                    for metric in metrics} for i, model in
        #                                                   enumerate(ordered_chosen_model_data_rolling_averages)]
        # else:
        ordered_chosen_model_data_rolling_averages = [{metric: interpolate_metric_data(model[metric],
                                                                                           scaffold_switching_points[i],
                                                                                           )
                                                           for metric in metrics} for i, model in
                                                          enumerate(ordered_chosen_model_data_rolling_averages)]

    num_metrics = len(metrics)

    inset_scaffold_points = []
    inset_metric_vals = []
    metric_index = None

    fig, axs = plt.subplots(num_metrics, 1, figsize=(50, int(12 * num_metrics)), sharex=True)
    for model in ordered_chosen_model_data_rolling_averages:
        for i, metric in enumerate(metrics):
            metric_name = get_metric_name(metric)

            if metric_name == "AHI":
                to_zero = (model[metric][:, 0] < 2)
                model[metric][to_zero, 1] = 0

            # if metric_name == "Phototaxis Index":
            #     to_switch = (model[metric][:, 0] < 31)
            #     model[metric][to_switch, 1] -= 0.5
            #     model[metric][to_switch, 1] *= 2
            axs[i].plot(model[metric][:, 0], model[metric][:, 1], alpha=0.5)
            if min(model[metric][:, 1]) < 0:
                axs[i].hlines(0, 0, max(model[metric][:, 0]), color="black", linestyles="dashed")

            axs[i].grid(True, axis="x")

            if key_scaffold_points is not None:
                ylim = axs[i].get_ylim()
                for p in key_scaffold_points:
                    axs[i].vlines(p, ylim[0], ylim[1], color="r")
                axs[i].set_ylim(ylim[0], ylim[1])
            axs[i].set_ylabel(metric_name)

            if show_inset is not None:
                if show_inset[0] == metric:
                    scaffold_point = show_inset[1]
                    data_to_keep = (model[metric][:, 0] >= scaffold_point) * (model[metric][:, 0] < scaffold_point + 1)
                    inset_scaffold_points.append(model[metric][data_to_keep, 0])
                    inset_metric_vals.append(model[metric][data_to_keep, 1])
                    metric_index = i

            # plt.setp(axs[i].get_yticklabels(), rotation=30, horizontalalignment='right')

    if interpolate_scaffold_points:
        axs[-1].set_xlabel("Scaffold Point")
        sc = np.concatenate(([np.array(s) for s in scaffold_switching_points]))
        scaffold_indices = [t for t in range(0, int(np.max(sc[:, 1])))]
        axs[-1].set_xticks(scaffold_indices)
        axs[-1].set_xlim(1, np.max(sc[:, 1]) + 1)
    else:
        axs[-1].set_xlabel("Episode")

    if show_inset is not None:
        inset_ylim = axs[metric_index].get_ylim()

    plt.savefig(f"../../Analysis-Output/Training/{figure_name}.jpg")
    plt.clf()
    plt.close()

    if show_inset is not None:
        create_zoomed_inset(inset_scaffold_points, inset_metric_vals, get_metric_name(show_inset[0]), inset_ylim)


def create_zoomed_inset(scaffold_points, metric_vals, metric_name, inset_ylim):
    for model in range(len(scaffold_points)):
        plt.plot(scaffold_points[model], metric_vals[model])

    ax = plt.gca()
    ax.set_ylim(inset_ylim)
    plt.ylabel(metric_name)
    plt.xlabel("Scaffold Point")
    plt.show()


"""
Possible metrics:
   - "prey capture index (fraction caught)"
   - "prey capture rate (fraction caught per step)"
   - "Energy Efficiency Index
   - "Episode Duration"
   - "Mean salt damage taken per step"
   - "Phototaxis Index"
   - "capture success rate"
   - "episode reward"
   - "predator avoidance index (avoided/p_pred)"
   - "predators avoided"
   - "prey caught"
   - "Exploration Quotient"
   - "turn chain preference"                      
   - "Cause of Death"
   - Action Heterogeneity Score
"""

if __name__ == "__main__":
    dqn_models_old = ["dqn_scaffold_30-1", "dqn_scaffold_30-2"]
    ppo_models_old = ["ppo_scaffold_21-1", "ppo_scaffold_21-2"]

    dqn_models = ["dqn_beta-1", "dqn_beta-2", "dqn_beta-3", "dqn_beta-4", "dqn_beta-5"]
    dqn_models_mod = ["dqn_beta_mod-1", "dqn_beta_mod-2", "dqn_beta_mod-3", "dqn_beta_mod-4", "dqn_beta_mod-5"]
    ppo_models = ["ppo_beta-1", "ppo_beta-2", "ppo_beta-3", "ppo_beta-4", "ppo_beta-5"]
    ppo_models_mod = ["ppo_beta_mod-1", "ppo_beta_mod-2", "ppo_beta_mod-3", "ppo_beta_mod-4", "ppo_beta_mod-5"]

    limited_metrics_dqn = ["prey capture index (fraction caught)",
                          "capture success rate",
                          "episode reward",
                          "Phototaxis Index"
                          ]
    chosen_metrics_dqn = ["prey capture index (fraction caught)",
                          "capture success rate",
                          "episode reward",
                          "Energy Efficiency Index",
                          "Episode Duration",
                          "Exploration Quotient",
                          "Action Heterogeneity Score",

                          "turn chain preference",
                          # "Cause of Death",
                          # Sand grain attempted captures.
                          # DQN only
                          # "predator avoidance index (avoided/p_pred)",
                          "Phototaxis Index"
                          ]
    chosen_metrics_dqn_mod = ["prey capture index (fraction caught)",
                              "capture success rate",
                              "episode reward",
                              "Energy Efficiency Index",
                              "Episode Duration",
                              "Exploration Quotient",
                              "Action Heterogeneity Score",

                              "turn chain preference",
                              # "Cause of Death",
                              # Sand grain attempted captures.

                              "predator avoidance index (avoided/p_pred)",
                              "Phototaxis Index"
                              ]
    chosen_metrics_ppo = ["prey capture index (fraction caught)",
                          "capture success rate",
                          # "episode reward",
                          # "Energy Efficiency Index",
                          "Episode Duration",
                          # "Exploration Quotient",
                          # "turn chain preference",
                          # "Cause of Death",
                          # Sand grain attempted captures.
                          # DQN only
                          # "predator avoidance index (avoided/p_pred)",
                          # "Phototaxis Index"
                          ]
    chosen_metrics_ppo_mod = ["prey capture index (fraction caught)",
                          "capture success rate",
                          # "episode reward",
                          # "Energy Efficiency Index",
                          "Episode Duration",
                          # "Exploration Quotient",
                          # "turn chain preference",
                          # "Cause of Death",
                          # Sand grain attempted captures.
                          # DQN only
                          "predator avoidance index (avoided/p_pred)",
                          # "Phototaxis Index"
                          ]
    # plot_multiple_metrics_multiple_models(dqn_models, chosen_metrics_dqn, window=40, interpolate_scaffold_points=True,
    #                                       figure_name="dqn_beta", scaled_window=False,
    #                                       show_inset=["capture success rate", 10])  # , key_scaffold_points=[10, 16, 31])
    # plot_multiple_metrics_multiple_models(dqn_models_mod, chosen_metrics_dqn_mod, window=40, interpolate_scaffold_points=True,
    #                                       figure_name="dqn_beta_mod")#, key_scaffold_points=[10, 16, 31])
    plot_multiple_metrics_multiple_models(ppo_models, chosen_metrics_ppo, window=40, interpolate_scaffold_points=False,
                                          figure_name="ppo_beta")
    plot_multiple_metrics_multiple_models(ppo_models_mod, chosen_metrics_ppo_mod, window=40, interpolate_scaffold_points=False,
                                          figure_name="ppo_beta_mod")