import matplotlib.pyplot as plt
import numpy as np

from Analysis.Training.load_from_logfiles import load_all_log_data, order_metric_data
from Analysis.Training.tools import find_nearest
from Analysis.Training.plot_metrics_newest import order_chosen_model_data, compute_rolling_averages


def plot_reward_pre_post_scaffold(model_list, window):
    model_data = [load_all_log_data(model) for model in model_list]
    metrics = ["episode reward", "Episode Duration"]

    ordered_chosen_model_data = order_chosen_model_data(metrics, model_data)
    scaffold_switching_points = [np.array(model["Configuration change"]) for model in model_data]
    scaffold_switching_points = [scaffold_points[scaffold_points[:, 1].argsort()] for scaffold_points in scaffold_switching_points]

    scaffold_start_episode = [model[0, 0] for model in scaffold_switching_points]
    all_episode_steps = [np.cumsum(model, axis=1) for model in ordered_chosen_model_data]
    scaffold_start_step_index = [np.where(episode_steps == scaffold_start_episode[i]) for i, episode_steps
                                 in enumerate(all_episode_steps)]
    scaffold_start_step = [all_episode_steps[scaffold_start_index] for scaffold_start_index in scaffold_start_step_index]

    # Compute rolling averages
    ordered_chosen_model_data_rolling_averages = compute_rolling_averages(ordered_chosen_model_data, model_data,
                                                                          "episode reward", window,
                                                                          scaled_window=False)

    all_reward_data = [data["episode reward"][:, 1] for data in ordered_chosen_model_data_rolling_averages]
    max_reward = np.max(np.concatenate(all_reward_data))
    min_reward = np.min(np.concatenate(all_reward_data))

    plt.plot(ordered_chosen_model_data_rolling_averages[0]["episode reward"][:, 0], ordered_chosen_model_data_rolling_averages[0]["episode reward"][:, 1])
    plt.vlines(scaffold_start_step[0], min_reward, max_reward, color="r")
    plt.show()
    # Plot reward across all steps
    # Add vline at scaffold start.
    ...


if __name__ == "__main__":
    plot_reward_pre_post_scaffold(["dqn_gamma-1", "dqn_gamma-2"], 20)
