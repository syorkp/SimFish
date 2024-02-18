import matplotlib.pyplot as plt
import numpy as np

from Analysis.Training.load_from_logfiles import load_all_log_data, order_metric_data
from Analysis.Training.tools import find_nearest
from Analysis.Training.plot_metrics import order_chosen_model_data, compute_rolling_averages


def plot_reward_pre_post_scaffold(model_list, model_list_no_scaffold, window, figure_name):
    metrics = ["episode reward", "Episode Duration"]

    # For normal models
    model_data = [load_all_log_data(model) for model in model_list]

    ordered_chosen_model_data = order_chosen_model_data(metrics, model_data)
    scaffold_switching_points = [np.array(model["Configuration change"]) for model in model_data]
    scaffold_switching_points = [scaffold_points[scaffold_points[:, 1].argsort()] for scaffold_points in scaffold_switching_points]

    scaffold_start_episode = [model[0, 0] for model in scaffold_switching_points]
    all_episode_steps = [np.concatenate((model["Episode Duration"][:, 0:1], np.expand_dims(np.cumsum(model["Episode Duration"][:, 1]), 1)), axis=1)
                         for model in ordered_chosen_model_data]
    scaffold_start_step_index = [np.where(episode_steps[:, 0] == scaffold_start_episode[i]) for i, episode_steps
                                 in enumerate(all_episode_steps)]
    scaffold_start_step = [all_episode_steps[i][scaffold_start_index[0]] for i, scaffold_start_index in enumerate(scaffold_start_step_index)]


    # For models with no scaffold
    model_data_no_scaffold = [load_all_log_data(model) for model in model_list_no_scaffold]

    ordered_chosen_model_data_no_scaffold = order_chosen_model_data(metrics, model_data_no_scaffold)

    all_episode_steps_no_scaffold = [np.concatenate((model["Episode Duration"][:, 0:1], np.expand_dims(np.cumsum(model["Episode Duration"][:, 1]), 1)), axis=1)
                         for model in ordered_chosen_model_data_no_scaffold]

    # Compute rolling averages
    ordered_chosen_model_data_rolling_averages = compute_rolling_averages(ordered_chosen_model_data, model_data,
                                                                          metrics, window,
                                                                          scaled_window=False)
    ordered_chosen_model_data_rolling_averages_no_scaffold = compute_rolling_averages(ordered_chosen_model_data_no_scaffold,
                                                                                      model_data_no_scaffold,
                                                                          metrics, window,
                                                                          scaled_window=False)

    all_reward_data = [data["episode reward"][:, 1] for data in ordered_chosen_model_data_rolling_averages] + \
                      [data["episode reward"][:, 1] for data in ordered_chosen_model_data_rolling_averages_no_scaffold]
    max_reward = np.max(np.concatenate(all_reward_data))
    min_reward = np.min(np.concatenate(all_reward_data))

    max_steps = min(np.max(model_steps[:, 1]) for model_steps in all_episode_steps)

    fig, axs = plt.subplots(figsize=(10, 6))
    # Normal models
    for m, model in enumerate(model_list):
        # Interpolate steps so is before and after introduction of scaffold. Scale to halfway point between both
        scaffold_switch_steps = scaffold_start_step[m][0, 1]
        reward = ordered_chosen_model_data_rolling_averages[m]["episode reward"][:, 1]
        relevant_steps = all_episode_steps[m][:len(reward), 1]
        data_before_switch = relevant_steps < scaffold_switch_steps

        relevant_steps /= (2 * np.max(relevant_steps[data_before_switch]))
        # relevant_steps[data_before_switch] /= (2 * np.max(relevant_steps[data_before_switch]))
        # relevant_steps[~data_before_switch] /= (np.max(relevant_steps[data_before_switch]))
        relevant_steps *= max_steps

        if m == 0:
            axs.plot(relevant_steps, reward, color="b", label="With Scaffold")
        else:
            axs.plot(relevant_steps, reward, color="b")


    # No scaffold models
    for m, model in enumerate(model_list_no_scaffold):
        # Interpolate steps so is before and after introduction of scaffold. Scale to halfway point between both
        reward = ordered_chosen_model_data_rolling_averages_no_scaffold[m]["episode reward"][:, 1]
        relevant_steps = all_episode_steps_no_scaffold[m][:len(reward), 1]

        # relevant_steps[data_before_switch] /= (2 * np.max(relevant_steps[data_before_switch]))
        # relevant_steps[~data_before_switch] /= (np.max(relevant_steps[data_before_switch]))

        if m == 0:
            axs.plot(relevant_steps, reward, color="y", label="No Scaffold")
        else:
            axs.plot(relevant_steps, reward, color="y")


    axs.vlines(max_steps/2, min_reward, max_reward, color="r")

    axs.set_xlim(0, max_steps)
    axs.set_xlabel("Training Steps", size=20)
    axs.set_ylabel("Episode Reward", size=20)
    axs.legend()

    plt.savefig(f"../../Analysis-Output/Training/Metric-Plots/{figure_name}_normalised_reward.jpg")

    plt.clf()
    plt.close()


if __name__ == "__main__":
    dqn_models = ["dqn_delta-1", "dqn_delta-2", "dqn_delta-3", "dqn_delta-4", "dqn_delta-5"]
    dqn_models_no_scaffold = ["dqn_delta_ns-1", "dqn_delta_ns-2"]

    plot_reward_pre_post_scaffold(dqn_models, dqn_models_no_scaffold, 10, figure_name="dqn_delta")
