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
    all_episode_steps = [np.concatenate((model["Episode Duration"][:, 0:1], np.expand_dims(np.cumsum(model["Episode Duration"][:, 1]), 1)), axis=1)
                         for model in ordered_chosen_model_data]
    scaffold_start_step_index = [np.where(episode_steps[:, 0] == scaffold_start_episode[i]) for i, episode_steps
                                 in enumerate(all_episode_steps)]
    scaffold_start_step = [all_episode_steps[i][scaffold_start_index[0]] for i, scaffold_start_index in enumerate(scaffold_start_step_index)]

    # Compute rolling averages
    ordered_chosen_model_data_rolling_averages = compute_rolling_averages(ordered_chosen_model_data, model_data,
                                                                          metrics, window,
                                                                          scaled_window=False)

    all_reward_data = [data["episode reward"][:, 1] for data in ordered_chosen_model_data_rolling_averages]
    max_reward = np.max(np.concatenate(all_reward_data))
    min_reward = np.min(np.concatenate(all_reward_data))

    max_steps = min(np.max(model_steps[:, 1]) for model_steps in all_episode_steps)

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

        plt.plot(relevant_steps, reward)


    plt.vlines(max_steps/2, min_reward, max_reward, color="r")
    plt.xlim(0, max_steps)
    plt.xlabel("Training Steps")
    plt.ylabel("Episode Reward")
    plt.show()
    # TODO: Add adjustment for window size? (also to metric plot)


if __name__ == "__main__":
    dqn_models = ["dqn_gamma-1", "dqn_gamma-2", "dqn_gamma-3", "dqn_gamma-4", "dqn_gamma-5"]

    plot_reward_pre_post_scaffold(dqn_models, 30)
    # TODO: Add option for second set of models, which dont have scaffold switches