import numpy as np
import matplotlib.pyplot as plt

from Analysis.Training.load_from_logfiles import load_all_log_data, order_metric_data
from Analysis.Training.plot_metrics_newest import compute_rolling_averages_over_data, remove_repeated_switching_points, \
    interpolate_metric_data
from Analysis.load_model_config import load_configuration_files_by_scaffold_point, get_scaffold_num_points
from Analysis.Training.tools import find_nearest


def get_available_reward(environment_params, learning_params):
    prey_reward_available = environment_params["capture_basic_reward"] * environment_params["prey_num"]
    return prey_reward_available


def scale_rewards(reward_data, config_name):
    num_configs = get_scaffold_num_points(config_name)
    for config in range(1, num_configs+1):
        env, params = load_configuration_files_by_scaffold_point(config_name, config)
        available_reward = get_available_reward(env, params)

        for m, model in enumerate(reward_data):
            to_scale = (model[:, 0] >= config) * (model[:, 0] < config + 1)
            reward_data[m][to_scale, 1] /= available_reward

    return reward_data


def plot_normalised_reward(model_list, config_name, window, interpolate_scaffold_points, figure_name):
    model_data = [load_all_log_data(model) for model in model_list]
    ordered_chosen_model_data = [order_metric_data(model["episode reward"]) for model in model_data]

    ordered_chosen_model_data_rolling_averages = [compute_rolling_averages_over_data(model, window) for model in
                                                  ordered_chosen_model_data]
    scaffold_point_rewards = [get_rewards_at_each_scaffold_point(model["episode reward"], model["Configuration change"], window) for model in model_data]


    if interpolate_scaffold_points:
        scaffold_switching_points = [model["Configuration change"] for model in model_data]
        scaffold_switching_points = remove_repeated_switching_points(scaffold_switching_points)

        new_orders = [np.argsort(np.array(model)[:, 1]) for model in scaffold_switching_points]
        scaffold_switching_points = [np.array(model)[new_orders[i]] for i, model in
                                     enumerate(scaffold_switching_points)]
        ordered_chosen_model_data_rolling_averages = [interpolate_metric_data(model, scaffold_switching_points[i])
                                                      for i, model in
                                                      enumerate(ordered_chosen_model_data_rolling_averages)]

    # ordered_chosen_model_data_rolling_averages = scale_rewards(ordered_chosen_model_data_rolling_averages, config_name)

    for model in ordered_chosen_model_data_rolling_averages:
        plt.plot(model[:, 0], model[:, 1])

    plt.show()


def get_rewards_at_each_scaffold_point(reward_data, scaffold_points, window):
    reward_data = np.array(reward_data)
    end_rewards = []

    for p, s in scaffold_points:
        end_index = find_nearest(reward_data[:, 0], p)
        mean_reward = np.mean(reward_data[end_index-window:end_index, 1])
        end_rewards.append(mean_reward)

    cumulative_end_reward = [np.sum(end_rewards[:i+1]) for i in range(len(end_rewards))]
    # TODO: Compute added difficulty by previous r - new r. Could account for ongoing training for old environment by initialising to the max.
    max_index = np.argmax(np.array(end_rewards))

    plt.plot(end_rewards)
    plt.show()
    x = True
    return scaled_reward


if __name__ == "__main__":

    dqn_models = ["dqn_beta-1", "dqn_beta-2", "dqn_beta-3", "dqn_beta-4", "dqn_beta-5"]

    plot_normalised_reward(dqn_models, "dqn_beta", 10, True, "dqn_beta")

