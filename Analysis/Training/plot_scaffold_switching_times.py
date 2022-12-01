import numpy as np
import matplotlib.pyplot as plt

from Analysis.Training.load_from_logfiles import load_all_log_data


def get_all_switching_times(model_name, steps):
    data = load_all_log_data(model_name)
    scaffold_switching_points = np.array(data["Configuration change"])
    scaffold_switching_points = scaffold_switching_points[scaffold_switching_points[:, 1].argsort()]

    scaffold_durations_episodes = scaffold_switching_points[1:, 0] - scaffold_switching_points[:-1, 0]
    scaffold_durations_episodes = np.concatenate((scaffold_switching_points[0:1, 0], scaffold_durations_episodes))

    config = scaffold_switching_points[:, 1]

    if not steps:
        return config, list(scaffold_durations_episodes)

    # Make it steps instead...
    scaffold_durations_steps = []
    episode_durations = np.array(data["Episode Duration"])
    start_eps = 0
    for s_eps in scaffold_switching_points[:, 0]:
        episodes_in_interval = (episode_durations[:, 0] >= start_eps) * (episode_durations[:, 0] < s_eps )
        total_steps = np.sum(episode_durations[episodes_in_interval, 1])
        scaffold_durations_steps.append(total_steps)
        start_eps = s_eps

    return config, scaffold_durations_steps


def plot_scaffold_durations(model_name):
    config, duration = get_all_switching_times(model_name)
    plt.plot(config, duration)
    plt.show()


def plot_scaffold_durations_multiple_models(model_list, steps=True):
    fig, ax = plt.subplots()
    config_compiled = []
    duration_compiled = []

    for model in model_list:
        config, duration = get_all_switching_times(model, steps)
        config_compiled.append(config)
        duration_compiled.append(duration)

    max_config = max([len(c) for c in config_compiled])
    switching_points = [i for i in range(max_config)]

    for duration in duration_compiled:
        while len(duration) < max_config:
            duration.append(0)

    for d, duration in enumerate(duration_compiled):
        if d > 0:
            ax.bar(switching_points, duration, bottom=bottom_compiled)
            bottom_compiled += np.array(duration)
        else:
            bottom_compiled = np.array(duration)
            ax.bar(switching_points, duration)

    plt.xlabel("Scaffold Point")
    plt.ylabel("Total Steps")
    plt.legend(model_list)
    plt.show()


if __name__ == "__main__":
    dqn_models_old = ["dqn_scaffold_30-1", "dqn_scaffold_30-2"]
    dqn_models = ["dqn_beta-1", "dqn_beta-2", "dqn_beta-3", "dqn_beta-4", "dqn_beta-5"]

    plot_scaffold_durations_multiple_models(dqn_models_old, steps=False)












