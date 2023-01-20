import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_model_config import load_configuration_files_by_scaffold_point, get_scaffold_num_points
from Environment.Action_Space.action_space_display_comparison import calculate_energy_cost
from Environment.Action_Space.draw_angle_dist_new import draw_angle_dist_new


def plot_log_bout_energy_costs(model_config):
    num_scaffold_points = get_scaffold_num_points(model_config)
    env, params = load_configuration_files_by_scaffold_point(model_config, num_scaffold_points)

    actions = [10, 0, 4, 44, 5, 8, 7]
    action_names = ["AS", "sCS", "J-turn", "J-turn 2", "C-start", "Slow2", "RT"]
    costs = []

    for i, a in enumerate(actions):
        distance, angle = draw_angle_dist_new(a)
        impulse = distance * 3.4452532909386484

        cost = calculate_energy_cost(env, impulse, angle)
        costs.append(abs(cost))

    costs = np.array(costs)
    costs /= np.min(costs)

    colors = ["Purple", "Lightblue", "Green", "Lightgreen", "Red", "Black", "Blue"]

    fig, ax = plt.subplots()
    ax.scatter(costs, [len(action_names)-i for i in range(len(action_names))], c=colors)
    ax.set_xscale('log')

    ax.set_xlabel("Energy Cost (logarithmic)")
    ax.set_ylabel("Bout (ordered)")
    ax.set_yticklabels([" "] + list(reversed(action_names)))

    plt.show()
    x = True



if __name__ == "__main__":
    plot_log_bout_energy_costs("dqn_gamma")



