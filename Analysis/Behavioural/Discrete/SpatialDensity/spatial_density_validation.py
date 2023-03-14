import numpy as np
import matplotlib.pyplot as plt

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_fish_prey_incidence import get_fish_prey_incidence_multiple_prey


def get_fish_prey_relative_positions(fish_positions, fish_orientations, prey_positions, accepted_distance=300):
    fish_prey_vectors = prey_positions - np.expand_dims(fish_positions, 1)
    fish_prey_distances = (fish_prey_vectors[:, :, 0] ** 2 + fish_prey_vectors[:, :, 1] ** 2) ** 0.5
    fish_prey_incidence = get_fish_prey_incidence_multiple_prey(fish_positions, fish_orientations, prey_positions)
    fish_prey_y = fish_prey_distances * np.sin(fish_prey_incidence)
    fish_prey_x = fish_prey_distances * np.cos(fish_prey_incidence)
    fish_prey_relative = np.concatenate((np.expand_dims(fish_prey_x, 2),
                                         np.expand_dims(fish_prey_y, 2)), axis=2)
    return fish_prey_relative[fish_prey_distances < accepted_distance]


def plot_all_relative_prey_positions_with_action(model_name, assay_config, assay_id, n):
    """"""
    for action_num in range(0, 12):
        fish_prey_relative_positions = []

        for i in range(1, n + 1):
            data = load_data(model_name, assay_config, f"{assay_id}-{i}")
            steps_with_action = [i for i, a in enumerate(data["action"]) if a == action_num]
            prey_positions = np.array([data["prey_positions"][s] for s in steps_with_action])
            fish_positions = np.array([data["fish_position"][s] for s in steps_with_action])
            fish_orientations = np.array([data["fish_angle"][s] for s in steps_with_action])

            if len(steps_with_action) > 0:
                fish_prey_relative_positions += list(get_fish_prey_relative_positions(fish_positions, fish_orientations, prey_positions))

        fish_prey_relative_positions = np.array(fish_prey_relative_positions)
        plt.scatter(fish_prey_relative_positions[:, 1], fish_prey_relative_positions[:, 0], alpha=0.05)
        plt.savefig(f"Relative Prey: {action_num}")
        plt.clf()
        plt.close()


def plot_all_relative_pred_positions_with_action(model_name, assay_config, assay_id, n):
    """"""
    for action_num in range(0, 12):
        fish_pred_relative_positions = []

        for i in range(1, n + 1):
            data = load_data(model_name, assay_config, f"{assay_id}-{i}")
            steps_with_action = [i for i, a in enumerate(data["action"]) if a == action_num]
            pred_positions = np.array([data["predator_positions"][s] for s in steps_with_action])
            fish_positions = np.array([data["fish_position"][s] for s in steps_with_action])
            fish_orientations = np.array([data["fish_angle"][s] for s in steps_with_action])

            if len(steps_with_action) > 0:
                fish_pred_relative_positions += list(get_fish_prey_relative_positions(fish_positions, fish_orientations, pred_positions))

        fish_pred_relative_positions = np.array(fish_pred_relative_positions)
        plt.scatter(fish_pred_relative_positions[:, 1], fish_pred_relative_positions[:, 0], alpha=0.05)
        plt.savefig(f"Relative Predators: {action_num}")
        plt.clf()
        plt.close()


if __name__ == "__main__":
    plot_all_relative_prey_positions_with_action(f"dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 50)
    plot_all_relative_pred_positions_with_action(f"dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 50)





