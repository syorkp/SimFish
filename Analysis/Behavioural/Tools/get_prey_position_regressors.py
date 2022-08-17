import numpy as np

from Analysis.load_data import load_data


def get_egocentric_prey_positions(data):
    """Returns egocentric prey positions, normalised to fish orientation."""

    fish_position = data["fish_position"]
    fish_position = np.expand_dims(fish_position, 1)
    fish_orientations = data["fish_angle"]

    prey_positions = data["prey_positions"]
    relative_prey_positions = prey_positions - fish_position

    x = True


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"
    data = load_data(model_name, "Behavioural-Data-Endless", f"Naturalistic-1")
    get_egocentric_prey_positions(data)