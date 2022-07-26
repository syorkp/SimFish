import numpy as np


def label_wall_interaction_steps(data, wall_within_range, environment_size):
    within_range_of_wall = ((data["fish_position"] < wall_within_range) * 1) + ((data["fish_position"] > environment_size - wall_within_range) * 1)
    within_range_of_wall = within_range_of_wall[:, 0] + within_range_of_wall[:, 1]
    within_range_of_wall_timestamps = [i for i, v in enumerate(list(within_range_of_wall)) if v > 0]
    wall_interaction = [1 if i in within_range_of_wall_timestamps else 0 for i in range(len(data["fish_position"]))]
    return np.array(wall_interaction)







