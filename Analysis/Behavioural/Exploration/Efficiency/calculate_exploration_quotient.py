import numpy as np


def calculate_exploration_quotient(fish_position, env_width, env_height):
    fish_positions = np.array(fish_position) / 100
    fish_positions = np.around(fish_positions).astype(int)
    grid = np.zeros((int(env_width / 100) + 1, int(env_height / 100) + 1))
    for p in fish_positions:
        grid[p] += 1
    vals = grid[(grid > 0)]
    vals /= fish_positions.shape[0]
    vals = 1 / vals
    exploration_quotient = np.sum(vals)

    return exploration_quotient