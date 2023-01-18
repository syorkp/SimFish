import numpy as np


def remove_near_wall_data_from_position_data(fish_positions, width, height, buffer, *args):
    fish_positions = np.array(fish_positions)
    away_from_wall = (fish_positions > buffer) * (fish_positions < width - buffer)
    x = np.array(args[0])
    to_return = [np.array(k)[away_from_wall[:, 0]] for k in args]
    to_return = [fish_positions] + to_return
    return to_return