import numpy as np


def generate_random_fish_position_data(n=1000, w=1500, h=1500):
    x = np.random.randint(0, w, n)
    y = np.random.randint(0, h, n)
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)
    fish_positions = np.concatenate((x, y), axis=1)
    return fish_positions
