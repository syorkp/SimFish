import numpy as np

from Analysis.load_data import load_data


def remove_near_wall_data(data, env_x, env_y, buffer=200):
    steps_near_wall = [i for i, position in enumerate(data["position"]) if position[0] < buffer or
                                                                          position[1] < buffer or
                                                                          position[0] > env_x-buffer or
                                                                          position[1] > env_y-buffer]
    # Doesnt work as this, might be best to decide what to do with the data - removal possibly, but would require splitting of the data.
    for key in data.keys():
        if key == 'consumed' or key == 'predator':
            pass
        else:
            # for step in reversed(steps_near_wall):
            data[key] = np.delete(data[key], steps_near_wall, axis=0)
    return data


data = load_data("even_prey-1", "Naturalistic", "Naturalistic-1")
data = remove_near_wall_data(data, 1500, 1500)

x = True
