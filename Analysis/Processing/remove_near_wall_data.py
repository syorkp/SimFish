
from Analysis.load_data import load_data


def remove_near_wall_data(data, env_x, env_y, buffer=100):
    steps_near_wall = [i for i, position in enumerate(data["position"]) if position[0] < buffer or
                                                                          position[1] < buffer or
                                                                          position[0] > env_x-buffer or
                                                                          position[1] > env_y-buffer]
    # Doesnt work as this, might be best to decide what to do with the data - removal possibly, but would require splitting of the data.
    for key in data.keys():
        if key == "position":
            pass
        else:
            for step in steps_near_wall:
                print(data[key][step])
                data[key][step] = None
    return data


data = load_data("large_all_features-1", "Naturalistic", "Naturalistic-1")
data = remove_near_wall_data(data, 1500, 1500)

x = True
