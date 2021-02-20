import numpy as np


from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector, get_conv_neuron_vectors
from Analysis.Behavioural.circle_maths import make_circle

def calculate_circle_diameter_for_points(data):
    x_values = [i[0] for i in data]
    y_values = [i[1] for i in data]
    max_x_i = x_values.index(max(x_values))
    min_x_i = x_values.index(min(x_values))
    max_y_i = y_values.index(max(y_values))
    min_y_i = y_values.index(min(y_values))

    ex_indexes = np.unique([max_x_i, min_y_i, min_x_i, max_y_i])
    extremeties = [data[i] for i in ex_indexes]
    r = make_circle(extremeties)[-1]
    return r * 2


def calculate_exploration_index(data, window_size):
    exploration_i = []
    for i in range(window_size, len(data["step"])):
        window_data = data["position"][i-window_size: i]
        exploration_i.append(calculate_circle_diameter_for_points(window_data))
    return exploration_i



data = load_data("large_all_features-1", "Naturalistic", "Naturalistic-1")

ex = calculate_exploration_index(data, 10)

x = True













