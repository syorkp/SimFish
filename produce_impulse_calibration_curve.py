import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Environment.naturalistic_environment import NaturalisticEnvironment

try:
    arg = str(sys.argv[1])
except IndexError:
    arg = "0"
env = "empty_base"  # Default arg

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, f"Configurations/Single-Configs/{env}_env.json")

with open(file_path, 'r') as f:
    env = json.load(f)


def produce_calibration_curve_data(sim_state):
    sim_state.reset()

    impulses = np.arange(0, 100, 1).tolist()
    distances = []

    sim_state.fish.body.position = (500, 300)
    for i in impulses:    # action = None
        action = 0
        previous_position = sim_state.fish.body.position

        s, r, internal, d, fb = sim_state.simulation_step(action=0, impulse=i)

        distance_moved = np.sqrt((sim_state.fish.body.position[0]-previous_position[0])**2 + np.sqrt((sim_state.fish.body.position[1]-previous_position[1])**2))

        # print(f"Distance moved: {distance_moved}")
        distances.append(distance_moved)
        sim_state.reset()
        sim_state.fish.body.position = (500, 300)
    return distances


def produce_weighted_average(data1, data2, weight_of_1):
    averages = []
    for index, point in enumerate(data1):
        averages.append(((point * weight_of_1) + data2[index])/(1+weight_of_1))
    return averages


def produce_values_for_a_m_value(m):
    s = NaturalisticEnvironment(env, draw_screen=False, fish_mass=m)
    data_a = produce_calibration_curve_data(s)

    for i in range(100):
        s = NaturalisticEnvironment(env, draw_screen=False, fish_mass=m)
        data_b = produce_calibration_curve_data(s)
        data_a = produce_weighted_average(data_a, data_b, i)

    return data_a


mass_range = np.arange(100, 200, 10)

df1 = pd.DataFrame(columns=["Mass", "Impulses", "Distances"])

for m in mass_range:
    print(m)
    distances = produce_values_for_a_m_value(m)
    data = {
        "Mass": [m for i in range(len(distances))],
        "Impulses": np.arange(0, 100, 1).tolist(),
        "Distances": distances,
    }
    df2 = pd.DataFrame(data)
    df1 = pd.concat([df1, df2])

df1.to_csv("out.csv", index=False)
# For mass of 140
#     x = np.arange(0, 100, 1).tolist()
#     plt.clf()
#     plt.plot(x, data_a)
#     plt.title(f"Mass = {m}")
#     x = np.array(x)
#     data_a = np.array(data_a)
#     v = np.polyfit(x, data_a, 1)
#     print(f"Mass: {m}, Params: {v}")

# plt.plot(np.unique(x), np.poly1d(np.polyfit(x, data_a, 1))(np.unique(x)))

# plt.savefig(arg)



