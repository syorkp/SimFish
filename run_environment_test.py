import os
import sys
import json
import time

import numpy as np

from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Environment.controlled_stimulus_environment import ControlledStimulusEnvironment
from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment

"""
Due to PyCharm plots error, currently needs to be run in terminal"""

try:
    arg = sys.argv[1]
except IndexError:
    arg = "continuous_assay"  # Default arg

stimuli = {"prey 1": [
                        {"step": 0,
                         "position": [100, 100]},
                        {"step": 20,
                         "position": [300, 100]},
                        {"step": 40,
                         "position": [300, 300]},
                        {"step": 60,
                         "position": [100, 300]},
                    ]
                }

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, f"Configurations/Assay-Configs/{arg}_env.json")

with open(file_path, 'r') as f:
    env = json.load(f)

# sim_state = ProjectionEnvironment(env, stimuli, tethered=True, draw_screen=True)
# sim_state = NaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True)
# sim_state = ContinuousNaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True, new_simulation=False, using_gpu=False)
sim_state = ContinuousNaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True, new_simulation=True, using_gpu=False)

q = False
d = False
sim_state.reset()
while not q:
    # action = None
    # key = input()
    # action_input = int(key)
    #
    impulse = input()
    angle = input()

    # impulse = 0
    # angle = 0
    # time.sleep(0.1)

    impulse = float(impulse)
    angle = float(angle)

    s, r, internal, d, fb = sim_state.simulation_step([impulse, angle])
    position = sim_state.fish.body.position
    distance = ((position[0] - sim_state.prey_bodies[-1].position[0]) ** 2 +
                (position[1] - sim_state.prey_bodies[-1].position[1]) ** 2) ** 0.5
    l_uv = s[:, 1, 0]
    r_uv = s[:, 1, 1]

    print(sim_state.prey_bodies[-1].position)
    print(sim_state.fish.body.position)
    print(f"Distance: {distance}")
    print(np.max(s[:, 1, :]))
    print(f"Max stimulus at L: {np.argmax(s[:, 1, 0])} and R: {np.argmax(s[:, 1, 1])}")
    print("\n")

    # if angle > 1.0:
    #     sim_state.reset()

    if d:
        sim_state.reset()

    # print(f"Distance moved: {np.sqrt((sim_state.fish.body.position[0]-previous_position[0])**2 + np.sqrt((sim_state.fish.body.position[1]-previous_position[1])**2))}")
