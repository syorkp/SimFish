import os
import sys
import json

from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.virtual_virtual_reality import VirtualVirtualReality
from Environment.controlled_stimulus_environment import ProjectionEnvironment

"""
Due to PyCharm plots error, currently needs to be run in terminal"""

try:
    arg = sys.argv[1]
except IndexError:
    arg = "base"  # Default arg

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
file_path = os.path.join(dirname, f"Configurations/JSON-Data/{arg}_env.json")

with open(file_path, 'r') as f:
    env = json.load(f)

# sim_state = ProjectionEnvironment(env, stimuli, tethered=True, draw_screen=True)
sim_state = NaturalisticEnvironment(env, draw_screen=True)

q = False
d = False
sim_state.reset()
while not q:
    # action = None
    key = input()
    action_input = int(key)

    if action_input < 7:
        s, r, internal, d, fb = sim_state.simulation_step(action_input)

    if action_input == 7:
        q = True

    if action_input == 9:
        sim_state.reset()

    if d:
        sim_state.reset()
