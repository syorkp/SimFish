import os
import sys
import json

from Legacy.sim_state import SimState
from Environment.virtual_virtual_reality import VirtualVirtualReality
from Environment.naturalistic_environment import NaturalisticEnvironment

"""
Due to PyCharm plots error, currently needs to be run in terminal"""

try:
    arg = sys.argv[1]
except IndexError:
    arg = "base"  # Default arg

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, f"Configurations/JSON-Data/{arg}_env.json")

with open(file_path, 'r') as f:
    env = json.load(f)
# TODO: Make possible to test each environment type
sim_state = VirtualVirtualReality(env, draw_screen=True)
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
