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
    arg = None

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


# sim_state = ProjectionEnvironment(env, stimuli, tethered=True, draw_screen=True)
# sim_state = NaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True)
# sim_state = ContinuousNaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True, new_simulation=False, using_gpu=False)

continuous = False


if continuous:
    if arg is None:
        arg = "continuous_assay"  # Default arg
    file_path = os.path.join(dirname, f"Configurations/Assay-Configs/{arg}_env.json")
    with open(file_path, 'r') as f:
        env = json.load(f)
    sim_state = ContinuousNaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True, new_simulation=True, using_gpu=False)
else:
    if arg is None:
        arg = "dqn_26_1_videos"  # Default arg

    file_path = os.path.join(dirname, f"Configurations/Assay-Configs/{arg}_env.json")
    with open(file_path, 'r') as f:
        env = json.load(f)
    env["prey_num"] = 30
    env["prey_cloud_num"] = 5
    env["max_current_strength"] = 0.04
    # env["probability_of_predator"] = 1
    # env["immunity_steps"] = 0
    # env["distance_from_fish"] = 181.71216752587327
    # env["phys_dt"] = 0.2
    # env["predator_mass"] = 200
    # env["predator_inertia"] = 0.0001
    # env["predator_size"] = 32
    # env["predator_impulse"] = 25

    env['prey_mass'] = 1.
    env['prey_inertia'] = 40.
    env['prey_size'] = 1.0  # FINAL VALUE - 0.2mm diameter, so 1.
    env['prey_max_turning_angle'] = 0.25
    env['p_slow'] = 1.0
    env['p_fast'] = 0.0
    env['p_escape'] = 0.5
    env['p_switch'] = 0.01  # Corresponds to 1/average duration of movement type.
    env['p_reorient'] = 0.04
    env['slow_speed_paramecia'] = 0.0  # Impulse to generate 0.5mms-1 for given prey mass
    env['fast_speed_paramecia'] = 0.07  # Impulse to generate 1.0mms-1 for given prey mass
    env['jump_speed_paramecia'] = 0.7  # Impulse to generate 10.0mms-1 for given prey mass
    env['prey_fluid_displacement'] = True
    env["prey_reproduction_mode"] = True

    env['birth_rate'] = 0.1
    env['birth_rate_current_pop_scaling'] = 1
    env['p_prey_death'] = 0.003
    env['prey_safe_duration'] = 100
    env['current_setting'] = "Circular"

    sim_state = DiscreteNaturalisticEnvironment(env, realistic_bouts=True, draw_screen=True, new_simulation=True,
                                                using_gpu=False)


q = False
d = False
sim_state.reset()

if continuous:
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

        # print(f"Prey position: {np.array(sim_state.prey_bodies[-1].position)}")
        # print(f"Fish position: {np.array(sim_state.fish.body.position)}")


        # print(f"Distance: {distance}")
        # print(f"Max UV: {np.max(s[:, 1, :])}")
        # print(f"Max stimulus at L: {np.argmax(s[:, 1, 0])} and R: {np.argmax(s[:, 1, 1])}")
        # print("\n")

        # if angle > 1.0:
        #     sim_state.reset()

        if d:
            sim_state.reset()

        # print(f"Distance moved: {np.sqrt((sim_state.fish.body.position[0]-previous_position[0])**2 + np.sqrt((sim_state.fish.body.position[1]-previous_position[1])**2))}")
else:
    step = 0
    sim_state.fish.body.position = np.array([2025, 1100])
    while not q:
        # action = None

        print(f"{step}: Prey num = {len(sim_state.prey_bodies)}")
        step += 1
        key = 6 #input()
        action_input = int(key)
        s, r, internal, d, fb = sim_state.simulation_step(action_input)

        # if angle > 1.0:
        #     sim_state.reset()
        position = sim_state.fish.body.position
        # print(sim_state.vector_agreement)
        # distance = ((position[0] - sim_state.prey_bodies[-1].position[0]) ** 2 +
        #             (position[1] - sim_state.prey_bodies[-1].position[1]) ** 2) ** 0.5
        # print(f"Distance: {distance}")
        # print(f"Max UV: {np.max(s[:, 1, :])}")

        # print(f"""
        # Red: {np.min(s[:, 0, :])}
        # UV: {np.min(s[:, 1, :])}
        # Red2: {np.min(s[:, 2, :])}
        # """)

        if d:
            sim_state.reset()

        # print(f"Distance moved: {np.sqrt((sim_state.fish.body.position[0]-previous_position[0])**2 + np.sqrt((sim_state.fish.body.position[1]-previous_position[1])**2))}")
