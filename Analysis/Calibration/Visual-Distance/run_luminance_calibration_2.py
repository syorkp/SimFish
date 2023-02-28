import os
import sys
import json
import copy
import matplotlib.pyplot as plt
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

np.random.seed(404)

dirname = os.path.dirname(__file__)
file_path = os.path.join(dirname, f"Configurations/Assay-Configs/{arg}_env.json")

n_steps = 3000

with open(file_path, 'r') as f:
    env = json.load(f)
env["light_gain"] = 1.0
env["bkg_scatter"] = 0.00051

env2 = copy.deepcopy(env)
env2["bkg_scatter"] = 0
env2["shot_noise"] = False

env3 = copy.deepcopy(env)
env3["light_gain"] = 1.0

env4 = copy.deepcopy(env3)
env4["bkg_scatter"] = 0
env4["shot_noise"] = False

sim_state_1 = ContinuousNaturalisticEnvironment(env, using_gpu=False)
sim_state_2 = ContinuousNaturalisticEnvironment(env2,
                                                using_gpu=False)
sim_state_3 = ContinuousNaturalisticEnvironment(env3,
                                                using_gpu=False)
sim_state_4 = ContinuousNaturalisticEnvironment(env4,
                                                using_gpu=False)
q = False
d = False

sim_state_1.reset()
sim_state_2.reset()
sim_state_3.reset()
sim_state_4.reset()

sim_state_1.fish.body.position = np.array([750, 750])
sim_state_2.fish.body.position = np.array([750, 750])
sim_state_3.fish.body.position = np.array([750, 750])
sim_state_4.fish.body.position = np.array([750, 750])

sim_state_1.fish.body.angle = 0
sim_state_2.fish.body.angle = 0
sim_state_3.fish.body.angle = 0
sim_state_4.fish.body.angle = 0


# Run simulation randomly, gathering data of positions of features and corresponding SNR
impulses = np.random.random_sample(n_steps) * 10
angles = (np.random.random_sample(n_steps) - 0.5) * 2


# Note is percentage of stimulus that is signal, rather than the SNR.
distances = []

stimulus_present_1 = []
stimulus_present_2 = []
stimulus_absent_1 = []
stimulus_absent_2 = []

distances_stimulus_present_1 = []
distances_stimulus_present_2 = []
distances_stimulus_absent_1 = []
distances_stimulus_absent_2 = []

for t in range(n_steps):
    print(t)
    # print(sim_state_3.prey_bodies[-1].position)
    # print(sim_state_4.prey_bodies[-1].position)
    impulse = float(impulses[t])
    angle = float(angles[t])

    # Light gain 1.0
    o, r, internal, d, fb = sim_state_1.simulation_step([impulse, angle])
    o2, r2, internal2, d2, fb2 = sim_state_2.simulation_step([impulse, angle])

    # Light gain 0.33
    o3, r3, internal3, d3, fb3 = sim_state_3.simulation_step([impulse, angle])
    # o4, r4, internal4, d4, fb4 = sim_state_4.simulation_step([impulse, angle])

    stimuli_present = o2 > 0

    fish_position = sim_state_1.fish.body.position

    distance = ((fish_position[0] - sim_state_1.prey_bodies[-1].position[0]) ** 2 +
                (fish_position[1] - sim_state_1.prey_bodies[-1].position[1]) ** 2) ** 0.5

    stimulus_point_magnitude = o[:, 1, :][stimuli_present[:, 1, :]]
    no_stimulus_point_magnitude = o[:, 1, :][~stimuli_present[:, 1, :]]
    stimulus_point_magnitude_2 = o3[:, 1, :][stimuli_present[:, 1, :]]
    no_stimulus_point_magnitude_2 = o3[:, 1, :][~stimuli_present[:, 1, :]]

    distances.append(distance)

    stimulus_present_1 = np.concatenate((stimulus_present_1, stimulus_point_magnitude), axis=0)
    stimulus_present_2 = np.concatenate((stimulus_present_2, stimulus_point_magnitude_2), axis=0)
    stimulus_absent_1 = np.concatenate((stimulus_absent_1, no_stimulus_point_magnitude), axis=0)
    stimulus_absent_2 = np.concatenate((stimulus_absent_2, no_stimulus_point_magnitude_2), axis=0)

    distances_stimulus_present_1 = np.concatenate((distances_stimulus_present_1, np.array([distance for i in range(len(stimulus_point_magnitude))])))
    distances_stimulus_present_2 = np.concatenate((distances_stimulus_present_2, np.array([distance for i in range(len(stimulus_point_magnitude_2))])))
    distances_stimulus_absent_1 = np.concatenate((distances_stimulus_absent_1, np.array([distance for i in range(len(no_stimulus_point_magnitude))])))
    distances_stimulus_absent_2 = np.concatenate((distances_stimulus_absent_2, np.array([distance for i in range(len(no_stimulus_point_magnitude_2))])))

    if d:
        sim_state_1.reset()
        sim_state_2.reset()
        sim_state_3.reset()
        sim_state_4.reset()


# stimulus_present_1 = np.array(stimulus_present_1)
# stimulus_present_2 = np.array(stimulus_present_2)
# stimulus_absent_1 = np.array(stimulus_absent_1)
# stimulus_absent_2 = np.array(stimulus_absent_2)
# distances_stimulus_present_1 = np.array(distances_stimulus_present_1)
# distances_stimulus_present_2 = np.array(distances_stimulus_present_2)
# distances_stimulus_absent_1 = np.array(distances_stimulus_absent_1)
# distances_stimulus_absent_2 = np.array(distances_stimulus_absent_2)

# Note is percentage of stimulus that is signal, rather than the SNR.
with open(f"Analysis/Calibration/LuminanceCalibration2/stimulus_present-L{env['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(stimulus_present_1))

with open(f"Analysis/Calibration/LuminanceCalibration2/stimulus_present-L{env3['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(stimulus_present_2))

with open(f"Analysis/Calibration/LuminanceCalibration2/stimulus_absent-L{env['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(stimulus_absent_1))

with open(f"Analysis/Calibration/LuminanceCalibration2/stimulus_absent-L{env3['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(stimulus_absent_2))


with open(f"Analysis/Calibration/LuminanceCalibration2/distances_stimulus_present_1-L{env['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(distances_stimulus_present_1))

with open(f"Analysis/Calibration/LuminanceCalibration2/distances_stimulus_present_2-L{env['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(distances_stimulus_present_2))

with open(f"Analysis/Calibration/LuminanceCalibration2/distances_stimulus_absent_1-L{env['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(distances_stimulus_absent_1))

with open(f"Analysis/Calibration/LuminanceCalibration2/distances_stimulus_absent_2-L{env['light_gain']}-BK{env['bkg_scatter']}.npy", "wb") as f:
    np.save(f, np.array(distances_stimulus_absent_2))



