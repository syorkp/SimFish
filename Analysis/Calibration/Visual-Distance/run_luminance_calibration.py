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

n_steps = 2000

with open(file_path, 'r') as f:
    env = json.load(f)
env["light_gain"] = 1.0
env["background_brightness"] = 0.0008

env2 = copy.deepcopy(env)
env2["background_brightness"] = 0
env2["shot_noise"] = False

env3 = copy.deepcopy(env)
env3["light_gain"] = 0.2

env4 = copy.deepcopy(env3)
env4["background_brightness"] = 0
env4["shot_noise"] = False

sim_state_1 = ContinuousNaturalisticEnvironment(env, using_gpu=False)
sim_state_2 = ContinuousNaturalisticEnvironment(env2, using_gpu=False)
sim_state_3 = ContinuousNaturalisticEnvironment(env3, using_gpu=False)
sim_state_4 = ContinuousNaturalisticEnvironment(env4, using_gpu=False)
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
red_snr = []
uv_snr = []
uv_distances = []
red2_snr = []
distinguishability_score = []

second_red_snr = []
second_uv_snr = []
second_uv_distances = []
second_red2_snr = []
second_distinguishability_score = []


for t in range(n_steps):
    print(t)
    # print(sim_state_3.prey_bodies[-1].position)
    # print(sim_state_4.prey_bodies[-1].position)
    impulse = float(impulses[t])
    angle = float(angles[t])

    o, r, internal, d, fb = sim_state_1.simulation_step([impulse, angle])
    o2, r2, internal2, d2, fb2 = sim_state_2.simulation_step([impulse, angle])

    o3, r3, internal3, d3, fb3 = sim_state_3.simulation_step([impulse, angle])
    o4, r4, internal4, d4, fb4 = sim_state_4.simulation_step([impulse, angle])

    stimuli_present = o2 > 0

    fish_position = sim_state_1.fish.body.position

    distance = ((fish_position[0] - sim_state_1.prey_bodies[-1].position[0]) ** 2 +
                (fish_position[1] - sim_state_1.prey_bodies[-1].position[1]) ** 2) ** 0.5

    # Light gain 0.5
    noise_uv = np.abs(o2 - (o * stimuli_present))
    snr = noise_uv/(noise_uv+o)
    stim_present_uv = stimuli_present[:, 1, :]
    red_snr.append(np.nanmean(snr[:, 0, :][stimuli_present[:, 0, :]]))
    uv_snr.append(np.nanmean(snr[:, 1, :][stimuli_present[:, 1, :]]))
    red2_snr.append(np.nanmean(snr[:, 2, :][stimuli_present[:, 2, :]]))
    uv_distances.append(distance)

    # Distinguishability score
    magnitude_signal_prs = np.mean((o * stimuli_present)[:, 1, :])
    magnitude_noise_prs = np.mean((o * ~stimuli_present)[:, 1, :])
    diff = np.abs(magnitude_signal_prs - magnitude_noise_prs)
    distinguishability_score.append(diff/magnitude_signal_prs)

    # Light gain 1.0
    noise_uv = np.abs(o4 - (o3 * stimuli_present))
    snr = noise_uv/(noise_uv+o3)
    second_red_snr.append(np.nanmean(snr[:, 0, :][stimuli_present[:, 0, :]]))
    second_uv_snr.append(np.nanmean(snr[:, 1, :][stimuli_present[:, 1, :]]))
    second_red2_snr.append(np.nanmean(snr[:, 2, :][stimuli_present[:, 2, :]]))
    second_uv_distances.append(distance)

    # Distinguishability score
    magnitude_signal_prs = np.mean((o3 * stimuli_present)[:, 1, :])
    magnitude_noise_prs = np.mean((o3 * ~stimuli_present)[:, 1, :])
    diff = np.abs(magnitude_signal_prs - magnitude_noise_prs)
    second_distinguishability_score.append(diff / magnitude_signal_prs)

    if d:
        sim_state_1.reset()
        sim_state_2.reset()


# uv_ok = np.isfinite(uv_snr)
#
# uv_snr = uv_snr[uv_ok]
# distances = uv_distances[uv_ok]
#
#
# z = np.polyfit(distances, uv_snr, 1)
# p = np.poly1d(z)
# plt.scatter(uv_distances, uv_snr)
# plt.plot(distances, p(distances), color="r")
# plt.show()

with open(f"Analysis/Calibration/LuminanceCalibration/UVSNR-L{env['light_gain']}-S{env['light_decay_rate']}-W{env['width']}-BK{env['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(uv_snr))

with open(f"Analysis/Calibration/LuminanceCalibration/UVDistance-L{env['light_gain']}-S{env['light_decay_rate']}-W{env['width']}-BK{env['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(uv_distances))

with open(f"Analysis/Calibration/LuminanceCalibration/RedSNR-L{env['light_gain']}-S{env['light_decay_rate']}-W{env['width']}-BK{env['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(red_snr))

with open(f"Analysis/Calibration/LuminanceCalibration/Red2SNR-L{env['light_gain']}-S{env['light_decay_rate']}-W{env['width']}-BK{env['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(red2_snr))


with open(f"Analysis/Calibration/LuminanceCalibration/Dist-L{env['light_gain']}-S{env['light_decay_rate']}-W{env['width']}-BK{env['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(distinguishability_score))



with open(f"Analysis/Calibration/LuminanceCalibration/UVSNR-L{env3['light_gain']}-S{env3['light_decay_rate']}-W{env3['width']}-BK{env3['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(second_uv_snr))

with open(f"Analysis/Calibration/LuminanceCalibration/UVDistance-L{env3['light_gain']}-S{env3['light_decay_rate']}-W{env3['width']}-BK{env3['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(second_uv_distances))

with open(f"Analysis/Calibration/LuminanceCalibration/RedSNR-L{env3['light_gain']}-S{env3['light_decay_rate']}-W{env3['width']}-BK{env3['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(second_red_snr))

with open(f"Analysis/Calibration/LuminanceCalibration/Red2SNR-L{env3['light_gain']}-S{env3['light_decay_rate']}-W{env3['width']}-BK{env3['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(second_red2_snr))




with open(f"Analysis/Calibration/LuminanceCalibration/Dist-L{env3['light_gain']}-S{env3['light_decay_rate']}-W{env3['width']}-BK{env3['background_brightness']}.npy", "wb") as f:
    np.save(f, np.array(second_distinguishability_score))
