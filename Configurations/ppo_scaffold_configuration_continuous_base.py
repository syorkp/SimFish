#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 07:52:17 2020

@author: asaph
"""
import json
import os

# all distances in pixels
import numpy as np

from Utilities.scaffold_creation import create_scaffold, build_changes_list_gradual
from Networks.original_network import connectivity, reflected, base_network_layers, modular_network_layers, ops

ops["join_eyes"][1].remove("prev_action_impulse")
ops["join_eyes"][1].remove("prev_action_angle")

params = {
       # Learning (Universal)
       'batch_size': 1,  # How many experience traces to use for each training step. Works okay with 1.
       'trace_length': 50,  # How long each experience trace will be when training. Works okay with 50
       'num_episodes': 100000,  # How many episodes of game environment to train network with.
       'max_epLength': 5000,  # The max allowed length of our episode.
       'epsilon_greedy': True,
       'startE': 0.2,  # Starting chance of random action
       'endE': 0.01,  # Final chance of random action

       # Learning (DQN Only)
       'update_freq': 100,  # How often to perform a training step.
       'y': .99,  # Discount factor on the target Q-values
       'anneling_steps': 1000000,  # How many steps of training to reduce startE to endE.
       'pre_train_steps': 50000,  # How many steps of random actions before training begins.
       'exp_buffer_size': 500,  # Number of episodes to keep in the experience buffer
       'tau': 0.001,  # target network update time constant

       # Learning (PPO only)
       'n_updates_per_iteration': 4,
       'learning_rate': 0.000001,  # Formerly 0.000001
       'lambda_entropy': 0.00,
       'gamma': 0.99,
       'lambda': 0.9,
       'clip_param': 0.2,
       'max_sigma_impulse': 0.7,  # Formerly 0.4/0.3. Note in the case of static sigma, this is the static value.
       'max_sigma_angle': 0.7,  # Formerly 0.4/0.3. Note in the case of static sigma, this is the static value.
       'min_sigma_impulse': 0.4,  # Formerly 0.1
       'min_sigma_angle': 0.4,  # Formerly 0.1
       'sigma_reduction_time': 5000000,  # Number of steps to complete sigma trajectory.
       'sigma_mode': "Static",  # Options: Decreasing (linear reduction with reduction time), Static

       # Discrete Action Space
       'num_actions': 10,  # size of action space

       # Saving and video parameters
       'time_per_step': 0.03,  # Length of each step used in gif creation
       'summaryLength': 200,  # Number of episodes to periodically save for analysis
       'rnn_dim_shared': 512,  # number of rnn cells

       # Dynamic network construction
       'reflected': reflected,
       'base_network_layers': base_network_layers,
       'modular_network_layers': modular_network_layers,
       'ops': ops,
       'connectivity': connectivity,

       'reuse_eyes': True,

       # Specify how often to save network parameters
       'network_saving_frequency_steps': 100000,

       # Specify how many episodes required before another scaffold switch can occur.
       'min_scaffold_interval': 50,
       'scaffold_stasis_requirement': True,
}

env = {
       #                                     Shared

       # Arena and physical parameters
       'arena_width': 1500,  # arena size
       'arena_height': 1500,
       'drag': 0.7,  # water drag
       'phys_dt': 0.2,  # physics time step
       'phys_steps_per_sim_step': 100,  # number of physics time steps per simulation step. each time step is 2ms
       'displacement_scaling_factor': 0.018,  # Multiplied by previous impulse to displace nearby features.
       'known_max_fish_i': 100,

       # Illumination Parameters
       'dark_light_ratio': 0.0,  # fraction of arena in the dark
       'light_gradient': 0,
       'background_brightness': 0.0,  # base brightness of the background FORMERLY 0.00001; 0.01
       'dark_gain': 60.0,  # gain of brightness in the dark side
       'light_gain': 200.0,  # gain of brightness in the bright side
       'light_decay_rate': 0.01,  # Formerly 0.0006
       'shot_noise': False,  # Whether to model observation of individual photons as a poisson process.

       # Fish specification
       'fish_mass': 140.,
       'fish_mouth_radius': 8.,  # FINAL VALUE - 0.2mm diameter, so 1.
       'fish_head_radius': 2.5,  # Old - 10
       'fish_tail_length': 41.5,  # Old: 70
       'eyes_verg_angle': 77.,  # in deg
       'visual_field': 163.,  # single eye angular visual field
       'eyes_biasx': 2.5,  # distance of eyes from midline - interretinal distance of 0.5mm

       # Fish Visual System
       'uv_photoreceptor_rf_size': 0.0133 * 3,  # Radians (0.76 degrees) - Yoshimatsu et al. (2019)
       'red_photoreceptor_rf_size': 0.0133 * 3,  # Kept same
       'uv_photoreceptor_num': 55,  # Computed using density from 2400 in full 2D retina. Yoshimatsu et al. (2020)
       'red_photoreceptor_num': 63,
       "sz_rf_spacing": 0.04,  # 2.3 deg
       "sz_size": 1.047,  # 60 deg
       "sz_oversampling_factor": 2.5,
       "sigmoid_steepness": 5.0,
       'red_scaling_factor': 1,  # Pixel counts are multiplied by this
       'uv_scaling_factor': 1,  # Pixel counts are multiplied by this
       'red_2_scaling_factor': 0.2,  # Pixel counts are multiplied by this

       # Fish-Paramecium Capture restrictions
       'fraction_capture_permitted': 1.0,  # Should be 1.0 if no temporal restriction imposed.
       'capture_angle_deviation_allowance': np.pi,  # The possible deviation from 0 angular distance of collision
       # between prey and fish, where pi would be allowing capture from any angle.

       # Max Impulse and Angle (for continuous action space)
       'max_angle_change': 1,  # Final 4, Formerly np.pi / 5,
       'max_impulse': 5.0,  # Final 100

       # Fish Motor effect noise (for continuous action space)
       'impulse_effect_noise_sd_x': 0.1,  # 0.98512558,
       'impulse_effect_noise_sd_c': 0.5,  # 0.06,
       'angle_effect_noise_sd_x': 0.6,  # 0.86155083,
       'angle_effect_noise_sd_c': 0.02,  # 0.0010472,

       # Paramecia Specification
       'prey_mass': 1.,
       'prey_inertia': 40.,
       'prey_radius': 1.,  # FINAL VALUE - 0.1mm diameter, so 1.
       'prey_radius_visualisation': 4.,  # Prey size for visualisation purposes
       'prey_num': 30,
       'prey_sensing_distance': 20,
       'prey_max_turning_angle': 0.25,  # Max turn (radians) that happens every step, to replicate linear wavy movement.
       'prey_fluid_displacement': False,
       'prey_jump': False,
       'differential_prey': True,
       'fixed_prey_distribution': False,
       'prey_cloud_num': 7,
       'p_slow': 1.0,
       'p_fast': 0.0,
       'p_escape': 0.5,
       'p_switch': 0.01,  # Corresponds to 1/average duration of movement type.
       'p_reorient': 0.04,
       'slow_impulse_paramecia': 0.0035,  # Impulse to generate 0.5mms-1 for given prey mass
       'fast_impulse_paramecia': 0.007,  # Impulse to generate 1.0mms-1 for given prey mass
       'jump_impulse_paramecia': 0.07,  # Impulse to generate 10.0mms-1 for given prey mass
       'prey_reproduction_mode': True,
       'birth_rate': 0.002,  # Probability per step of new prey appearing at each source.
       'birth_rate_current_pop_scaling': 1,  # Sets scaling of birth rate according to number of prey currently present
       'prey_cloud_region_size': 240,  # Same square as before for simplicity
       'prey_safe_duration': 100,
       'p_prey_death': 0.001,

       # Predator Specification
       'predator_mass': 200.,
       'predator_inertia': 0.0001,
       'predator_radius': 32,
       'predator_impulse': 25,  # To produce speed of 13.7mms-1, formerly 1.0
       'immunity_steps': 65,  # number of steps in the beginning of an episode where the fish is immune from predation
       'distance_from_fish': 181.71216,  # Distance from the fish at which the predator appears. Formerly 498
       'probability_of_predator': 0.0,  # Probability with which the predator appears at each step.

       # Sand Grain Specification
       'sand_grain_mass': 1.,
       'sand_grain_inertia': 40.,
       'sand_grain_radius': 4.,
       'sand_grain_num': 0,
       'sand_grain_displacement_impulse_scaling_factor': 0.5,
       'sand_grain_displacement_distance': 20,
       'sand_grain_red_component': 2.0,
       'sand_grain_touch_penalty': 20000,

       # Reward and Penalties
       'capture_swim_extra_cost': 0,
       'predator_cost': 50000,
       'predator_avoidance_reward': 20000,
       'baseline_penalty': 0.002,
       'reward_distance': 100,
       'proximity_reward': 0.002,
       'salt_reward_penalty': 0,  # Scales with salt concentration.
       'action_reward_scaling': 0,  # Arbitrary (practical) hyperparameter for penalty for action
       'consumption_reward_scaling': 1000000,  # Arbitrary (practical) hyperparameter for reward for consumption
       'wall_touch_penalty': 0.2,

       # Internal State Variables
       'stress': False,
       'stress_compound': 0.9,
       'energy_state': True,
       'in_light': False,
       'salt': True,  # Inclusion of olfactory salt input and salt death.
       'salt_concentration_decay': 0.002,  # Scale for exponential salt concentration decay from source.
       'salt_recovery': 0.005,  # Amount by which salt health recovers per step
       'max_salt_damage': 0.0,  # Salt damage at centre of source.

       # Energy state
       'action_energy_use_scaling': "Sublinear",  # Options: Nonlinear, linear, sublinear.
       'i_scaling_energy_cost': 1.5e-04,  # 0.0004 in previous best.
       'a_scaling_energy_cost': 1.5e-04,  # 0.0004 in previous best.
       'baseline_energy_use': 0.0002,
       'consumption_energy_gain': 1.0,

       # Wall Interaction
       'wall_reflection': True,

       # Currents
       'current_setting': False,  # Current setting. If none, should be False. Current options: Circular, Linear
       'max_current_strength': 0.04,  # Arbitrary impulse variable to be calibrated
       'current_width': 0.2,
       'current_strength_variance': 1,
       'unit_circle_diameter': 0.7,  # Circular current options
}

scaffold_name = "ppo_gamma"

# env["probability_of_predator"] = 0.003
env["immunity_steps"] = 200
env["distance_from_fish"] *= 4

# 2-3
changes = [
       ["PCI", 0.3, "anneling_steps", 500000],

       # 1) Rewards and Penalties
       ["PCI", 0.3, "wall_reflection", False],
]

# 4-11
changes += build_changes_list_gradual("PCI", 0.2, "fraction_capture_permitted", env["fraction_capture_permitted"], 0.25,
                                      8, discrete=False)


changes += [
       # 2) Visual System
       ["PCI", 0.3, "red_photoreceptor_rf_size", 0.0133 * 2],
       ["PCI", 0.3, "uv_photoreceptor_rf_size", 0.0133 * 2],
       ["PCI", 0.3, "red_photoreceptor_rf_size", 0.0133 * 1],
       ["PCI", 0.3, "uv_photoreceptor_rf_size", 0.0133 * 1],
       ["PCI", 0.3, "shot_noise", True],
       ["PCI", 0.3, "background_brightness", 0.1],
]


changes += build_changes_list_gradual("PCI", 0.3, "light_gain", env["light_gain"], 125.7, 4, discrete=False)


# 4) Prey Capture
changes += [["PCI", 0.3, "prey_fluid_displacement", True]]
changes += build_changes_list_gradual("PCI", 0.3, "fish_mouth_radius", env["fish_mouth_radius"], 4, 4, discrete=False)
changes += build_changes_list_gradual("PCI", 0.3, "capture_angle_deviation_allowance", env["capture_angle_deviation_allowance"],
                                      np.pi/5, 8, discrete=False)

# 12-27
# 3) Available actions
changes += build_changes_list_gradual("PCI", 0.3, "max_impulse", env["max_impulse"], 16, 16, discrete=False)
# changes += build_changes_list_gradual("PCI", 0.3, "action_reward_scaling", env["action_reward_scaling"], 10000, 4,
#                                       discrete=False)


changes += [["PCI", 0.4, "anneling_steps", 1000000]]



# 5) Predator avoidance
changes += [["PCI", 0.4, "probability_of_predator", 0.003]]
changes += build_changes_list_gradual("PCI", 0.3, "distance_from_fish", env["distance_from_fish"], env["distance_from_fish"] / 4, 5, discrete=False)

changes += [["PCI", 0.4, "anneling_steps", 500000]]

# 6) Other Behaviours
changes += [["PCI", 0.4, "max_salt_damage", 0.02]]
changes += [["PCI", 0.5, "current_setting", "Circular"]]

# 7) Final Features
# changes += build_changes_list_gradual("PCI", 0.4, "impulse_effect_noise_sd_x", env["impulse_effect_noise_sd_x"], 0.98512558, 8, discrete=False)
# changes += build_changes_list_gradual("PCI", 0.4, "impulse_effect_noise_sd_c", env["impulse_effect_noise_sd_c"], 0.06, 8, discrete=False)
# changes += build_changes_list_gradual("PCI", 0.4, "angle_effect_noise_sd_x", env["angle_effect_noise_sd_x"], 0.86155083, 8, discrete=False)
# changes += build_changes_list_gradual("PCI", 0.4, "angle_effect_noise_sd_c", env["angle_effect_noise_sd_c"], 0.0010472, 8, discrete=False)

finished_condition = {"PCI": 0.3,
                      "PAI": 300.0}

create_scaffold(scaffold_name, env, params, changes, finished_condition)
