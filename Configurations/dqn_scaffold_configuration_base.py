#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 07:52:17 2020

@author: asaph
"""
import json
import os
import numpy as np

# all distances in pixels

from Utilities.scaffold_creation import create_scaffold, build_changes_list_gradual
from Networks.original_network import connectivity, reflected, base_network_layers, modular_network_layers, ops


params = {
       # Learning (Universal)
       'batch_size': 16,  # How many experience traces to use for each training step.
       'trace_length': 64,  # How long each experience trace will be when training
       'num_episodes': 50000,  # How many episodes of game environment to train network with.
       'max_epLength': 1000,  # The max allowed length of our episode.
       'epsilon_greedy': True,
       'epsilon_greedy_scaffolding': True,
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
       'rnn_state_computation': False,
       'learning_rate': 0.0001,
       'lambda_entropy': 0.01,

       # Learning (PPO Continuous Only)
       'multivariate': False,
       'beta_distribution': False,
       'gamma': 0.99,
       'lambda': 0.9,
       'input_sigmas': True,
       'sigma_scaffolding': False,  # Reset sigma progression if move along configuration scaffold.

       # Discrete Action Space
       'num_actions': 10,  # size of action space

       # Saving and video parameters
       'time_per_step': 0.03,  # Length of each step used in gif creation
       'summaryLength': 200,  # Number of episodes to periodically save for analysis
       'rnn_dim_shared': 512,  # number of rnn cells. Should no longer be used.
       'extra_rnn': False,
       'save_gifs': True,

       # Dynamic network construction
       'reflected': reflected,
       'base_network_layers': base_network_layers,
       'modular_network_layers': modular_network_layers,
       'ops': ops,
       'connectivity': connectivity,

       # For RND
       'use_rnd': False,  # Whether to use RND.
}

env = {
       #                            Old Simulation (Parameters ignored in new simulation)
       'num_photoreceptors': 120,  # number of visual 'rays' per eye
       'min_vis_dist': 20,
       'max_vis_dist': 180,

       'prey_impulse_rate': 0.25,  # fraction of prey receiving impulse per step

       'sand_grain_mass': 1.,
       'sand_grain_inertia': 40.,
       'sand_grain_size': 4.,
       'sand_grain_num': 0,
       'sand_grain_displacement_impulse_scaling_factor': 0.5,
       'sand_grain_displacement_distance': 20,

       'vegetation_size': 100.,
       'vegetation_num': 0,
       'vegetation_effect_distance': 150,

       'photon_ratio': 100,  # expected number of photons for unit brightness

       'forward_swim_cost': 3,
       'forward_swim_impulse': 10,
       'routine_turn_cost': 3,
       'routine_turn_impulse': 5,
       'routine_turn_dir_change': 0.6,
       'capture_swim_cost': 5,
       'capture_swim_impulse': 5,
       'j_turn_cost': 2.5,
       'j_turn_impulse': 0.1,
       'j_turn_dir_change': 0.07,
       'rest_cost': 2,

       'capture_swim_extra_cost': 5,
       'capture_basic_reward': 10000,  # Used only when not using energy state.

       'hunger_inc_tau': 0.1,  # fractional increase in hunger per step of not cathing prey
       'hunger_dec_tau': 0.7,  # fractional decrease in hunger when catching prey

       'distance_penalty_scaling_factor': 1.0,
       # NOTE: THESE ARE IGNORED IN NEW SIMULATION, where penalties are set by energy system.
       'angle_penalty_scaling_factor': 0.5,
       # NOTE: THESE ARE IGNORED IN NEW SIMULATION, where penalties are set by energy system.

       #                                     Shared

       'width': 1500,  # arena size
       'height': 1500,
       'drag': 0.7,  # water drag
       'phys_dt': 0.1,  # physics time step
       'phys_steps_per_sim_step': 100,  # number of physics time steps per simulation step. each time step is 2ms

       'fish_mass': 140.,
       'fish_mouth_size': 8.,  # FINAL VALUE - 0.2mm diameter, so 1.
       'fish_head_size': 2.5,  # Old - 10
       'fish_tail_length': 41.5,  # Old: 70
       'eyes_verg_angle': 77.,  # in deg
       'visual_field': 163.,  # single eye angular visual field
       'eyes_biasx': 2.5,  # distance of eyes from midline - interretinal distance of 0.5mm

       'prey_mass': 1.,
       'prey_inertia': 40.,
       'prey_size': 1.,  # FINAL VALUE - 0.1mm diameter, so 1.
       'prey_size_visualisation': 4.,  # Prey size for visualisation purposes
       'prey_num': 25,
       'prey_impulse': 0.0,  # impulse each prey receives per step
       'prey_escape_impulse': 2,
       'prey_sensing_distance': 20,
       'prey_max_turning_angle': 0.04,
       # This is the turn (pi radians) that happens every step, designed to replicate linear wavy movement.
       'prey_fluid_displacement': False,
       'prey_jump': False,
       'differential_prey': True,
       'prey_cloud_num': 4,

       'predator_mass': 10.,
       'predator_inertia': 40.,
       'predator_size': 43.5,  # To be 8.7mm in diameter, formerly 100
       'predator_impulse': 0.39,  # To produce speed of 13.7mms-1, formerly 1.0
       'immunity_steps': 65,  # number of steps in the beginning of an episode where the fish is immune from predation
       'distance_from_fish': 100,  # Distance from the fish at which the predator appears. Formerly 498
       'probability_of_predator': 0.0,  # Probability with which the predator appears at each step.

       'dark_light_ratio': 0.3,  # fraction of arena in the dark
       'light_gradient': 200,
       'read_noise_sigma': 0.,  # gaussian noise added to photon count. Formerly 5.
       'bkg_scatter': 0.0,  # base brightness of the background FORMERLY 0.00001
       'dark_gain': 60.0,  # gain of brightness in the dark side
       'light_gain': 200.0,  # gain of brightness in the bright side

       'predator_cost': 1000,

       'hunger': False,
       'reafference': True,
       'stress': False,
       'stress_compound': 0.9,

       # For continuous Actions space:
       'max_angle_change': 1,  # Final 4, Formerly np.pi / 5,
       'max_impulse': 100.0,  # Final 100

       'baseline_penalty': 0.002,
       'reward_distance': 100,
       'proximity_reward': 0.002,

       'max_sigma_impulse': 0.3,  # Formerly 0.4
       'max_sigma_angle': 0.3,  # Formerly 0.4
       'min_sigma_impulse': 0.1,
       'min_sigma_angle': 0.1,
       'sigma_reduction_time': 5000000,  # Number of steps to complete sigma trajectory.
       'sigma_mode': "Decreasing",  # Options: Decreasing (linear reduction with reduction time), Static

       'clip_param': 0.2,
       'cs_required': True,

       #                                  New Simulation

       # Action mask
       'impose_action_mask': True,

       # Sensory inputs
       'energy_state': True,
       'in_light': True,
       'salt': True,  # Inclusion of olfactory salt input and salt death.
       'salt_reward_penalty': 10000,  # Scales with salt concentration.
       "use_dynamic_network": True,
       'salt_concentration_decay': 0.002,  # Scale for exponential salt concentration decay from source.
       'salt_recovery': 0.01,  # Amount by which salt health recovers per step
       'max_salt_damage': 0.0,  # Salt damage at centre of source. Before, was 0.02

       # GIFs and debugging
       'visualise_mask': False,  # For debugging purposes.
       'show_channel_sectors': False,
       'show_fish_body_energy_state': False,
       'show_action_space_usage': True,
       'show_previous_actions': 200,  # False if not, otherwise the number of actions to save.

       # Environment
       'decay_rate': 0.01,  # Formerly 0.0006
       'sim_steps_per_second': 5,  # For converting isomerization frequency.
       'background_grating_frequency': 50,  # For extra layer motion:
       'displacement_scaling_factor': 0.018,
       # Multiplied by previous impulse size to cause displacement of nearby features.
       'known_max_fish_i': 100,

       # Prey movement
       'p_slow': 1.0,
       'p_fast': 0.0,
       'p_escape': 0.5,
       'p_switch': 0.01,  # Corresponds to 1/average duration of movement type.
       'p_reorient': 0.04,
       'slow_speed_paramecia': 0.0037,  # Impulse to generate 0.5mms-1 for given prey mass
       'fast_speed_paramecia': 0.0074,  # Impulse to generate 1.0mms-1 for given prey mass
       'jump_speed_paramecia': 0.074,  # Impulse to generate 10.0mms-1 for given prey mass

       # Prey reproduction
       'prey_reproduction_mode': False,
       'birth_rate': 0.1,  # Probability per step of new prey appearing at each source.
       'birth_rate_current_pop_scaling': 1,  # Sets scaling of birth rate according to number of prey currently present
       'birth_rate_region_size': 240,  # Same square as before for simplicity
       'prey_safe_duration': 100,
       'p_prey_death': 0.1,

       # Predators - Repeated attacks in localised region. Note, can make some of these arbitrarily high so predator keeps attacking when fish enters a certain region for whole episode.
       'max_predator_attacks': 5,
       'further_attack_probability': 0.4,
       'max_predator_attack_range': 2000,
       'max_predator_reorient_distance': 400,
       'predator_presence_duration_steps': 100,

       # Predator - Expanding disc
       'predator_first_attack_loom': False,
       'initial_predator_size': 20,  # Size in degrees
       'final_predator_size': 200,  # "
       'duration_of_loom': 10,  # Number of steps for which loom occurs.

       # Visual system scaling factors (to set CNN inputs into 0 to 255 range):
       'red_scaling_factor': 1/5,  # Pixel counts are multiplied by this
       'uv_scaling_factor': 1,  # Pixel counts are multiplied by this
       'red_2_scaling_factor': 1/500.0,  # Pixel counts are multiplied by this
       'red_occlusion_gain': 0.0,  # 0 Being complete construction.
       'uv_occlusion_gain': 1.0,
       'red2_occlusion_gain': 0.0,

       'wall_buffer_distance': 40,  # Parameter to avoid visual system errors and prey cloud spawning close to walls.

       # Arbitrary fish parameters

       # Fish Visual System
       'uv_photoreceptor_rf_size': 0.0133 * 3,  # Pi Radians (0.76 degrees) - Yoshimatsu et al. (2019)
       'red_photoreceptor_rf_size': 0.0133 * 3,  # Kept same
       'uv_photoreceptor_num': 55,  # Computed using density from 2400 in full 2D retina. Yoshimatsu et al. (2020)
       'red_photoreceptor_num': 63,
       'minimum_observation_size': 100,  # Parameter to determine padded observation size (avoids conv layer size bug).
       'shared_photoreceptor_channels': False,  # Whether two channels have the same RF angles (saves computation time)
       'incorporate_uv_strike_zone': True,
       'strike_zone_sigma': 1.5,
       # If there is a strike zone, is standard deviation of normal distribution formed by photoreceptor density.

       # Shot noise
       'shot_noise': False,  # Whether to model observation of individual photons as a poisson process.

       # For dark noise:
       'isomerization_frequency': 0.0,  # Average frequency of photoisomerization per second per photoreceptor
       'max_isomerization_size': 0.0,

       # Energy state and hunger-based rewards
       'ci': 0.0000008,  # Final for sublinear PPO: 0.0003
       'ca': 0.0000008,  # Final for sublinear PPO: 0.0003
       'baseline_decrease': 0.0015,  # Final for sublinear PPO: 0.0015
       'trajectory_A': 5.0,
       'trajectory_B': 2.5,
       'consumption_energy_gain': 1.0,

       # Reward
       'action_reward_scaling': 10000,  # 1942,  # Arbitrary (practical) hyperparameter for penalty for action
       'consumption_reward_scaling': 1000000,  # Arbitrary (practical) hyperparameter for reward for consumption

       'wall_reflection': True,
       'wall_touch_penalty': 2,

       # Currents
       'current_setting': False,  # Current setting. If none, should be False. Current options: Circular, Linear
       'max_current_strength': 0.01,  # Arbitrary impulse variable to be calibrated
       'current_width': 0.2,
       'current_strength_variance': 1,
       'unit_circle_diameter': 0.7,  # Circular current options

       # Motor effect noise (for continuous)
       'impulse_effect_noise_sd_x': 0,  # 0.98512558,
       'impulse_effect_noise_sd_c': 0,  # 0.06,
       'angle_effect_noise_sd_x': 0,  # 0.86155083,
       'angle_effect_noise_sd_c': 0,  # 0.0010472,

       # Complex capture swim dynamics
       'fraction_capture_permitted': 1.0,  # Should be 1.0 if no temporal restriction imposed.
       'capture_angle_deviation_allowance': np.pi,
       # The possible deviation from 0 angular distance of collision between prey and fish, where pi would be allowing capture from any angle.

       'action_energy_use_scaling': "Sublinear",  # Options: Nonlinear, linear, sublinear.

       'max_visual_range': 1500,
}


scaffold_name = "dqn_scaffold_dn_23"

# 2-10
changes = [
       ["PCI", 0.25, "anneling_steps", 500000],
       # 1) Rewards and Penalties
       ["PCI", 0.25, "capture_swim_extra_cost", 50],
       ["PCI", 0.25, "wall_reflection", False],

       # 2) Visual System
       ["PCI", 0.25, "red_photoreceptor_rf_size", 0.0133 * 2],
       ["PCI", 0.25, "uv_photoreceptor_rf_size", 0.0133 * 2],
       ["PCI", 0.25, "red_photoreceptor_rf_size", 0.0133 * 1],
       ["PCI", 0.25, "uv_photoreceptor_rf_size", 0.0133 * 1],
       ["PCI", 0.35, "shot_noise", True],
       ["PCI", 0.35, "bkg_scatter", 0.1],
]

# 11-14
changes += build_changes_list_gradual("PCI", 0.3, "light_gain", env["light_gain"], 125.7, 4)

# 2) Exploration 15-18
changes += [["PCI", 0.35, "max_epLength", 2000, "do_to_params"]]
changes += [["PCI", 0.35, "baseline_decrease", 0.00075]]
changes += [["PCI", 0.35, "width", 3000, "height", 3000]]

# 3) Fine Prey Capture 19-34
changes += [["PCI", 0.35, "prey_fluid_displacement", True]]
changes += [["PCI", 0.35, "prey_jump", True]]
changes += build_changes_list_gradual("PCI", 0.35, "fish_mouth_size", env["fish_mouth_size"], 4, 4)
changes += build_changes_list_gradual("PCI", 0.35, "fraction_capture_permitted", env["fraction_capture_permitted"], 0.5, 4)
changes += build_changes_list_gradual("PCI", 0.35, "capture_angle_deviation_allowance", env["capture_angle_deviation_allowance"], (34*np.pi)/180, 4)
changes += [["PCI", 0.35, "capture_swim_extra_cost", 100]]
changes += [["PCI", 0.35, "anneling_steps", 1000000]]

# 4) Predator avoidance 35
changes += [["PCI", 0.35, "probability_of_predator", 0.01]]
# TODO: Complex predator

# 5) Other Behaviours 36-37
changes += [["PCI", 0.35, "max_salt_damage", 0.02]]
changes += [["PCI", 0.35, "current_setting", "Circular"]]

create_scaffold(scaffold_name, env, params, changes)
