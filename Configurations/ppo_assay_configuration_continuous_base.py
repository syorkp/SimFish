#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 07:52:17 2020

@author: asaph
"""
import json
import numpy as np

# all distances in pixels


env = {'width': 1500,  # arena size
       'height': 1500,
       'drag': 0.7,  # water drag
       'phys_dt': 0.1,  # physics time step
       'phys_steps_per_sim_step': 100,  # number of physics time steps per simulation step

       'fish_mass': 140.,
       'fish_mouth_size': 4.,
       'fish_head_size': 10.,
       'fish_tail_length': 70.,
       'eyes_verg_angle': 77.,  # in deg
       'visual_field': 163.,  # single eye angular visual field
       'eyes_biasx': 15,  # distance of eyes from midline
       'num_photoreceptors': 120,  # number of visual 'rays' per eye
       'min_vis_dist': 20,
       'max_vis_dist': 180,

       'prey_mass': 1.,
       'prey_inertia': 40.,
       'prey_size': 4.,
       'prey_num': 15,
       'prey_impulse': 0.0,  # impulse each prey receives per step
       'prey_impulse_rate': 0.25,  # fraction of prey receiving impulse per step
       'prey_escape_impulse': 2,
       'prey_sensing_distance': 30,
       'prey_max_turning_angle': 0.3,
       'prey_jump': False,
       'differential_prey': False,
       'prey_cloud_num': 2,

       'sand_grain_mass': 1.,
       'sand_grain_inertia': 40.,
       'sand_grain_size': 4.,
       'sand_grain_num': 0,
       'sand_grain_displacement_impulse_scaling_factor': 0.5,
       'sand_grain_displacement_distance': 20,

       'vegetation_size': 100.,
       'vegetation_num': 0,
       'vegetation_effect_distance': 150,

       'predator_mass': 10.,
       'predator_inertia': 40.,
       'predator_size': 100.,
       'predator_impulse': 1.0,
       'immunity_steps': 65,
       # number of steps in the beginning of an episode where the fish is immune from predation
       'distance_from_fish': 300,  # Distance from the fish at which the predator appears.
       'probability_of_predator': 0.0,  # Probability with which the predator appears at each step.

       'dark_light_ratio': 0.,  # fraction of arena in the dark
       'read_noise_sigma': 5,  # gaussian noise added to photon count
       'photon_ratio': 100,  # expected number of photons for unit brightness
       'bkg_scatter': 0.3,  # base brightness of the background
       'dark_gain': 0.02,  # gai nof brightness in the dark side
       'light_gain': 1.,  # gain of brightness in the bright side

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

       'capture_swim_extra_cost': 25,
       'capture_basic_reward': 0.05,
       'predator_cost': 100,

       'hunger': True,
       'hunger_inc_tau': 0.1,  # fractional increase in hunger per step of not cathing prey
       'hunger_dec_tau': 0.7,  # fractional decrease in hunger when catching prey
       'reafference': False,
       'stress': False,
       'stress_compound': 0.9,

       # For continuous Actions space:
       'max_angle_change': np.pi/5,
       'max_impulse': 10.0,  # Up to 50ish

       'distance_penalty_scaling_factor': 0.001,
       'angle_penalty_scaling_factor': 0.001,
       'baseline_penalty': 0.002,

       # Policy scaffolding
       'reward_distance': 100,
       'proximity_reward': 0.002,

       'max_sigma_impulse': 0.4,
       'max_sigma_angle': 0.4,
       'min_sigma_impulse': 0.1,
       'min_sigma_angle': 0.1,

       'sigma_time_constant': 0.000001,

       'clip_param': 0.2,
       'cs_required': False,

       # New simulation variables
       'decay_rate': 0.01,  # For scatter mask (eyeballed it for practical reasons) # NO DATA YET
       'uv_photoreceptor_rf_size': 0.014,  # Pi Radians (0.8 degrees) - Yoshimatsu et al. (2019)
       'red_photoreceptor_rf_size': 0.01,  # NO DATA YET
       'uv_photoreceptor_num': 55,  # Computed using density from 2400 in full 2D retina. Yoshimatsu et al. (2020)
       'red_photoreceptor_num': 120,  # NO DATA YET
       'shared_photoreceptor_channels': False,
       # Whether the two channels have the same RF angles (saves computation time)
       'incorporate_uv_strike_zone': True,
       'strike_zone_sigma': 1,
       # If there is a strike zone, is standard deviation of normal distribution formed by photoreceptor density.
       'visualise_mask': False,  # For debugging purposes.

       # For dark noise:
       'isomerization_frequency': 1.0,  # Average frequency of photoisomerization per second per photoreceptor
       'max_isomerization_size': 0.01,
       # TODO: Calibrated to produce events of similar size to practical visual distance.
       'sim_steps_per_second': 5,  # For converting isomerization frequency.

       # For extra layer motion:
       'background_grating_frequency': 50,

       # Observation scaling factors (to set CNN inputs into 0 to 1 range):
       'red_scaling_factor': 0.2,  # max was 3.44
       'uv_scaling_factor': 0.2,  # max was 4.1
       'red_2_scaling_factor': 0.01,  # max was 64.2

       'wall_buffer_distance': 40,  # Parameter to avoid visual system errors and prey cloud spawning close to walls.

       'displacement_scaling_factor': 0.005,
       # Multiplied by previous impulse size to cause displacement of nearby features.

       # For new energy state system
       'ci': 0.01,
       'ca': 0.01,
       'cc': 0.2,
       'baseline_decrease': 0.001,
       'trajectory_A': 9.0,
       'trajectory_B': 2.5,

       'action_reward_scaling': 10,  # Arbitrary (practical) hyperparameter for penalty for action
       'consumption_reward_scaling': 100,  # Arbitrary (practical) hyperparameter for reward for consumption

       'energy_state': True,
       # For control of in light:
       'in_light': False,

       # Currents
       'current_setting': "Circular",  # Current setting. If none, should be False. Current options: Circular
       'max_current_strength': 0.01,  # Arbitrary impulse variable to be calibrated
       'current_width': 0.2,
       'current_strength_variance': 1,

       # Circular current options
       'unit_circle_diameter': 0.7,

       # If want complex or simple GIFS:
       'show_channel_sectors': False,
       }


params = {'num_actions': 10,  # size of action space
          'batch_size': 1,  # How many experience traces to use for each training step.
          'trace_length': 50,  # How long each experience trace will be when training
          'update_freq': 100,  # How often to perform a training step.
          'y': .99,  # Discount factor on the target Q-values
          'startE': 0.2,  # Starting chance of random action
          'endE': 0.01,  # Final chance of random action
          'anneling_steps': 1000000,  # How many steps of training to reduce startE to endE.
          'num_episodes': 50000,  # How many episodes of game environment to train network with.
          'pre_train_steps': 50000,  # How many steps of random actions before training begins.
          'max_epLength': 1000,  # The max allowed length of our episode.
          'time_per_step': 0.03,  # Length of each step used in gif creation
          'summaryLength': 200,  # Number of episodes to periodically save for analysis
          'tau': 0.001,  # target network update time constant
          'rnn_dim_shared': 512,  # number of rnn cells
          'extra_rnn': False,

          'learning_rate_actor': 0.000001,
          'learning_rate_critic': 0.000001,

          'n_updates_per_iteration': 5,
          'rnn_state_computation': False,

          'epsilon_greedy': False,
          'multivariate': False,
          'beta_distribution': True,

          'gamma': 0.99,
          'lambda': 0.9,
          'input_sigmas': True

          }

# Equal to that given in the file name.
environment_name = "continuous_assay"
with open(f"Configurations/Assay-Configs/{environment_name}_env.json", 'w') as f:
    json.dump(env, f)

with open(f"Configurations/Assay-Configs/{environment_name}_learning.json", 'w') as f:
    json.dump(params, f)
