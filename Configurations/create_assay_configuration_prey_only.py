#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 07:52:17 2020

@author: asaph
"""
import json

# all distances in pixels


env = {'width': 1500,  # arena size
       'height': 1500,
       'drag': 0.7,  # water drag
       'phys_dt': 0.1,  # physics time step
       'phys_steps_per_sim_step': 100,  # number of physics time steps per simulation step

       'fish_mass': 140.,
       'fish_mouth_size': 3.,
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
       'prey_num': 30,
       'prey_impulse': 0.1,  # impulse each prey receives per step
       'prey_impulse_rate': 0.25,  # fraction of prey receiving impulse per step
       'prey_escape_impulse': 2,
       'prey_sensing_distance': 30,
       'prey_max_turning_angle': 0.3,
       'prey_jump': True,
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
       'distance_from_fish': 200,  # Distance from the fish at which the predator appears.
       'probability_of_predator': 0.05,  # Probability with which the predator appears at each step.

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
       'capture_basic_reward': 10000,
       'predator_cost': 100,

       'hunger': True,
       'hunger_inc_tau': 0.1,  # fractional increase in hunger per step of not cathing prey
       'hunger_dec_tau': 0.7,  # fractional decrease in hunger when catching prey
       'reafference': False,
       'stress': False,
       'stress_compound': 0.9
       }


params = {'num_actions': 10,  # size of action space
          'batch_size': 16,  # How many experience traces to use for each training step.
          'trace_length': 64,  # How long each experience trace will be when training
          'update_freq': 100,  # How often to perform a training step.
          'y': .99,  # Discount factor on the target Q-values
          'startE': 0.2,  # Starting chance of random action
          'endE': 0.01,  # Final chance of random action
          'anneling_steps': 1000000,  # How many steps of training to reduce startE to endE.
          'num_episodes': 50000,  # How many episodes of game environment to train network with.
          'pre_train_steps': 50000,  # How many steps of random actions before training begins.
          'max_epLength': 1000,  # The max allowed length of our episode.
          'time_per_step': 0.03,  # Length of each step used in gif creation
          'summaryLength': 2000,  # Number of epidoes to periodically save for analysis
          'tau': 0.001,  # target network update time constant
          'rnn_dim': 512,  # number of rnn cells
          'extra_rnn': False,

          'exp_buffer_size': 500,  # Number of episodes to keep in the experience buffer
          'learning_rate': 0.0001}

# Equal to that given in the file name.
environment_name = "even_naturalistic"
# environment_name = "prey_only"
with open(f"Configurations/Assay-Configs/{environment_name}_env.json", 'w') as f:
    json.dump(env, f, indent=4)

with open(f"Configurations/Assay-Configs/{environment_name}_learning.json", 'w') as f:
    json.dump(params, f, indent=4)
