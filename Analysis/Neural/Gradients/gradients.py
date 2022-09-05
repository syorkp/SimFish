"""
Idea is to use tf.gradients to see what the influence of different inputs on each neuron is with respect to the inputs.

Ways to determine inputs:
- Average over full episode.
- Average over specific contexts.
"""

import os
import json

import numpy as np
import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data
from Analysis.Model.build_network import build_network_dqn
from Analysis.load_model_config import load_configuration_files
from Analysis.Neural.Gradients.get_average_inputs import get_most_common_network_inputs_from_data

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment




def get_target_unit(network, target_layer, i):
    if target_layer == "rnn":
        particular_unit = network.rnn[:, i]
    elif target_layer == "conv1l":
        particular_unit = network.conv1l[:, i]
    elif target_layer == "conv2l":
        particular_unit = network.conv2l[:, i]
    elif target_layer == "conv3l":
        particular_unit = network.conv3l[:, i]
    elif target_layer == "conv4l":
        particular_unit = network.conv4l[:, i]
    elif target_layer == "conv1r":
        particular_unit = network.conv1r[:, i]
    elif target_layer == "conv2r":
        particular_unit = network.conv2r[:, i]
    elif target_layer == "conv3r":
        particular_unit = network.conv3r[:, i]
    elif target_layer == "conv4r":
        particular_unit = network.conv4r[:, i]
    elif target_layer == "Advantage":
        particular_unit = network.Advantage[:, i]
    elif target_layer == "Q_out":
        particular_unit = network.Q_out[:, i]
    elif target_layer == "convFlat":
        particular_unit = network.convFlat[:, i]
    else:
        print("Unrecognised target layer.")
    return particular_unit


def get_num_target_units(params, network, target_layer):
    params["rnn_dim_shared"]
    if target_layer == "rnn":
        num_units = network.rnn.shape[-1]
    elif target_layer == "conv1l":
        num_units = network.conv1l.shape[-1]
    elif target_layer == "conv2l":
        num_units = network.conv2l.shape[-1]
    elif target_layer == "conv3l":
        num_units = network.conv3l.shape[-1]
    elif target_layer == "conv4l":
        num_units = network.conv4l.shape[-1]
    elif target_layer == "conv1r":
        num_units = network.conv1r.shape[-1]
    elif target_layer == "conv2r":
        num_units = network.conv2r.shape[-1]
    elif target_layer == "conv3r":
        num_units = network.conv3r.shape[-1]
    elif target_layer == "conv4r":
        num_units = network.conv4r.shape[-1]
    elif target_layer == "Advantage":
        num_units = 1
    elif target_layer == "Q_out":
        num_units = 1
    elif target_layer == "convFlat":
        num_units = network.convFlat.shape[-1]
    else:
        print("Unrecognised target layer.")
    return num_units


def save_all_gradients(model_name, target_layer, dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnns):
    if not os.path.exists("./Gradients-Data/"):
        os.makedirs("./Gradients-Data/")

    if not os.path.exists(f"./Gradients-Data/{model_name}/"):
        os.makedirs(f"./Gradients-Data/{model_name}/")

    json_format = {
        "dY_dOBS": dy_dobs.tolist(),
        "dY_dEFF": dy_deff.tolist(),
        "dY_dLIT": dy_dlight.tolist(),
        "dY_dENR": dy_denergy.tolist(),
        "dY_dSLT": dy_dsalt.tolist(),
        "dY_dRNN": dy_drnns.tolist()
    }

    with open(f"./Gradients-Data/{model_name}/{target_layer}.json", "w") as outfile:
        json.dump(json_format, outfile, indent=4)


def compute_gradient_for_input(model_name, observation, energy_state, salt_input, efference, in_light, rnn_state,
                               dqn=True, full_reafference=True, target_layer="rnn", save_gradients=True):
    model_location = f"../../../Training-Output/{model_name}"
    params, environment_params, _, _, _ = load_configuration_files(model_name)

    sess = tf.Session()
    with sess as sess:
        if dqn:
            simulation = DiscreteNaturalisticEnvironment(environment_params, True, True, False)
            network = build_network_dqn(environment_params, params, simulation, full_reafference=full_reafference)
        else:
            print("Error, not built for PPO yet")

        saver = tf.train.Saver(max_to_keep=5)
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # TODO: Build system for targeting specific layers - can input though keys.
        # TODO: Build the same for the dynamic naming system.

        # Gradients with respect to each input
        unit_gradients_obs = {}
        unit_gradients_efference = {}
        unit_gradients_internal_state = {}
        unit_gradients_rnn_state = {}

        num_target_units = get_num_target_units(params, network, target_layer)

        if environment_params["use_dynamic_network"]:
            print("Error, dynamic version not built")
        else:
            for i in range(num_target_units):
                particular_unit = get_target_unit(network, target_layer, i)

                unit_gradients_obs[f"Unit {i}"] = tf.gradients(particular_unit, network.observation)
                unit_gradients_internal_state[f"Unit {i}"] = tf.gradients(particular_unit, network.internal_state)
                unit_gradients_efference[f"Unit {i}"] = tf.gradients(particular_unit, network.prev_actions_one_hot)
                unit_gradients_rnn_state[f"Unit {i}"] = tf.gradients(particular_unit, network.rnn_state_in)

        unit_gradients_obs_vals = [unit_gradients_obs[key][0] for key in unit_gradients_obs.keys()]
        unit_gradients_internal_state_vals = [unit_gradients_internal_state[key][0] for key in
                                              unit_gradients_internal_state.keys()]
        unit_gradients_efference_vals = [unit_gradients_efference[key][0] for key in unit_gradients_efference.keys()]
        unit_gradients_rnn_state_vals = [unit_gradients_rnn_state[key][0] for key in unit_gradients_rnn_state.keys()]

        if None in unit_gradients_obs_vals:
            dy_dobs = np.array([])
        else:
            dy_dobs = sess.run([unit_gradients_obs[unit] for unit in unit_gradients_obs.keys()],
                               feed_dict={network.observation: observation,
                                          network.prev_actions: [efference],
                                          network.internal_state: [[in_light, energy_state, salt_input]],
                                          network.batch_size: 1,
                                          network.trainLength: 1,
                                          network.rnn_state_in: rnn_state
                                          }
                               )

        if None in unit_gradients_internal_state_vals:
            dy_dis = np.array([])
        else:
            dy_dis = sess.run([unit_gradients_internal_state[unit] for unit in unit_gradients_internal_state.keys()],
                          feed_dict={network.observation: observation,
                                     network.prev_actions: [efference],
                                     network.internal_state: [[in_light, energy_state, salt_input]],
                                     network.batch_size: 1,
                                     network.trainLength: 1,
                                     network.rnn_state_in: rnn_state
                                     }
                          )
        if None in unit_gradients_efference_vals:
            dy_deff = np.array([])
        else:
            dy_deff = sess.run([unit_gradients_efference[unit] for unit in unit_gradients_efference.keys()],
                           feed_dict={network.observation: observation,
                                      network.prev_actions: [efference],
                                      network.internal_state: [[in_light, energy_state, salt_input]],
                                      network.batch_size: 1,
                                      network.trainLength: 1,
                                      network.rnn_state_in: rnn_state
                                      }
                           )
        if None in unit_gradients_rnn_state_vals:
            dy_drnns = np.array([])
        else:
            dy_drnns = sess.run([unit_gradients_rnn_state[unit] for unit in unit_gradients_rnn_state.keys()],
                           feed_dict={network.observation: observation,
                                      network.prev_actions: [efference],
                                      network.internal_state: [[in_light, energy_state, salt_input]],
                                      network.batch_size: 1,
                                      network.trainLength: 1,
                                      network.rnn_state_in: rnn_state
                                      }
                           )

    # Tidy and return all gradients.
    dy_dobs = np.array([v[0] for v in dy_dobs])
    dy_deff = np.array([v[0][0] for v in dy_deff])
    dy_dis = np.array([v[0][0] for v in dy_dis])
    dy_drnns = np.array([[v[0][0], v[1][0]] for v in dy_drnns])
    try:
        dy_dlight = dy_dis[:, 0]
        dy_denergy = dy_dis[:, 1]
        dy_dsalt = dy_dis[:, 2]
    except IndexError:
        dy_dlight = dy_dis
        dy_denergy = dy_dis
        dy_dsalt = dy_dis

    if save_gradients:
        save_all_gradients(model_name, target_layer, dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnns)

    return dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnns


if __name__ == "__main__":
    model_name = "dqn_scaffold_14-1"
    data = load_data(model_name, "Behavioural-Data-Free", "Naturalistic-1")
    mean_observation, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light, mean_rnn_state = get_most_common_network_inputs_from_data(
        data)
    dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnn = compute_gradient_for_input(model_name, mean_observation,
                                                                                   mean_energy_state, mean_salt_input,
                                                                                   inputted_action,
                                                                                   inputted_in_light,
                                                                                   mean_rnn_state,
                                                                                   full_reafference=False,
                                                                                   target_layer="Advantage",
                                                                                   )
    # Load full graph
    # Compute tf.gradients for activity of specific neurons with respect to inputs.
    # See how this differs for different behavioural contexts
    ...
