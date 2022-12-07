"""
Idea is to use tf.gradients to see what the influence of different inputs on each neuron is with respect to the inputs.

Ways to determine inputs:
- Average over full episode.
- Average over specific contexts.
"""

import os
import json
import matplotlib.pyplot as plt

import numpy as np
import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data
from Analysis.Model.build_network import build_network_dqn
from Analysis.load_model_config import load_assay_configuration_files
from Analysis.Neural.Gradients.get_average_inputs import get_mean_inputs_for_context, get_all_inputs_for_context

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Environment.Action_Space.draw_angle_dist import get_modal_impulse_and_angle


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
        num_units = 10
    elif target_layer == "Q_out":
        num_units = 10
    elif target_layer == "Q":
        num_units = 1
    elif target_layer == "convFlat":
        num_units = network.convFlat.shape[-1]
    else:
        print("Unrecognised target layer.")
    return num_units


def save_all_inputs(model_name, context_name, observation, rnn_state, energy_state, salt, efference_copy, in_light):
    if not os.path.exists("./Gradients-Data/"):
        os.makedirs("./Gradients-Data/")

    if not os.path.exists(f"./Gradients-Data/{model_name}/"):
        os.makedirs(f"./Gradients-Data/{model_name}/")

    rnn_state = np.array([rnn_state[0], rnn_state[1]])

    json_format = {
        "observation": observation.tolist(),
        "rnn_state": rnn_state.tolist(),
        "energy_state": energy_state.tolist(),
        "salt": salt.tolist(),
        "efference_copy": efference_copy.tolist(),
        "in_light": in_light.tolist(),
    }
    with open(f"./Gradients-Data/{model_name}/Stimuli-{context_name}.json", "w") as outfile:
        json.dump(json_format, outfile, indent=4)


def save_all_gradients(model_name, target_layer, context_name, dy_dobs, dy_deff, dy_deff2, dy_dlight, dy_denergy,
                       dy_dsalt, dy_drnns, method="averaged"):
    if not os.path.exists("./Gradients-Data/"):
        os.makedirs("./Gradients-Data/")

    if not os.path.exists(f"./Gradients-Data/{model_name}/"):
        os.makedirs(f"./Gradients-Data/{model_name}/")

    json_format = {
        "dY_dOBS": dy_dobs.tolist(),
        "dY_dEFF": dy_deff.tolist(),
        "dY_dEFF2": dy_deff2.tolist(),
        "dY_dLIT": dy_dlight.tolist(),
        "dY_dENR": dy_denergy.tolist(),
        "dY_dSLT": dy_dsalt.tolist(),
        "dY_dRNN": dy_drnns.tolist()
    }

    with open(f"./Gradients-Data/{model_name}/Gradients-{target_layer}-{context_name}-{method}.json", "w") as outfile:
        json.dump(json_format, outfile, indent=4)


def compute_gradient_for_input(model_name, observation, energy_state, salt_input, action, in_light, rnn_state,
                               context_name, dqn=True, full_reafference=True, target_layer="rnn", save_gradients=True):
    model_location = f"../../../Training-Output/{model_name}"
    params, environment_params, _, _, _ = load_assay_configuration_files(model_name)

    if full_reafference:
        i, a = get_modal_impulse_and_angle(action)
        efference = [action, i, a]
    else:
        efference = action

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
        unit_gradients_efference_cons = {}
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
                if full_reafference:
                    unit_gradients_efference_cons[f"Unit {i}"] = tf.gradients(particular_unit, network.prev_action_consequences)
                unit_gradients_rnn_state[f"Unit {i}"] = tf.gradients(particular_unit, network.rnn_state_in)

        unit_gradients_obs_vals = [unit_gradients_obs[key][0] for key in unit_gradients_obs.keys()]
        unit_gradients_internal_state_vals = [unit_gradients_internal_state[key][0] for key in
                                              unit_gradients_internal_state.keys()]
        unit_gradients_efference_vals = [unit_gradients_efference[key][0] for key in unit_gradients_efference.keys()]
        unit_gradients_efference_cons_vals = [unit_gradients_efference_cons[key][0] for key in unit_gradients_efference_cons.keys()]
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
                                          network.rnn_state_in: (rnn_state[0], rnn_state[1])
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
                                     network.rnn_state_in: (rnn_state[0], rnn_state[1])
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
                                      network.rnn_state_in: (rnn_state[0], rnn_state[1])
                                      }
                           )
        if None in unit_gradients_efference_cons_vals:
            dy_deff2 = np.array([])
        else:
            dy_deff2 = sess.run([unit_gradients_efference_cons[unit] for unit in unit_gradients_efference_cons.keys()],
                           feed_dict={network.observation: observation,
                                      network.prev_actions: [efference],
                                      network.internal_state: [[in_light, energy_state, salt_input]],
                                      network.batch_size: 1,
                                      network.trainLength: 1,
                                      network.rnn_state_in: (rnn_state[0], rnn_state[1])
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
                                      network.rnn_state_in: (rnn_state[0], rnn_state[1])
                                      }
                           )

    # Tidy and return all gradients.
    dy_dobs = np.array([v[0] for v in dy_dobs])
    dy_deff = np.array([v[0][0] for v in dy_deff])
    dy_deff2 = np.array([v[0][0] for v in dy_deff2])
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
        save_all_gradients(model_name, target_layer, context_name, dy_dobs, dy_deff, dy_deff2, dy_dlight, dy_denergy, dy_dsalt, dy_drnns)
        save_all_inputs(model_name, context_name, observation.astype(float), rnn_state, np.array([energy_state]), np.array([salt_input]),
                        np.array(efference).astype(float), np.array([in_light]))

    return dy_dobs, dy_deff, dy_deff2, dy_dlight, dy_denergy, dy_dsalt, dy_drnns


def compute_average_gradient_many_inputs(model_name, observation, energy_state, salt_input, action, in_light, rnn_state,
                                         context_name, dqn=True, full_reafference=True, target_layer="rnn",
                                         save_gradients=True):
    n_gradients_to_compute = observation.shape[0]
    model_location = f"../../../Training-Output/{model_name}"
    params, environment_params, _, _, _ = load_assay_configuration_files(model_name)

    dy_dobs_compiled, dy_deff_compiled, dy_deff2_compiled, dy_dlight_compiled, dy_denergy_compiled, dy_dsalt_compiled, \
    dy_drnns_compiled = [], [], [], [], [], [], []

    if full_reafference:
        efference =[]
        for ac in action:
            i, a = get_modal_impulse_and_angle(ac)
            efference.append([ac, i, a])
    else:
        efference = action

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
        unit_gradients_efference_cons = {}
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
                if full_reafference:
                    unit_gradients_efference_cons[f"Unit {i}"] = tf.gradients(particular_unit, network.prev_action_consequences)
                unit_gradients_rnn_state[f"Unit {i}"] = tf.gradients(particular_unit, network.rnn_state_in)

        unit_gradients_obs_vals = [unit_gradients_obs[key][0] for key in unit_gradients_obs.keys()]
        unit_gradients_internal_state_vals = [unit_gradients_internal_state[key][0] for key in
                                              unit_gradients_internal_state.keys()]
        unit_gradients_efference_vals = [unit_gradients_efference[key][0] for key in unit_gradients_efference.keys()]
        unit_gradients_efference_cons_vals = [unit_gradients_efference_cons[key][0] for key in unit_gradients_efference_cons.keys()]
        unit_gradients_rnn_state_vals = [unit_gradients_rnn_state[key][0] for key in unit_gradients_rnn_state.keys()]

        for n in range(n_gradients_to_compute):
            if None in unit_gradients_obs_vals:
                dy_dobs = np.array([])
            else:
                dy_dobs = sess.run([unit_gradients_obs[unit] for unit in unit_gradients_obs.keys()],
                                   feed_dict={network.observation: observation[n].astype(int),
                                              network.prev_actions: [efference[n]],
                                              network.internal_state: [[in_light[n], energy_state[n], salt_input[n]]],
                                              network.batch_size: 1,
                                              network.trainLength: 1,
                                              network.rnn_state_in: (rnn_state[n, 0], rnn_state[n, 1])
                                              }
                                   )

            if None in unit_gradients_internal_state_vals:
                dy_dis = np.array([])
            else:
                dy_dis = sess.run([unit_gradients_internal_state[unit] for unit in unit_gradients_internal_state.keys()],
                              feed_dict={network.observation: observation[n].astype(int),
                                         network.prev_actions: [efference[n]],
                                         network.internal_state: [[in_light[n], energy_state[n], salt_input[n]]],
                                         network.batch_size: 1,
                                         network.trainLength: 1,
                                         network.rnn_state_in: (rnn_state[n, 0], rnn_state[n, 1])
                                         }
                              )

            if None in unit_gradients_efference_vals:
                dy_deff = np.array([])
            else:
                dy_deff = sess.run([unit_gradients_efference[unit] for unit in unit_gradients_efference.keys()],
                               feed_dict={network.observation: observation[n].astype(int),
                                          network.prev_actions: [efference[n]],
                                          network.internal_state: [[in_light[n], energy_state[n], salt_input[n]]],
                                          network.batch_size: 1,
                                          network.trainLength: 1,
                                          network.rnn_state_in: (rnn_state[n, 0], rnn_state[n, 1])
                                          }
                               )
            if None in unit_gradients_efference_cons_vals:
                dy_deff2 = np.array([])
            else:
                dy_deff2 = sess.run([unit_gradients_efference_cons[unit] for unit in unit_gradients_efference_cons.keys()],
                               feed_dict={network.observation: observation[n].astype(int),
                                          network.prev_actions: [efference[n]],
                                          network.internal_state: [[in_light[n], energy_state[n], salt_input[n]]],
                                          network.batch_size: 1,
                                          network.trainLength: 1,
                                          network.rnn_state_in: (rnn_state[n, 0], rnn_state[n, 1])
                                          }
                               )
            if None in unit_gradients_rnn_state_vals:
                dy_drnns = np.array([])
            else:
                dy_drnns = sess.run([unit_gradients_rnn_state[unit] for unit in unit_gradients_rnn_state.keys()],
                               feed_dict={network.observation: observation[n].astype(int),
                                          network.prev_actions: [efference[n]],
                                          network.internal_state: [[in_light[n], energy_state[n], salt_input[n]]],
                                          network.batch_size: 1,
                                          network.trainLength: 1,
                                          network.rnn_state_in: (rnn_state[n, 0], rnn_state[n, 1])
                                          }
                               )

            # Tidy and return all gradients.
            dy_dobs = np.array([v[0] for v in dy_dobs])
            dy_deff = np.array([v[0][0] for v in dy_deff])
            dy_deff2 = np.array([v[0][0] for v in dy_deff2])
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
            dy_dobs_compiled.append(dy_dobs[0])
            dy_deff_compiled.append(dy_deff[0])
            dy_deff2_compiled.append(dy_deff2[0])
            dy_dlight_compiled.append(dy_dlight[0])
            dy_denergy_compiled.append(dy_denergy[0])
            dy_dsalt_compiled.append(dy_dsalt[0])
            dy_drnns_compiled.append(dy_drnns[0])

    dy_dobs_compiled = np.array(dy_dobs_compiled)
    dy_deff_compiled = np.array(dy_deff_compiled)
    dy_deff2_compiled = np.array(dy_deff2_compiled)
    dy_dlight_compiled = np.array(dy_dlight_compiled)
    dy_denergy_compiled = np.array(dy_denergy_compiled)
    dy_dsalt_compiled = np.array(dy_dsalt_compiled)
    dy_drnns_compiled = np.array(dy_drnns_compiled)

    if save_gradients:
        save_all_gradients(model_name, target_layer, context_name, dy_dobs_compiled, dy_deff_compiled, dy_deff2_compiled,
                           dy_dlight_compiled, dy_denergy_compiled, dy_dsalt_compiled, dy_drnns_compiled, method="all_inputs")
        save_all_inputs(model_name, context_name, observation.astype(int), rnn_state, energy_state, salt_input, efference, in_light)

    return dy_dobs_compiled, dy_deff_compiled, dy_deff2_compiled, dy_dlight_compiled, dy_denergy_compiled, \
           dy_dsalt_compiled, dy_drnns_compiled


def get_inputs_prior_to_capture(data):
    """Returns the inputs in the final step before a capture"""
    desired_timestamps = [i for i, c in enumerate(data["consumed"]) if c]
    t_chosen = desired_timestamps[-1]
    t_chosen = 3

    observation = data["observation"][t_chosen]
    rnn_state = data["rnn_state_actor"][t_chosen]
    energy_state = data["energy_state"][t_chosen]
    salt = data["salt"][t_chosen]
    action = data["action"][t_chosen]
    in_light = data["in_light"][t_chosen]
    return observation, rnn_state, energy_state, salt, action, in_light


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"

    # DOING FOR A SINGLE, WELL DEFINED INPUT
    d = load_data(model_name, "Behavioural-Data-CNN", "Naturalistic-1")
    observation, rnn_state, energy_state, salt, action, in_light = get_inputs_prior_to_capture(d)
    dy_dobs, dy_deff, dy_deff2, dy_dlight, dy_denergy, dy_dsalt, dy_drnn = compute_gradient_for_input(model_name, observation,
                                                                                            energy_state,
                                                                                            salt,
                                                                                            action,
                                                                                            in_light,
                                                                                            rnn_state,
                                                                                            context_name="Random",
                                                                                            full_reafference=True,
                                                                                            target_layer="Advantage",


                                                                                            )

    # plt.imshow(dy_dobs[:, :, :, 0]/np.absolute(np.min(dy_dobs[:, :, :, 0])))
    # plt.savefig("dyObs-stim-left.png")
    # plt.imshow(dy_dobs[:, :, :, 1]/np.absolute(np.min(dy_dobs[:, :, :, 1])))
    # plt.savefig("dyObs-stim-right.png")

    # DOING VIA ALL INPUTS (averaged over all gradients)
    # compiled_observations, compiled_rnn_state, compiled_salt, compiled_energy_state, compiled_actions, \
    # compiled_in_light = get_all_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 40, 1)
    # dy_dobs_compiled, dy_deff_compiled, dy_deff2_compiled, dy_dlight_compiled, dy_denergy_compiled, dy_dsalt_compiled, \
    # dy_drnns_compiled = compute_average_gradient_many_inputs(model_name, compiled_observations,
    #                                                          compiled_energy_state,
    #                                                          compiled_salt,
    #                                                          compiled_actions,
    #                                                          compiled_in_light,
    #                                                          compiled_rnn_state,
    #                                                          context_name="Prey Capture",
    #                                                          full_reafference=True,
    #                                                          target_layer="Advantage",)

    # DOING VIA MEAN INPUTS
    #
    # mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
    #     get_mean_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, 1)
    #
    # dy_dobs, dy_deff, dy_deff2, dy_dlight, dy_denergy, dy_dsalt, dy_drnn = compute_gradient_for_input(model_name, mean_observation,
    #                                                                                         mean_energy_state,
    #                                                                                         mean_salt_input,
    #                                                                                         inputted_action,
    #                                                                                         inputted_in_light,
    #                                                                                         mean_rnn_state,
    #                                                                                         context_name="Prey Capture",
    #                                                                                         full_reafference=True,
    #                                                                                         target_layer="Advantage",
    #                                                                                         )
    #
    # mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
    #     get_mean_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, 4)
    #
    # dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnn = compute_gradient_for_input(model_name,
    #                                                                                         mean_observation,
    #                                                                                         mean_energy_state,
    #                                                                                         mean_salt_input,
    #                                                                                         inputted_action,
    #                                                                                         inputted_in_light,
    #                                                                                         mean_rnn_state,
    #                                                                                         context_name="Exploration",
    #                                                                                         full_reafference=True,
    #                                                                                         target_layer="Advantage",
    #                                                                                         )
    #
    # mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
    #     get_mean_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, 5)
    #
    # dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnn = compute_gradient_for_input(model_name,
    #                                                                                         mean_observation,
    #                                                                                         mean_energy_state,
    #                                                                                         mean_salt_input,
    #                                                                                         inputted_action,
    #                                                                                         inputted_in_light,
    #                                                                                         mean_rnn_state,
    #                                                                                         context_name="Wall Interaction",
    #                                                                                         full_reafference=True,
    #                                                                                         target_layer="Advantage",
    #                                                                                         )
    #
    # mean_observation, mean_rnn_state, mean_energy_state, mean_salt_input, inputted_action, inputted_in_light = \
    #     get_mean_inputs_for_context("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, 9)
    #
    # dy_dobs, dy_deff, dy_dlight, dy_denergy, dy_dsalt, dy_drnn = compute_gradient_for_input(model_name,
    #                                                                                         mean_observation,
    #                                                                                         mean_energy_state,
    #                                                                                         mean_salt_input,
    #                                                                                         inputted_action,
    #                                                                                         inputted_in_light,
    #                                                                                         mean_rnn_state,
    #                                                                                         context_name="Starving",
    #                                                                                         full_reafference=True,
    #                                                                                         target_layer="Advantage",
    #                                                                                         )


