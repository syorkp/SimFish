import numpy as np
import matplotlib.pyplot as plt
import copy

import tensorflow.compat.v1 as tf


from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files
from Analysis.Stimuli.observation_stimuli import get_prey_stimuli_across_visual_field

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment

from Networks.DQN.q_network import QNetwork
from Networks.DQN.q_network_dynamic import QNetworkDynamic



def get_internal_state_order(environment_params):
    internal_state_order = []
    if environment_params['in_light']:
        internal_state_order.append("in_light")
    if environment_params['hunger']:
        internal_state_order.append("hunger")
    if environment_params['stress']:
        internal_state_order.append("stress")
    if environment_params['energy_state']:
        internal_state_order.append("energy_state")
    if environment_params['salt']:
        internal_state_order.append("salt")
    return internal_state_order


def build_network(environment_params, learning_params, simulation, full_reafference=True):
    internal_states = sum(
        [1 for x in [environment_params['hunger'], environment_params['stress'],
                     environment_params['energy_state'], environment_params['in_light'],
                     environment_params['salt']] if x is True])
    internal_states = max(internal_states, 1)
    internal_state_names = get_internal_state_order(environment_params)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=learning_params['rnn_dim_shared'], state_is_tuple=True)
    if environment_params["use_dynamic_network"]:
        if "reuse_eyes" in learning_params:
            reuse_eyes = learning_params['reuse_eyes']
        else:
            reuse_eyes = False
        main_QN = QNetworkDynamic(simulation=simulation,
                                  my_scope='main',
                                  internal_states=internal_states,
                                  internal_state_names=internal_state_names,
                                  num_actions=learning_params['num_actions'],
                                  base_network_layers=learning_params[
                                      'base_network_layers'],
                                  modular_network_layers=learning_params[
                                      'modular_network_layers'],
                                  ops=learning_params['ops'],
                                  connectivity=learning_params[
                                      'connectivity'],
                                  reflected=learning_params['reflected'],
                                  reuse_eyes=reuse_eyes,
                                  )
    else:
        main_QN = QNetwork(simulation=simulation,
                           rnn_dim=learning_params['rnn_dim_shared'],
                           rnn_cell=cell,
                           my_scope='main',
                           num_actions=learning_params['num_actions'],
                           internal_states=internal_states,
                           learning_rate=learning_params['learning_rate'],
                           extra_layer=learning_params['extra_rnn'],
                           full_reafference=full_reafference)
    return main_QN


def get_feature_maps(model_name, observations):
    model_location = f"../../../Training-Output/{model_name}"
    params, environment_params, _, _, _ = load_configuration_files(model_name)
    simulation = DiscreteNaturalisticEnvironment(environment_params, True, True, False)
    sess = tf.Session()
    conv = []
    rnn_in_compiled = []
    with sess as sess:
        network = build_network(environment_params, params, simulation)
        saver = tf.train.Saver(max_to_keep=5)
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        for observation in observations:
            conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r, rnn_in = sess.run(
                [
                    network.conv1l, network.conv2l, network.conv3l, network.conv4l,
                    network.conv1r, network.conv2r, network.conv3r, network.conv4r,
                    network.rnn_in
                ],
                feed_dict={network.observation: observation,
                           network.internal_state: [[0, 0, 0]],
                           network.prev_actions: [[0, 0, 0]]}
            )
            conv.append([conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r])
            rnn_in_compiled.append(rnn_in)

    return conv, rnn_in_compiled


def plot_feature_map_with_observation(feature_map, observation):
    input_image_p = copy.copy(observation.astype(int))
    input_image_p = np.expand_dims(input_image_p, 0)

    n_units = feature_map.shape[-1]
    computed_filter = np.swapaxes(feature_map, 0, 1)

    fig, axs = plt.subplots(n_units + 1, 1)
    fig.set_size_inches(16, 16)
    axs[0].imshow(input_image_p, aspect="auto")
    for unit in range(n_units):
        axs[unit + 1].imshow(computed_filter[unit:unit + 1, :], aspect="auto")
        axs[unit + 1].tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
    axs[0].autoscale()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = "dqn_scaffold_18-1"
    observations = get_prey_stimuli_across_visual_field(30, 10, "dqn_scaffold_18-1")

    feature_maps, rnn_in_compiled = get_feature_maps(model, observations)
    rnn_in_compiled = np.array(rnn_in_compiled)
    rnn_in_compiled = rnn_in_compiled[:, 0, :20]

    # for i in range(rnn_in_compiled.shape[-1]):
    #     plt.plot(rnn_in_compiled[:, i])
    #     plt.show()

    for i in range(observations.shape[0]):

        plot_feature_map_with_observation(feature_maps[i][1][0], observations[i, :, :, 0])
        # for j in range(4):
        #     plot_feature_map_with_observation(feature_maps[j][0], observations[i, :, :, 0])
