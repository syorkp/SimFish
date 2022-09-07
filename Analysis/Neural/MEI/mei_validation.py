import numpy as np

import tensorflow.compat.v1 as tf

from Analysis.Model.build_network import build_network_dqn
from Analysis.load_model_config import load_configuration_files
from Analysis.Stimuli.observation_stimuli import get_prey_stimuli_across_visual_field
from Analysis.Model.build_network import build_network_dqn
from Analysis.load_data import load_data

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment


def load_optimal_activation(model_name, layer):
    with open(f"./MEI-Models/{model_name}/{layer}-optimal_activation.npy", "rb") as f:
        optimal_image = np.load(f)
    return optimal_image


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"
    data = np.concatenate(([load_data(model_name, "Behavioural-Data-CNN", f"Naturalistic-{i}")["conv1l"] for i in range(1, 11)]))
    image = load_optimal_activation(model_name, "layer_1")
    model_location = f"../../../Training-Output/{model_name}"

    params, environment_params, _, _, _ = load_configuration_files(model_name)
    simulation = DiscreteNaturalisticEnvironment(environment_params, True, True, False)
    sess = tf.Session()
    conv = []
    rnn_in_compiled = []
    with sess as sess:
        network = build_network_dqn(environment_params, params, simulation)
        saver = tf.train.Saver(max_to_keep=5)
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        conv1l_compiled = []

        for i in range(16):
            chosen_image = np.expand_dims(image[i], 2)
            chosen_image = np.concatenate((chosen_image, chosen_image), 2) * 255
            conv1l = sess.run(network.conv1l,
                feed_dict={network.observation: chosen_image.astype(float)}
            )
            conv1l_compiled.append(conv1l)
    conv1l_compiled = np.array(conv1l_compiled)[:, 0]

    conv1l_own_responses = np.array([np.max(c[:, i]) for i, c in enumerate(conv1l_compiled)])
    data = np.max(np.max(data, axis=(1)), axis=(0))
