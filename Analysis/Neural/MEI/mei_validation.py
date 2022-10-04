import matplotlib.pyplot as plt
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
    with open(f"./Generated-MEIs/Direct/{model_name}/{layer}-optimal_activation.npy", "rb") as f:
        optimal_image = np.load(f)
    return optimal_image


def get_maximal_activation(model_name, assay_config, assay_id, n, target_layer):
    target_layer_activity = np.concatenate(([load_data(model_name, assay_config, f"{assay_id}-{i}")[target_layer]
                                             for i in range(1, n+1)]))
    all_observations = np.concatenate(([load_data(model_name, assay_config, f"{assay_id}-{i}")["observation"]
                                             for i in range(1, n+1)]))
    maximal_activity_dim1 = np.max(target_layer_activity, axis=(1))  # Over repeated dim - e.g. 22
    maximal_activity_indices = np.argmax(maximal_activity_dim1, axis=0)
    maximal_activity = np.max(maximal_activity_dim1, axis=(0))
    observations_responsible = all_observations[maximal_activity_indices]
    if "l" in target_layer:
        observations_responsible = observations_responsible[:, :, :, 0]
    else:
        observations_responsible = observations_responsible[:, :, :, 1]

    return maximal_activity, observations_responsible


def get_activity_cnn(model_name, input_observation, target_layer):
    model_location = f"../../../Training-Output/{model_name}"
    params, environment_params, _, _, _ = load_configuration_files(model_name)
    simulation = DiscreteNaturalisticEnvironment(environment_params, True, True, False)
    sess = tf.Session()
    conv_compiled = []
    with sess as sess:
        network = build_network_dqn(environment_params, params, simulation)
        saver = tf.train.Saver(max_to_keep=5)
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        if target_layer == "conv1l":
            recording = network.conv1l
            n_units = 16
        elif target_layer == "conv2l":
            recording = network.conv2l
            n_units = 8
        elif target_layer == "conv3l":
            recording = network.conv3l
            n_units = 8
        elif target_layer == "conv4l":
            recording = network.conv4l
            n_units = 64
        elif target_layer == "conv1r":
            recording = network.conv1r
            n_units = 16
        elif target_layer == "conv2r":
            recording = network.conv2r
            n_units = 8
        elif target_layer == "conv3r":
            recording = network.conv3r
            n_units = 8
        elif target_layer == "conv4r":
            recording = network.conv4r
            n_units = 64
        else:
            print("Error, target layer incorrectly specified.")
            recording = None
            n_units = 0

        for i in range(n_units):
            chosen_image = np.expand_dims(input_observation[i], 2)
            chosen_image = np.concatenate((chosen_image, chosen_image), 2)
            conv = sess.run(recording,
                feed_dict={network.observation: chosen_image.astype(float)}
            )
            conv_compiled.append(conv)
    conv_compiled = np.array(conv_compiled)[:, 0]
    conv_own_responses = np.array([np.max(c[:, i]) for i, c in enumerate(conv_compiled)])
    return conv_compiled, conv_own_responses


def get_best_images(artificial_activity, artificial_images, real_activity, real_images):
    artificial_best = artificial_activity > real_activity

    best_images = real_images
    best_images[artificial_best] = artificial_images[artificial_best]
    print(100 * np.sum(artificial_best)/len(artificial_best))
    # Print the number of best images
    plt.imshow(best_images)
    plt.show()


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"
    layer = "conv4l"
    maximal_activity, observations_responsible = get_maximal_activation(model_name, "Behavioural-Data-CNN", f"Naturalistic", 40, layer)
    image = load_optimal_activation(model_name, layer)
    # Getting the actual activity of the artificial images
    c, cown = get_activity_cnn(model_name, image, layer)

    observations_responsible = np.concatenate((observations_responsible[:, :, 0:1], observations_responsible[:, :, 2:3],
                                               observations_responsible[:, :, 1:2]), axis=2)

    plt.imshow(observations_responsible)
    plt.savefig(f"{model_name}-{layer}-Actual-MEI")
    get_best_images(cown, image, maximal_activity, observations_responsible)
