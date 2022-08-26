import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data
from Analysis.Neural.CNN.graphs_for_mei import MEICore, MEIReadout, Trainer

tf.logging.set_verbosity(tf.logging.ERROR)


def get_all_observations(model_name, assay_config, assay_id, n):
    observations = []
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if i == 1:
            observations = data["observation"]
        else:
            observations = np.concatenate((observations, data["observation"]), axis=0)

    return np.array(observations)


def get_all_cnn_activity(model_name, assay_config, assay_id, n):
    keys_to_collect = ["conv1l", "conv2l", "conv3l", "conv4l", "conv1r", "conv2r", "conv3r", "conv4r"]
    collected_data = {}
    for i in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if i == 1:
            collected_data = {key: data[key] for key in keys_to_collect}
        else:
            for key in keys_to_collect:
                collected_data[key] = np.concatenate((collected_data[key], data[key]), axis=0)
    return collected_data


def normalise_cnn_data(cnn_data):
    for layer in cnn_data.keys():
        max_activity = np.max(cnn_data[layer], axis=(0, 1))
        cnn_data[layer] /= max_activity
    return cnn_data


def build_model(observation_data, activity_data):
    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        readout = MEIReadout(core.output)
        trainer = Trainer(readout.predicted_neural_activity)

        # Init variables
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()

        sess.run(init)

        compiled_loss = []

        for step in range(observation_data.shape[0]):
            _, loss = sess.run([trainer.train, trainer.total_loss],
                            feed_dict={
                                core.observation: observation_data[step:step+1],
                                trainer.actual_responses: activity_data[step:step+1]
                            })
            compiled_loss.append(loss)
    return compiled_loss


if __name__ == "__main__":
    cnn_activity = get_all_cnn_activity("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 1)
    observations = get_all_observations("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 1)
    cnn_activity = normalise_cnn_data(cnn_activity)
    observations = observations.astype(float) / 255

    relevant_observations = observations[:, :, :, 0]
    selected_activity_data = cnn_activity["conv1l"][:, 0, 0]
    # TODO: should actually find way of flattening across the data dimension (22), while repeating the relevant obsevations
    loss_data = build_model(relevant_observations, selected_activity_data)

