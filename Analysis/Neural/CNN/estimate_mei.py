import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data
from Analysis.Neural.CNN.graphs_for_mei import MEICore, MEIReadout, Trainer, TrainerExtended

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


def build_model(observation_data, activity_data, train_prop=0.9, save_model=True, model_name="test_model"):

    # Test and train split
    train_index = int(observation_data.shape[0] * train_prop)
    train_observation_data = observation_data[:train_index]
    train_activity_data = activity_data[:train_index]
    test_observation_data = observation_data[train_index:]
    test_activity_data = activity_data[train_index:]
    n_repeats = 300


    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        readout = MEIReadout(core.output)
        trainer = Trainer(readout.predicted_neural_activity)

        if save_model:
            saver = tf.train.Saver(max_to_keep=5)

        # Init variables
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()

        sess.run(init)

        compiled_loss = []

        # Pre-training evaluation
        pre_compiled_predicted_neural_activity = []
        for step in range(test_observation_data.shape[0]):
            predicted_neural_activity = sess.run(readout.predicted_neural_activity,
                            feed_dict={
                                core.observation: test_observation_data[step:step+1],
                            })
            pre_compiled_predicted_neural_activity.append(predicted_neural_activity[0, 0])

        # Training
        for i in range(n_repeats):
            for step in range(0, train_observation_data.shape[0], 50):
                _, loss = sess.run([trainer.train, trainer.total_loss],
                                feed_dict={
                                    core.observation: train_observation_data[step:step+50],
                                    trainer.actual_responses: train_activity_data[step:step+50]
                                })
                compiled_loss.append(loss)

        # Cross validation
        compiled_predicted_neural_activity = []
        for step in range(test_observation_data.shape[0]):
            predicted_neural_activity = sess.run(readout.predicted_neural_activity,
                            feed_dict={
                                core.observation: test_observation_data[step:step+1],
                            })
            compiled_predicted_neural_activity.append(predicted_neural_activity[0, 0])

        if save_model:
            saver.save(sess, f"MEI-Models/{model_name}/1")

    pre_compiled_predicted_neural_activity = np.array(pre_compiled_predicted_neural_activity)
    compiled_predicted_neural_activity = np.array(compiled_predicted_neural_activity)
    pre_prediction_error = (pre_compiled_predicted_neural_activity - test_activity_data) ** 2
    prediction_error = (compiled_predicted_neural_activity - test_activity_data) ** 2

    print(f"Pre-Prediction Error: {np.mean(pre_prediction_error)}")
    print(f"Prediction Error: {np.mean(prediction_error)}")

    compiled_loss = np.array([c[0][0] for c in compiled_loss])
    return compiled_loss


def build_model_multiple_neurons(observation_data, activity_data, train_prop=0.9, save_model=True,
                                 model_name="test_model", n_repeats=100):

    # Test and train split
    n_units = activity_data.shape[-1]
    train_index = int(observation_data.shape[0] * train_prop)
    train_observation_data = observation_data[:train_index]
    train_activity_data = activity_data[:train_index]
    test_observation_data = observation_data[train_index:]
    test_activity_data = activity_data[train_index:]

    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        # core_outputs = [tf.identity(core.output, name="output_"+str(unit)) for unit in range(n_units)]
        # readout_blocks = tf.split(core.output, axis=0, num_or_size_splits=n_units)
        readout_blocks = {}
        for unit in range(n_units):
            readout_blocks[f"Unit {unit}"] = MEIReadout(core.output, my_scope="MEIReadout_" + str(unit))
        combined_readout = tf.concat([readout_blocks[block].predicted_neural_activity for block in readout_blocks.keys()],
                                      axis=1)
        trainer = TrainerExtended(combined_readout, n_units)

        if save_model:
            saver = tf.train.Saver(max_to_keep=5)

        # Init variables
        init = tf.global_variables_initializer()
        trainables = tf.trainable_variables()

        sess.run(init)

        compiled_loss = []

        # Pre-training evaluation
        pre_compiled_predicted_neural_activity = []
        for step in range(test_observation_data.shape[0]):
            predicted_neural_activity = sess.run(combined_readout,
                            feed_dict={
                                core.observation: test_observation_data[step:step+1],
                            })
            pre_compiled_predicted_neural_activity.append(predicted_neural_activity[0])

        # Training
        for i in range(n_repeats):
            for step in range(0, train_observation_data.shape[0], 50):
                _, loss = sess.run([trainer.train, trainer.total_loss],
                                feed_dict={
                                    core.observation: train_observation_data[step:step+50],
                                    trainer.actual_responses: train_activity_data[step:step+50]
                                })
                compiled_loss.append(loss)

        # Cross validation
        compiled_predicted_neural_activity = []
        for step in range(test_observation_data.shape[0]):
            predicted_neural_activity = sess.run(combined_readout,
                            feed_dict={
                                core.observation: test_observation_data[step:step+1],
                            })
            compiled_predicted_neural_activity.append(predicted_neural_activity[0])

        if save_model:
            saver.save(sess, f"MEI-Models/{model_name}/1")

    pre_compiled_predicted_neural_activity = np.array(pre_compiled_predicted_neural_activity)
    compiled_predicted_neural_activity = np.array(compiled_predicted_neural_activity)
    pre_prediction_error = (pre_compiled_predicted_neural_activity - test_activity_data) ** 2
    prediction_error = (compiled_predicted_neural_activity - test_activity_data) ** 2

    print(f"Pre-Prediction Error: {np.mean(pre_prediction_error)}")
    print(f"Prediction Error: {np.mean(prediction_error)}")

    plt.plot(compiled_loss)
    plt.show()


def shuffle_data(observation_data, activity_data):
    indices = np.arange(observation_data.shape[0])
    np.random.shuffle(indices)
    observation_data = observation_data[indices]
    activity_data = activity_data[indices]

    return observation_data, activity_data


def produce_mei(model_name):

    # Initial, random image.
    image = np.random.normal(128, 8, size=(100, 1))
    image = np.clip(image, 0, 255)
    image /= 255
    # image = np.random.uniform(size=(100, 3))
    iterations = 10000

    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        readout = MEIReadout(core.output)
        # trainer = Trainer(readout.predicted_neural_activity)

        saver = tf.train.Saver(max_to_keep=5)
        model_location = "MEI-Models/" + model_name
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # Constants
        step_size = 1.5
        step_gain = 1
        eps = 1e-12

        grad = tf.gradients(readout.predicted_neural_activity, core.observation, name='grad')
        red_grad = tf.reduce_sum(grad)
        gradients = []
        # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
        pred = []
        reds = []
        for i in range(iterations):
            input_image = np.concatenate((np.zeros((100, 1)), image, np.zeros((100, 1))), axis=1)
            dy_dx, p, red = sess.run([grad, readout.predicted_neural_activity, red_grad],
                                     feed_dict={core.observation: np.expand_dims(input_image, 0)})
            gradients.append(dy_dx)
            update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (step_gain / 255)
            input_image += update * dy_dx[0][0]
            image = input_image[:, 1:2]
            image = np.clip(image, 0, 1)
            reds.append(red)
            if np.max(np.absolute(dy_dx[0])) == 0:
                print(i)
                # Shouldnt ever really occur.
                image = np.random.normal(128, 8, size=(100, 1))
                image = np.clip(image, 0, 255)
                image /= 255
            pred.append(p)

        gradients = np.array(gradients)
        plt.imshow(np.expand_dims(np.concatenate((np.zeros((100, 2)), image), axis=1), axis=0))
        plt.show()


def produce_meis(model_name, n_units=8):
    """Does the same thing for the multiple neurons of a given model"""

    # Initial, random image.
    all_images = np.zeros((n_units, 100, 1))

    # image = np.random.uniform(size=(100, 3))
    iterations = 10000

    with tf.Session() as sess:
        # Creating graph
        core = MEICore()
        readout_blocks = {}
        for unit in range(n_units):
            readout_blocks[f"Unit {unit}"] = MEIReadout(core.output, my_scope="MEIReadout_" + str(unit))

        saver = tf.train.Saver(max_to_keep=5)
        model_location = "MEI-Models/" + model_name
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # Constants
        step_size = 1.5
        step_gain = 1
        eps = 1e-12

        for unit in range(n_units):
            image = np.random.normal(128, 8, size=(100, 1))
            image = np.clip(image, 0, 255)
            image /= 255

            grad = tf.gradients(readout_blocks[f"Unit {unit}"].predicted_neural_activity, core.observation, name='grad')
            red_grad = tf.reduce_sum(grad)
            gradients = []
            # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
            pred = []
            reds = []
            for i in range(iterations):
                input_image = np.concatenate((np.zeros((100, 1)), image, np.zeros((100, 1))), axis=1)
                dy_dx, p, red = sess.run([grad, readout_blocks[f"Unit {unit}"].predicted_neural_activity, red_grad],
                                         feed_dict={core.observation: np.expand_dims(input_image, 0)})
                gradients.append(dy_dx)
                update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (step_gain / 255)
                input_image += update * dy_dx[0][0]
                image = input_image[:, 1:2]
                image = np.clip(image, 0, 1)
                reds.append(red)
                if np.max(np.absolute(dy_dx[0])) == 0:
                    print(i)
                    # Shouldnt ever really occur.
                    image = np.random.normal(128, 8, size=(100, 1))
                    image = np.clip(image, 0, 255)
                    image /= 255
                pred.append(p)
            all_images[unit] = image

    plt.imshow(np.concatenate((np.zeros((n_units, 100, 2)), all_images[unit]), axis=1))
    plt.show()


def build_unit_observation_pairs(cnn_activity_data, associated_observations):
    n_repeats = cnn_activity_data.shape[1]
    associated_observations_repeated = np.repeat(associated_observations, n_repeats, axis=(0))
    cnn_activity_data_flat = np.reshape(cnn_activity_data, (-1, cnn_activity_data.shape[-1]))
    return cnn_activity_data_flat, associated_observations_repeated


if __name__ == "__main__":
    cnn_activity = get_all_cnn_activity("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10)
    observations = get_all_observations("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10)
    cnn_activity = normalise_cnn_data(cnn_activity)
    observations = observations.astype(float) / 255

    # relevant_observations = observations[:, :, :, 0]
    # selected_activity_data = cnn_activity["conv3l"][:, 2, 0]
    selected_activity_data, relevant_observations = build_unit_observation_pairs(cnn_activity["conv4l"], observations[:, :, :, 0])
    relevant_observations, selected_activity_data = shuffle_data(relevant_observations, selected_activity_data)
    build_model_multiple_neurons(relevant_observations, selected_activity_data, model_name="layer_4",
                                 n_repeats=200)
    # loss_data = build_model(relevant_observations, selected_activity_data)

    # produce_meis("extended_test", n_units=64)
