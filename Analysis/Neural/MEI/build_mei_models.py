import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data
from Analysis.Neural.MEI.graphs_for_mei import MEICore, MEIReadout, Trainer, TrainerExtended

tf.logging.set_verbosity(tf.logging.ERROR)


def get_all_observations(model_name, assay_config, assay_id, n):
    observations = []
    for i in range(1, n + 1):
        data = load_data(model_name, assay_config, f"{assay_id}-{i}")
        if i == 1:
            observations = data["observation"]
        else:
            observations = np.concatenate((observations, data["observation"]), axis=0)

    return np.array(observations)


def get_all_cnn_activity(model_name, assay_config, assay_id, n):
    keys_to_collect = ["conv1l", "conv2l", "conv3l", "conv4l", "conv1r", "conv2r", "conv3r", "conv4r"]
    collected_data = {}
    for i in range(1, n + 1):
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


def build_model(observation_data, activity_data, n_repeats, train_prop=0.9, save_model=True, model_name="test_model"):
    # Test and train split
    train_index = int(observation_data.shape[0] * train_prop)
    train_observation_data = observation_data[:train_index]
    train_activity_data = activity_data[:train_index]
    test_observation_data = observation_data[train_index:]
    test_activity_data = activity_data[train_index:]

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
                                                     core.observation: test_observation_data[step:step + 1],
                                                 })
            pre_compiled_predicted_neural_activity.append(predicted_neural_activity[0, 0])

        # Training
        for i in range(n_repeats):
            for step in range(0, train_observation_data.shape[0], 50):
                _, loss = sess.run([trainer.train, trainer.total_loss],
                                   feed_dict={
                                       core.observation: train_observation_data[step:step + 50],
                                       trainer.actual_responses: train_activity_data[step:step + 50]
                                   })
                compiled_loss.append(loss)

        # Cross validation
        compiled_predicted_neural_activity = []
        for step in range(test_observation_data.shape[0]):
            predicted_neural_activity = sess.run(readout.predicted_neural_activity,
                                                 feed_dict={
                                                     core.observation: test_observation_data[step:step + 1],
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
    print(
        f"Performance difference: {100 * (np.mean(pre_prediction_error) - np.mean(prediction_error)) / np.mean(pre_prediction_error)}%")

    compiled_loss = np.array([c[0][0] for c in compiled_loss])


def build_model_multiple_neurons(observation_data, activity_data, train_prop=0.9, save_model=True,
                                 model_name="test_model", n_repeats=100, learning_rate=0.001, batch_size=50):
    # Test and train split
    n_units = activity_data.shape[-1]
    train_index = int(observation_data.shape[0] * train_prop)
    train_observation_data = observation_data[:train_index]
    train_activity_data = activity_data[:train_index]
    test_observation_data = observation_data[train_index:]
    test_activity_data = activity_data[train_index:]

    with tf.Session() as sess:
        # Creating graph
        core = MEICore(my_scope=model_name)
        # core_outputs = [tf.identity(core.output, name="output_"+str(unit)) for unit in range(n_units)]
        # readout_blocks = tf.split(core.output, axis=0, num_or_size_splits=n_units)
        readout_blocks = {}
        for unit in range(n_units):
            readout_blocks[f"Unit {unit}"] = MEIReadout(core.output, my_scope=model_name + "_readout_" + str(unit))
        combined_readout = tf.concat(
            [readout_blocks[block].predicted_neural_activity for block in readout_blocks.keys()],
            axis=1)
        trainer = TrainerExtended(combined_readout, n_units, learning_rate=learning_rate)

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
                                                     core.observation: test_observation_data[step:step + 1],
                                                 })
            pre_compiled_predicted_neural_activity.append(predicted_neural_activity[0])

        # Training
        for i in range(n_repeats):
            print(f"Repeat: {i}")
            for step in range(0, train_observation_data.shape[0], 50):
                _, loss, lossl2 = sess.run([trainer.train, trainer.total_loss, trainer.lossL2],
                                   feed_dict={
                                       core.observation: train_observation_data[step:step + batch_size],
                                       trainer.actual_responses: train_activity_data[step:step + batch_size]
                                   })
                compiled_loss.append(loss)

        # Cross validation
        compiled_predicted_neural_activity = []
        for step in range(test_observation_data.shape[0]):
            predicted_neural_activity = sess.run(combined_readout,
                                                 feed_dict={
                                                     core.observation: test_observation_data[step:step + 1],
                                                 })
            compiled_predicted_neural_activity.append(predicted_neural_activity[0])

        if save_model:
            saver.save(sess, f"MEI-Models/{model_name}/1")

    pre_compiled_predicted_neural_activity = np.array(pre_compiled_predicted_neural_activity)
    compiled_predicted_neural_activity = np.array(compiled_predicted_neural_activity)
    pre_prediction_error = (pre_compiled_predicted_neural_activity - test_activity_data) ** 2
    prediction_error = (compiled_predicted_neural_activity - test_activity_data) ** 2

    # print(f"Pre-Prediction Error: {np.mean(pre_prediction_error)}")
    print(f"Prediction Error: {np.mean(prediction_error)}")
    # print(f"Final-Score: {0.0001/np.mean(prediction_error)}")

    plt.plot(compiled_loss)
    plt.show()


def shuffle_data(observation_data, activity_data):
    indices = np.arange(observation_data.shape[0])
    np.random.shuffle(indices)
    observation_data = observation_data[indices]
    activity_data = activity_data[indices]

    return observation_data, activity_data


def build_unit_observation_pairs(cnn_activity_data, associated_observations):
    n_repeats = cnn_activity_data.shape[1]
    associated_observations_repeated = np.repeat(associated_observations, n_repeats, axis=(0))
    cnn_activity_data_flat = np.reshape(cnn_activity_data, (-1, cnn_activity_data.shape[-1]))
    return cnn_activity_data_flat, associated_observations_repeated


def fit_hyperparameters_to_models(model_name, assay_config, assay_id, n, layer):
    """Performs grid search on models trained"""

    cnn_activity = get_all_cnn_activity(model_name, assay_config, assay_id, n)
    observations = get_all_observations(model_name, assay_config, assay_id, n)
    cnn_activity = normalise_cnn_data(cnn_activity)
    observations = observations.astype(float) / 255

    selected_activity_data, relevant_observations = build_unit_observation_pairs(cnn_activity["conv3l"],
                                                                                 observations[:, :, :, 0])
    relevant_observations, selected_activity_data = shuffle_data(relevant_observations, selected_activity_data)

    # Hyperparameters to test
    batch_size_values = [1, 5, 10, 50, 100]
    learning_rate_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    batch_size_values = np.repeat(batch_size_values, 6, 0).astype(int)
    learning_rate_values = np.reshape(np.repeat([learning_rate_values], 5, 0), (-1))

    for lr, bs in zip(learning_rate_values, batch_size_values):
        print(f"LR: {lr}, BS: {bs}")
        model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                                 args=(
                                                 relevant_observations, selected_activity_data, 0.9, False, "layer_3",
                                                 1, lr, bs)
                                                 )
        model_building.start()
        model_building.join()

        model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                                 args=(
                                                 relevant_observations, selected_activity_data, 0.9, False, "layer_3",
                                                 1, lr, bs)
                                                 )
        model_building.start()
        model_building.join()

        model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                                 args=(
                                                 relevant_observations, selected_activity_data, 0.9, False, "layer_3",
                                                 1, lr, bs)
                                                 )
        model_building.start()
        model_building.join()


if __name__ == "__main__":
    model_name = "dqn_scaffold_18-1"
    cnn_activity = get_all_cnn_activity(model_name, "Behavioural-Data-CNN", "Naturalistic", 10)
    observations = get_all_observations(model_name, "Behavioural-Data-CNN", "Naturalistic", 10)
    cnn_activity = normalise_cnn_data(cnn_activity)
    observations = observations.astype(float) / 255

    # selected_activity_data_2, relevant_observations = build_unit_observation_pairs(cnn_activity["conv2l"],
    #                                                                                observations[:, :, :, 0])
    selected_activity_data_3, relevant_observations = build_unit_observation_pairs(cnn_activity["conv1l"],
                                                                                    observations[:, :, :, 0])
    # selected_activity_data_4, _relevant_observations = build_unit_observation_pairs(cnn_activity["conv4l"],
    #                                                                                 observations[:, :, :, 0])

    # selected_activity_data_3 = np.repeat(selected_activity_data_3, 2, 0)
    # selected_activity_data_4 = np.repeat(selected_activity_data_4, 5, 0)
    #
    # final_shape = selected_activity_data_2.shape[0]
    #
    # selected_activity_data_3 = selected_activity_data_3[:final_shape]
    # selected_activity_data_4 = selected_activity_data_4[:final_shape]

    # selected_activity_data = np.concatenate(
    #     (selected_activity_data_4, selected_activity_data_3, selected_activity_data_2), axis=1)
    selected_activity_data = selected_activity_data_3
    relevant_observations, selected_activity_data = shuffle_data(relevant_observations, selected_activity_data)

    # fit_hyperparameters_to_models("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 10, "conv3l")

    # relevant_observations = observations[:, :, :, 0]
    # selected_activity_data = cnn_activity["conv3l"][:, :, 0]
    # selected_activity_data2 = np.reshape(selected_activity_data, (-1))
    # relevant_observations2 = np.repeat(relevant_observations, 5, 0)
    # print("\n")
    # print("Only one neuron")
    # model_building = multiprocessing.Process(target=build_model, args=(relevant_observations2, selected_activity_data2, 10))
    # model_building.start()
    # model_building.join()

    # repeats_to_test = [1, 5, 10, 100]
    # selected_activity_data, relevant_observations = build_unit_observation_pairs(cnn_activity["conv3l"], observations[:, :, :, 0])
    # relevant_observations, selected_activity_data = shuffle_data(relevant_observations, selected_activity_data)
    # for repeats in repeats_to_test:
    #     print("\n")
    #     print(str(repeats) + " Repeat")
    #     model_building = multiprocessing.Process(target=build_model_multiple_neurons,
    #                                              args=(relevant_observations, selected_activity_data, 0.9, True, "layer_3",
    #                                                    repeats)
    #                                              )
    #     model_building.start()
    #     model_building.join()

    # Args order: observastions, cnn activity, train proportion, save model, model name, num repeats, learning rate, batch size.
    model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                             args=(
                                             relevant_observations, selected_activity_data, 0.9, True, model_name, "layer_1_1",
                                             10, 0.0005, 100)
                                             )
    model_building.start()
    model_building.join()

    model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                             args=(
                                             relevant_observations, selected_activity_data, 0.9, True, model_name, "layer_1_2",
                                             10, 0.0005, 100)
                                             )
    model_building.start()
    model_building.join()

    model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                             args=(
                                             relevant_observations, selected_activity_data, 0.9, True, model_name, "layer_1_3",
                                             10, 0.0005, 100)
                                             )
    model_building.start()
    model_building.join()

    model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                             args=(
                                             relevant_observations, selected_activity_data, 0.9, True, model_name, "layer_1_4",
                                             10, 0.0005, 100)
                                             )
    model_building.start()
    model_building.join()
