import multiprocessing
import numpy as np
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Analysis.load_data import load_data
from Analysis.Neural.MEI.graphs_for_mei import MEICore, MEIReadout, Trainer, TrainerExtended
tf.logging.set_verbosity(tf.logging.ERROR)

multiprocessing.set_start_method('spawn', force=True)


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


def build_model_multiple_neurons(observation_data, activity_data, train_prop=0.9, save_model=True, trial_name="dqn_ex",
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
                _, loss, lossl2 = sess.run([trainer.train, trainer.total_loss, trainer.total_loss],
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
            saver.save(sess, f"MEI-Models/{trial_name}/{model_name}/1")

    pre_compiled_predicted_neural_activity = np.array(pre_compiled_predicted_neural_activity)
    compiled_predicted_neural_activity = np.array(compiled_predicted_neural_activity)
    pre_prediction_error = (pre_compiled_predicted_neural_activity - test_activity_data) ** 2
    prediction_error = (compiled_predicted_neural_activity - test_activity_data) ** 2

    # print(f"Pre-Prediction Error: {np.mean(pre_prediction_error)}")
    print(f"Prediction Error: {np.mean(prediction_error)}")
    # print(f"Final-Score: {0.0001/np.mean(prediction_error)}")

    with open(f"MEI-Models/{trial_name}/{model_name}/prediction_error.npy", "wb") as f:
        np.save(f, np.array(prediction_error))

    with open(f"MEI-Models/{trial_name}/{model_name}/compiled_loss.npy", "wb") as f:
        np.save(f, np.array(compiled_loss))


    plt.plot(prediction_error)
    plt.savefig(f"MEI-Models/{trial_name}/{model_name}/prediction_error.png")
    plt.show()

    plt.plot(compiled_loss)
    plt.savefig(f"MEI-Models/{trial_name}/{model_name}/compiled_loss.png")
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
                                                     relevant_observations, selected_activity_data, 0.9, False,
                                                     "layer_3",
                                                     1, lr, bs)
                                                 )
        model_building.start()
        model_building.join()

        model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                                 args=(
                                                     relevant_observations, selected_activity_data, 0.9, False,
                                                     "layer_3",
                                                     1, lr, bs)
                                                 )
        model_building.start()
        model_building.join()

        model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                                 args=(
                                                     relevant_observations, selected_activity_data, 0.9, False,
                                                     "layer_3",
                                                     1, lr, bs)
                                                 )
        model_building.start()
        model_building.join()


def build_all_possible_cnn_models(model_name, assay_config, assay_id, n, n_models=4):
    cnn_activity = get_all_cnn_activity(model_name, assay_config, assay_id, n)
    observations = get_all_observations(model_name, assay_config, assay_id, n)

    cnn_activity = normalise_cnn_data(cnn_activity)

    layers = ["conv1l", "conv2l", "conv3l", "conv4l"]
    for layer in layers:
        selected_activity_data, relevant_observations = build_unit_observation_pairs(cnn_activity[layer],
                                                                                       observations[:, :, :, 0])
        relevant_observations, selected_activity_data = shuffle_data(relevant_observations, selected_activity_data)

        for i in range(1, n_models + 1):
            # Args order: observations, cnn activity, train proportion, save model, model name, num repeats,
            # learning rate, batch size.
            model_building = multiprocessing.Process(target=build_model_multiple_neurons,
                                                     args=(
                                                         relevant_observations, selected_activity_data, 0.9, True,
                                                         model_name,
                                                         f"{layer}_{i}",
                                                         10, 0.0005, 100)
                                                     )
            model_building.start()
            model_building.join()


if __name__ == "__main__":
    cnn_activity = build_all_possible_cnn_models("dqn_scaffold_18-1", "Behavioural-Data-CNN", "Naturalistic", 40)

