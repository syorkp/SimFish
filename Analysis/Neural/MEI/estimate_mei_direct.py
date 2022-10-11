import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf


from Analysis.load_model_config import load_configuration_files

from Networks.DQN.q_network import QNetwork
from Networks.DQN.q_network_dynamic import QNetworkDynamic
from Analysis.Model.build_network import get_internal_state_order

from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment


def create_network(simulation, environment_params, learning_params, full_reafference):
    """
    Create the main and target Q networks, according to the configuration parameters.
    :return: The main network and the target network graphs.
    """
    print("Creating networks...")
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
                           full_reafference=full_reafference
                           )
    return main_QN, internal_states


def produce_meis(model_name, layer_name, full_reafference, iterations=1000, conv=True):
    """Does the same thing for the multiple neurons of a given model"""
    if not os.path.exists(f"./Generated-MEIs/Direct/{model_name}/"):
        os.makedirs(f"./Generated-MEIs/Direct/{model_name}/")

    with tf.Session() as sess:
        # Build simulation
        params, environment_params, _, _, _ = load_configuration_files(model_name)
        simulation = DiscreteNaturalisticEnvironment(environment_params, True, True, False)

        # Creating graph
        network, n_internal_states = create_network(simulation, environment_params, params, full_reafference)

        saver = tf.train.Saver(max_to_keep=5)
        try:
            model_location = f"../../../Training-Output/{model_name}"
            checkpoint = tf.train.get_checkpoint_state(model_location)
            saver.restore(sess, checkpoint.model_checkpoint_path)
        except AttributeError:
            try:
                model_location = f"../../Training-Output/{model_name}"
                checkpoint = tf.train.get_checkpoint_state(model_location)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            except AttributeError:
                try:
                    model_location = f"../Training-Output/{model_name}"
                    checkpoint = tf.train.get_checkpoint_state(model_location)
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                except AttributeError:
                    model_location = f"Training-Output/{model_name}"
                    checkpoint = tf.train.get_checkpoint_state(model_location)
                    saver.restore(sess, checkpoint.model_checkpoint_path)


        # Defining Outputs to be measured
        # readout_blocks = {f"Unit {unit}": getattr(network, layer_name)[:, :, unit] for unit in range(n_units)}
        target_layer = getattr(network, layer_name)
        n_units = target_layer.shape[-1]
        all_images = np.zeros((n_units, 100, 3, 2))

        # Constants
        step_size = 0.15*10000 #1.5
        eps = 1e-12
        compiled_activity_log = []
        internal_state = [[0 for i in range(n_internal_states)]]

        for unit in range(n_units):
            print()
            print(f"Beginning Unit {unit+1}")
            input_image = np.random.normal(10, 8, size=(100, 3, 2))
            input_image = np.clip(input_image, 0, 255)

            # grad = tf.gradients(readout_blocks[f"Unit {unit}"], network.observation, name='grad')
            if conv:
                grad = tf.gradients(target_layer[:, :, unit], network.observation, name='grad')
            else:
                grad = tf.gradients(target_layer[:, unit], network.observation, name='grad')

            red_grad = tf.reduce_sum(grad)
            gradients = []
            # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
            reds = []
            activity_log = []
            for i in range(iterations):
                # input_image = np.concatenate((image, np.zeros((100, 1))), axis=1)
                if conv:
                    dy_dx, activity, red = sess.run([grad, tf.math.reduce_sum(target_layer[:, :, unit]), red_grad],
                                                feed_dict={network.observation: input_image.astype(int)})
                else:
                    dy_dx, activity, red = sess.run([grad, tf.math.reduce_sum(target_layer[:, unit]), red_grad],
                                                feed_dict={network.observation: input_image.astype(int),
                                                           network.prev_actions: [[0, 0, 0]],
                                                           network.internal_state: internal_state,

                                                           })
                gradients.append(dy_dx)

                update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (1 / 255)
                update = update * dy_dx[0]
                # image = input_image[:, 0:2]
                input_image = input_image.astype(float)
                input_image += update

                input_image = np.clip(input_image, 0, 255)
                reds.append(red)

                activity_log.append(activity)
                activity_changes = np.array(activity_log)[1:] - np.array(activity_log)[:-1]
                if i > 0:
                    print(f"{i}-Activity change: {activity_log[-1] - activity_log[-2]}")

                if i > 20:
                    if np.sum(activity_changes[-10:]) <= 0:
                        print(f"{i}-Reducing step size")
                        step_size = step_size * 0.9
                    # if np.sum(activity_changes[-100:]) <= 0:
                    #     print("Stopping early.")
                    #     activity_log += [activity for x in range(iterations-i)]
                    #     continue

                if np.max(np.absolute(dy_dx[0])) == 0:
                    print(f"Resetting: {unit + 1}")
                    input_image = np.random.normal(10, 8, size=(100, 3, 2))
                    input_image = np.clip(input_image, 0, 255)

            all_images[unit] = input_image
            compiled_activity_log.append(activity_log)

    compiled_activity_log = np.array(compiled_activity_log)
    compiled_activity_log = np.swapaxes(compiled_activity_log, 0, 1)
    plt.plot(compiled_activity_log)
    plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-activity")
    plt.clf()
    plt.close()

    all_images = all_images.astype(int)
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    fig, axs = plt.subplots(4, 1)
    axs[0].imshow(all_images[:, :, :, 0])
    axs[1].imshow(np.concatenate((np.zeros((n_units, 100, 2)), all_images[:, :, 1:2, 0]), axis=2).astype(int))
    axs[2].imshow(np.concatenate((all_images[:, :, 0:1, 0], np.zeros((n_units, 100, 2))), axis=2).astype(int))
    axs[3].imshow(np.concatenate((np.zeros((n_units, 100, 1)), all_images[:, :, 2:3, 0], np.zeros((n_units, 100, 1))), axis=2).astype(int))
    plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-mei")
    plt.clf()
    plt.close()

    plt.plot([np.sum(g) for g in gradients])
    plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-gradients")
    plt.clf()
    plt.close()

    fig, axs = plt.subplots(2, 1)
    axs[0].imshow(all_images[:, :, :, 0])
    axs[1].imshow(all_images[:, :, :, 1])
    plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-image-both-eyes")
    plt.clf()
    plt.close()

    # Save Optimal activation
    with open(f"Generated-MEIs/Direct/{model_name}/{layer_name}-optimal_activation.npy", "wb") as f:
        np.save(f, all_images)


def produce_meis_extended(model_name, layer_name, full_reafference, iterations=1000):
    """Does the same thing for the multiple neurons of a given model.

    For conv layers only. Applies separately for the spatial components of each conv layer.
    """

    if not os.path.exists(f"./Generated-MEIs/Direct/{model_name}/"):
        os.makedirs(f"./Generated-MEIs/Direct/{model_name}/")

    with tf.Session() as sess:
        # Build simulation
        params, environment_params, _, _, _ = load_configuration_files(model_name)
        simulation = DiscreteNaturalisticEnvironment(environment_params, True, True, False)

        # Creating graph
        network, n_is = create_network(simulation, environment_params, params, full_reafference)

        saver = tf.train.Saver(max_to_keep=5)
        model_location = f"../../../Training-Output/{model_name}"
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)

        # Defining Outputs to be measured
        # readout_blocks = {f"Unit {unit}": getattr(network, layer_name)[:, :, unit] for unit in range(n_units)}
        target_layer = getattr(network, layer_name)
        n_units = target_layer.shape[-1]
        n_applications = target_layer.shape[-2]
        all_images = np.zeros((n_units, n_applications, 100, 3, 2))

        # Constants
        step_size = 0.15*10000 #1.5
        eps = 1e-12
        compiled_activity_log = []

        for unit in range(n_units):
            print()
            print(f"Beginning Unit {unit+1}")
            for app in range(n_applications):
                input_image = np.random.normal(10, 8, size=(100, 3, 2))
                input_image = np.clip(input_image, 0, 255)

                # grad = tf.gradients(readout_blocks[f"Unit {unit}"], network.observation, name='grad')
                grad = tf.gradients(target_layer[:, app, unit], network.observation, name='grad')
                red_grad = tf.reduce_sum(grad)
                gradients = []
                # writer = tf.summary.FileWriter(f"MEI-Models/test/logs/", tf.get_default_graph())
                reds = []
                activity_log = []
                for i in range(iterations):
                    # input_image = np.concatenate((image, np.zeros((100, 1))), axis=1)
                    dy_dx, activity, red = sess.run([grad, tf.math.reduce_sum(target_layer[:, app, unit]), red_grad],
                                                    feed_dict={network.observation: input_image.astype(int)})
                    gradients.append(dy_dx)

                    update = (step_size / (np.mean(np.abs(dy_dx[0])) + eps)) * (1 / 255)
                    update = update * dy_dx[0]
                    # image = input_image[:, 0:2]
                    input_image = input_image.astype(float)
                    input_image += update

                    input_image = np.clip(input_image, 0, 255)
                    reds.append(red)

                    activity_log.append(activity)
                    activity_changes = np.array(activity_log)[1:] - np.array(activity_log)[:-1]
                    if i > 0:
                        print(f"Activity change: {activity_log[-1] - activity_log[-2]}")

                    if i > 20:
                        if np.sum(activity_changes[-10:]) <= 0:
                            print("Reducing step size")
                            step_size = step_size * 0.9
                        # if np.sum(activity_changes[-100:]) <= 0:
                        #     print("Stopping early.")
                        #     activity_log += [activity for x in range(iterations-i)]
                        #     continue

                    if np.max(np.absolute(dy_dx[0])) == 0:
                        print(f"Resetting: {unit + 1}")
                        input_image = np.random.normal(10, 8, size=(100, 3, 2))
                        input_image = np.clip(input_image, 0, 255)

                all_images[unit, app] = input_image
                compiled_activity_log.append(activity_log)

    compiled_activity_log = np.array(compiled_activity_log)
    compiled_activity_log = np.swapaxes(compiled_activity_log, 0, 1)
    plt.plot(compiled_activity_log)
    plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-activity")
    plt.clf()
    plt.close()

    all_images = all_images.astype(int)
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    # plt.imshow(np.concatenate((all_images[:, :, 0:1], np.zeros((n_units, 100, 1)), all_images[:, :, 1:2]), axis=2))
    for a in range(n_applications):
        fig, axs = plt.subplots(4, 1)
        axs[0].imshow(all_images[:, a, :, :, 0])
        axs[1].imshow(np.concatenate((np.zeros((n_units, 100, 2)), all_images[:, a, :, 1:2, 0]), axis=2).astype(int))
        axs[2].imshow(np.concatenate((all_images[:, a, :, 0:1, 0], np.zeros((n_units, 100, 2))), axis=2).astype(int))
        axs[3].imshow(np.concatenate((np.zeros((n_units, 100, 1)), all_images[:, a, :, 2:3, 0], np.zeros((n_units, 100, 1))), axis=2).astype(int))
        plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-application-{a}-mei")
        plt.clf()
        plt.close()

    plt.plot([np.sum(g) for g in gradients])
    plt.savefig(f"Generated-MEIs/Direct/{model_name}/{layer_name}-gradients")
    plt.clf()
    plt.close()

    # Save Optimal activation
    with open(f"Generated-MEIs/Direct/{model_name}/{layer_name}-optimal_activation.npy", "wb") as f:
        np.save(f, all_images)


if __name__ == "__main__":
    # produce_meis("dqn_scaffold_26-2", "conv4l", full_reafference=True, iterations=100)
    # produce_meis("dqn_scaffold_26-2", "rnn_in", full_reafference=True, iterations=2, conv=False)
    produce_meis_extended("dqn_scaffold_26-2", "conv4l", full_reafference=True, iterations=10)



