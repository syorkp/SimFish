import os
import json
from time import time

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.simfish_env import SimState
from Legacy.simfish_drqn import QNetwork
from Network.experience_buffer import ExperienceBuffer
from Tools.graph_functions import update_target_graph, update_target
from Tools.make_gif import make_gif

tf.disable_v2_behavior()


def run(arg="test"):
    # Setting the training parameters

    configuration_data = "./Configurations/JSON-Data/" + arg

    with open(configuration_data + '_learning.json', 'r') as f:
        params = json.load(f)

    with open(configuration_data + '_env.json', 'r') as f:
        env = json.load(f)

    sim = SimState(env)

    times = []

    # We define the cells for the primary and target q-networks
    cell = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], state_is_tuple=True)
    cellT = tf.nn.rnn_cell.LSTMCell(num_units=params['rnn_dim'], state_is_tuple=True)
    mainQN = QNetwork(sim, params['rnn_dim'], cell, 'main', params['num_actions'],
                      learning_rate=params['learning_rate'])
    targetQN = QNetwork(sim, params['rnn_dim'], cellT, 'target', params['num_actions'],
                        learning_rate=params['learning_rate'])

    init = tf.global_variables_initializer()

    saver = tf.train.Saver(max_to_keep=5)

    trainables = tf.trainable_variables()

    targetOps = update_target_graph(trainables, params['tau'])

    myBuffer = ExperienceBuffer(buffer_size=params['exp_buffer_size'])
    frame_buffer = []
    save_frames = False
    # Set the rate of random action decrease.
    e = params['startE']
    stepDrop = (params['startE'] - params['endE']) / params['anneling_steps']

    # create lists to contain total rewards and steps per episode
    rList = []
    total_steps = 0

    # Make a path for our model to be saved in.
    path = './Output/' + arg + '_output'
    if not os.path.exists(path):
        os.makedirs(path)
        os.makedirs(path + '/episodes')
        os.makedirs(path + '/logs')
        load_model = False
    else:
        load_model = True

    # Write the first line of the master log-file for the Control Center

    with tf.Session() as sess:
        if load_model:
            print(path)
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(path)
            print(ckpt)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(init)

        update_target(targetOps, sess)  # Set the target network to be equal to the primary network.

        writer = tf.summary.FileWriter(path + '/logs/', tf.get_default_graph())
        for i in range(params['num_episodes']):

            t0 = time()
            episodeBuffer = []
            env_frames = []
            sim.reset()
            sa = np.zeros((1, 128))
            sv = np.zeros((1, 128))
            s, r, internal_state, d, frame_buffer = sim.simulation_step(3, frame_buffer=frame_buffer,
                                                                        save_frames=save_frames, activations=(sa,))
            rAll = 0
            j = 0
            state = (np.zeros([1, mainQN.rnn_dim]), np.zeros([1, mainQN.rnn_dim]))  # Reset the recurrent layer's hidden state
            a = 0
            all_actions = []
            # The Q-Network
            while j < params['max_epLength']:
                j += 1
                # Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < params['pre_train_steps']:
                    [state1, sa, sv] = sess.run([mainQN.rnn_state, mainQN.streamA, mainQN.streamV],
                                                feed_dict={mainQN.observation: s, mainQN.internal_state: internal_state,
                                                           mainQN.prev_actions: [a], mainQN.trainLength: 1,
                                                           mainQN.state_in: state, mainQN.batch_size: 1,
                                                           mainQN.exp_keep: 1.0})
                    a = np.random.randint(0, params['num_actions'])
                else:
                    a, state1, sa, sv = sess.run([mainQN.predict, mainQN.rnn_state, mainQN.streamA, mainQN.streamV],
                                                 feed_dict={mainQN.observation: s,
                                                            mainQN.internal_state: internal_state,
                                                            mainQN.prev_actions: [a], mainQN.trainLength: 1,
                                                            mainQN.state_in: state, mainQN.batch_size: 1,
                                                            mainQN.exp_keep: 1.0})
                    a = a[0]

                all_actions.append(a)
                s1, r, internal_state, d, frame_buffer = sim.simulation_step(a, frame_buffer=frame_buffer,
                                                                             save_frames=save_frames,
                                                                             activations=(sa,))
                total_steps += 1
                episodeBuffer.append(np.reshape(np.array([s, a, r, internal_state, s1, d]), [1, 6]))
                if total_steps > params['pre_train_steps']:
                    if e > params['endE']:
                        e -= stepDrop

                    if total_steps % (params['update_freq']) == 0:
                        update_target(targetOps, sess)
                        # Reset the recurrent layer's hidden state
                        state_train = (np.zeros([params['batch_size'], mainQN.rnn_dim]),
                                       np.zeros([params['batch_size'], mainQN.rnn_dim]))

                        trainBatch = myBuffer.sample(params['batch_size'],
                                                     params['trace_length'])  # Get a random batch of experiences.
                        # Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.predict, feed_dict={
                            mainQN.observation: np.vstack(trainBatch[:, 4]),
                            mainQN.prev_actions: np.hstack(([0], trainBatch[:-1, 1])),
                            mainQN.trainLength: params['trace_length'],
                            mainQN.internal_state: np.vstack(trainBatch[:, 3]), mainQN.state_in: state_train,
                            mainQN.batch_size: params['batch_size'], mainQN.exp_keep: 1.0})
                        Q2 = sess.run(targetQN.Q_out, feed_dict={
                            targetQN.observation: np.vstack(trainBatch[:, 4]),
                            targetQN.prev_actions: np.hstack(([0], trainBatch[:-1, 1])),
                            targetQN.trainLength: params['trace_length'],
                            targetQN.internal_state: np.vstack(trainBatch[:, 3]), targetQN.state_in: state_train,
                            targetQN.batch_size: params['batch_size'], targetQN.exp_keep: 1.0})
                        end_multiplier = -(trainBatch[:, 5] - 1)

                        doubleQ = Q2[range(params['batch_size'] * params['trace_length']), Q1]
                        targetQ = trainBatch[:, 2] + (params['y'] * doubleQ * end_multiplier)
                        # Update the network with our target values.
                        sess.run(mainQN.updateModel,
                                 feed_dict={mainQN.observation: np.vstack(trainBatch[:, 0]),
                                            mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1],
                                            mainQN.internal_state: np.vstack(trainBatch[:, 3]),
                                            mainQN.prev_actions: np.hstack(([3], trainBatch[:-1, 1])),
                                            mainQN.trainLength: params['trace_length'],
                                            mainQN.state_in: state_train, mainQN.batch_size: params['batch_size'],
                                            mainQN.exp_keep: 1.0})
                rAll += r
                s = s1
                state = state1
                if d:
                    break

            # Add the episode to the experience buffer
            print('episode ' + str(i) + ': num steps = ' + str(sim.num_steps), flush=True)
            if not save_frames:
                times.append(time() - t0)
            episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode reward", simple_value=rAll)])
            writer.add_summary(episode_summary, total_steps)

            for act in range(params['num_actions']):
                action_freq = np.sum(np.array(all_actions) == act) / len(all_actions)
                a_freq = tf.Summary(value=[tf.Summary.Value(tag="action " + str(act), simple_value=action_freq)])
                writer.add_summary(a_freq, total_steps)

            bufferArray = np.array(episodeBuffer)
            episodeBuffer = list(zip(bufferArray))
            myBuffer.add(episodeBuffer)
            rList.append(rAll)
            # Periodically save the model.
            if i % params['summaryLength'] == 0 and i != 0:
                print('mean time:')
                print(np.mean(times))
                saver.save(sess, path + '/model-' + str(i) + '.cptk')

                print("Saved Model")
                print(total_steps, np.mean(rList[-50:]), e)
                print(frame_buffer[0].shape)
                make_gif(frame_buffer, path + '/episodes/episode-' + str(i) + '.gif',
                         duration=len(frame_buffer) * params['time_per_step'], true_image=True)
                frame_buffer = []
                save_frames = False

            if (i + 1) % params['summaryLength'] == 0:
                print('starting to save frames', flush=True)
                save_frames = True
            # print(f"Total training time: {sum(times)}")
            print(f"Total reward: {sum(rList)}")
