import os
import json
from time import time
import re

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.simfish_env import SimState
from Network.simfish_drqn import QNetwork
from Network.experience_buffer import ExperienceBuffer
from Tools.graph_functions import update_target_graph, update_target
from Tools.make_gif import make_gif

tf.disable_v2_behavior()


class TrainingService:

    def __init__(self, environment_name="test", trial_number="1"):
        """
        An instance of TraningService handles the training of the DQN within a specified environment, according to
        specified parameters.
        :param environment_name: The name of the environment, to match the naming of the configuration files.
        :param trial_number: The index of the trial, so that agents trained under the same configuration may be
        distinguished in their output files.
        """

        # TODO: Add hyperparameter control which may belong in RunService and could handle training of multiple models.

        # Configuration
        self.configuration_location = f"./Configurations/JSON-Data/{environment_name}"
        self.params, self.env = self.load_configuration()

        # Output location
        self.output_location = f"./Output/{environment_name}_{trial_number}_output"

        self.load_model = self.check_for_model()

        # Environment and agent
        self.simulation = SimState(self.env)

        # Create networks
        self.main_QN, self.target_QN = self.create_networks()

        # Experience buffer
        self.training_buffer = ExperienceBuffer(buffer_size=self.params["exp_buffer_size"])
        self.saver = tf.train.Saver(max_to_keep=5)
        self.frame_buffer = []

        # Mathematical variables
        self.e = self.params["startE"]
        self.step_drop = (self.params['startE'] - self.params['endE']) / self.params['anneling_steps']

        # Whether to save the frames of an episode
        self.save_frames = False

        # To save the graph (placeholder)
        self.writer = None

        # Global tensorflow variables
        self.init = tf.global_variables_initializer()
        self.trainables = tf.trainable_variables()
        self.target_ops = update_target_graph(self.trainables, self.params['tau'])
        self.sess = None  # Placeholder for the tf-session.

        # Tally of steps for deciding when to use training data or to finish training.
        self.total_steps = 0
        self.episode_number = 0

        # Used for performance monitoring (not essential for algorithm).
        self.training_times = []
        self.reward_list = []

    def run(self):
        """Run the simulation, either loading a checkpoint if there or starting from scratch. If loading, uses the
        previous checkpoint to set the episode number."""

        print("Running simulation")

        with tf.Session() as self.sess:
            if self.load_model:
                print(f"Attempting to load model at {self.output_location}")
                checkpoint = tf.train.get_checkpoint_state(self.output_location)
                if hasattr(checkpoint, "model_checkpoint_path"):
                    print(checkpoint)
                    output_file_contents = os.listdir(self.output_location)
                    numbers = []
                    for name in output_file_contents:
                        if ".cptk.index" in name:
                            numbers.append(int(re.sub("[^0-9]", "", name)))
                    self.episode_number = max(numbers) + 1
                    self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
                    print("Loading successful")
                else:
                    print("No saved checkpoints found, starting from scratch.")
                    self.sess.run(self.init)
            else:
                print("First attempt at running model. Starting from scratch.")
                self.sess.run(self.init)

            update_target(self.target_ops, self.sess)  # Set the target network to be equal to the primary network.
            self.writer = tf.summary.FileWriter(f"{self.output_location}/logs/", tf.get_default_graph())

            for e_number in range(self.episode_number, self.params["num_episodes"]):
                self.episode_number = e_number
                self.episode_loop()

    def load_configuration(self):
        """
        Load the configuration files for the environment and agent parameters.
        :return: The agent parameters and the environment parameters, as json objects.
        """

        print("Loading configuration...")
        with open(f"{self.configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{self.configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def check_for_model(self):
        """
        Check whether a model for the environment and trial number exists. If not, create the output file location.
        :return: A boolean to signal whether or not a checkpoint should be loaded.
        """

        print("Checking for existing model...")
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)
            os.makedirs(f"{self.output_location}/episodes")
            os.makedirs(f"{self.output_location}/logs")
            return False
        else:
            return True

    def create_networks(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        cell_t = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        main_QN = QNetwork(self.simulation, self.params['rnn_dim'], cell, 'main', self.params['num_actions'],
                           learning_rate=self.params['learning_rate'])
        target_QN = QNetwork(self.simulation, self.params['rnn_dim'], cell_t, 'target', self.params['num_actions'],
                             learning_rate=self.params['learning_rate'])
        return main_QN, target_QN

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        t0 = time()
        episode_buffer = []

        rnn_state = (np.zeros([1, self.main_QN.rnn_dim]), np.zeros([1, self.main_QN.rnn_dim]))  # Reset RNN hidden state
        self.simulation.reset()
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        sv = np.zeros((1, 128))  # Placeholder for the state value stream

        # Take the first simulation step, with a capture action. Assigns observation, reward, internal state, done, and
        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=3,
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=self.save_frames,
                                                                                     activations=(sa,))

        # For benchmarking each episode.
        all_actions = []
        total_episode_reward = 0  # Total reward over episode

        step_number = 0  # To allow exit after maximum steps.
        a = 0  # Initialise action for episode.
        while step_number < self.params["max_epLength"]:
            step_number += 1
            o, a, r, internal_state, s1, d, rnn_state = self.step_loop(o=o, internal_state=internal_state,
                                                                       a=a, rnn_state=rnn_state)
            all_actions.append(a)
            episode_buffer.append(np.reshape(np.array([o, a, r, internal_state, s1, d]), [1, 6]))
            total_episode_reward += r
            o = s1
            if self.total_steps > self.params['pre_train_steps']:
                if self.e > self.params['endE']:
                    self.e -= self.step_drop
                if self.total_steps % (self.params['update_freq']) == 0:
                    self.train_networks()
            if d:
                break

        # Add the episode to the experience buffer
        self.save_episode(episode_start_t=t0,
                          all_actions=all_actions,
                          total_episode_reward=total_episode_reward,
                          episode_buffer=episode_buffer)

        # Print saved metrics
        # print(f"Total training time: {sum(self.training_times)}")
        # print(f"Total reward: {sum(self.reward_list)}")

    def save_episode(self, episode_start_t, all_actions, total_episode_reward, episode_buffer):
        """
        Saves the episode the the experience buffer. Also creates a gif if at interval.
        :param episode_start_t: The time at the start of the episode, used to calculate the time the episode took.
        :param all_actions: The array of all the actions taken during the episode.
        :param total_episode_reward: The total reward of the episode.
        :param episode_buffer: A buffer containing all the state transitions, actions and associated rewards yielded by
        the environment.
        :return:
        """

        print(f"episode {str(self.episode_number)}: num steps = {str(self.simulation.num_steps)}", flush=True)
        if not self.save_frames:
            self.training_times.append(time() - episode_start_t)
        episode_summary = tf.Summary(value=[tf.Summary.Value(tag="episode reward", simple_value=total_episode_reward)])
        self.writer.add_summary(episode_summary, self.total_steps)

        for act in range(self.params['num_actions']):
            action_freq = np.sum(np.array(all_actions) == act) / len(all_actions)
            a_freq = tf.Summary(value=[tf.Summary.Value(tag="action " + str(act), simple_value=action_freq)])
            self.writer.add_summary(a_freq, self.total_steps)

        buffer_array = np.array(episode_buffer)
        episode_buffer = list(zip(buffer_array))
        self.training_buffer.add(episode_buffer)
        self.reward_list.append(total_episode_reward)
        # Periodically save the model.
        if self.episode_number % self.params['summaryLength'] == 0 and self.episode_number != 0:
            print(f"mean time: {np.mean(self.training_times)}")

            self.saver.save(self.sess, f"{self.output_location}/model-{str(self.episode_number)}.cptk")
            print("Saved Model")
            print(self.total_steps, np.mean(self.reward_list[-50:]), self.e)
            print(self.frame_buffer[0].shape)
            make_gif(self.frame_buffer, f"{self.output_location}/episodes/episode-{str(self.episode_number)}.gif",
                     duration=len(self.frame_buffer) * self.params['time_per_step'], true_image=True)
            self.frame_buffer = []
            self.save_frames = False

        if (self.episode_number + 1) % self.params['summaryLength'] == 0:
            print('starting to save frames', flush=True)
            self.save_frames = True

    def step_loop(self, o, internal_state, a, rnn_state):
        """
        Runs a step, choosing an action given an initial condition using the network/randomly, and running this in the
        environment.

        :param
        session: The TF session.
        internal_state: The internal state of the agent - whether it is in light, and whether it is hungry.
        a: The previous chosen action.
        rnn_state: The state inside the RNN.

        :return:
        s: The environment state.
        chosen_a: The action chosen randomly/by the network.
        given_reward: The reward returned.
        internal_state: The internal state of the agent - whether it is in light, and whether it is hungry.
        s1: The subsequent environment state
        d: Boolean indicating agent death.
        updated_rnn_state: The updated RNN state
        """

        # Generate actions and corresponding steps.
        if np.random.rand(1) < self.e or self.total_steps < self.params['pre_train_steps']:
            [updated_rnn_state, sa, sv] = self.sess.run(
                [self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.state_in: rnn_state,
                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
            chosen_a = np.random.randint(0, self.params['num_actions'])
        else:
            chosen_a, updated_rnn_state, sa, sv = self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.state_in: rnn_state,
                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
            chosen_a = chosen_a[0]

        # Simulation step
        o1, given_reward, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=chosen_a,
                                                                                                 frame_buffer=self.frame_buffer,
                                                                                                 save_frames=self.save_frames,
                                                                                                 activations=(sa,))
        self.total_steps += 1
        return o, chosen_a, given_reward, internal_state, o1, d, updated_rnn_state

    def train_networks(self):
        """
        Trains the two networks, copying over the target network
        :return:
        """
        update_target(self.target_ops, self.sess)
        # Reset the recurrent layer's hidden state
        state_train = (np.zeros([self.params['batch_size'], self.main_QN.rnn_dim]),
                       np.zeros([self.params['batch_size'], self.main_QN.rnn_dim]))

        # Get a random batch of experiences.
        train_batch = self.training_buffer.sample(self.params['batch_size'], self.params['trace_length'])

        # Below we perform the Double-DQN update to the target Q-values
        Q1 = self.sess.run(self.main_QN.predict, feed_dict={
            self.main_QN.observation: np.vstack(train_batch[:, 4]),
            self.main_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
            self.main_QN.trainLength: self.params['trace_length'],
            self.main_QN.internal_state: np.vstack(train_batch[:, 3]),
            self.main_QN.state_in: state_train,
            self.main_QN.batch_size: self.params['batch_size'],
            self.main_QN.exp_keep: 1.0})

        Q2 = self.sess.run(self.target_QN.Q_out, feed_dict={
            self.target_QN.observation: np.vstack(train_batch[:, 4]),
            self.target_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
            self.target_QN.trainLength: self.params['trace_length'],
            self.target_QN.internal_state: np.vstack(train_batch[:, 3]),
            self.target_QN.state_in: state_train,
            self.target_QN.batch_size: self.params['batch_size'],
            self.target_QN.exp_keep: 1.0})

        end_multiplier = -(train_batch[:, 5] - 1)

        double_Q = Q2[range(self.params['batch_size'] * self.params['trace_length']), Q1]
        target_Q = train_batch[:, 2] + (self.params['y'] * double_Q * end_multiplier)
        # Update the network with our target values.
        self.sess.run(self.main_QN.updateModel,
                      feed_dict={self.main_QN.observation: np.vstack(train_batch[:, 0]),
                                 self.main_QN.targetQ: target_Q,
                                 self.main_QN.actions: train_batch[:, 1],
                                 self.main_QN.internal_state: np.vstack(train_batch[:, 3]),
                                 self.main_QN.prev_actions: np.hstack(([3], train_batch[:-1, 1])),
                                 self.main_QN.trainLength: self.params['trace_length'],
                                 self.main_QN.state_in: state_train,
                                 self.main_QN.batch_size: self.params['batch_size'],
                                 self.main_QN.exp_keep: 1.0})
