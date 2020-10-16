import os
import json
from time import time

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
        Should be initialised with the
        name of the run version, which is used to find the configuration, as well as checkpoints and output data,
        For now, all attributes below are intended to refer to a single training period. In future, it may be useful to
        separate values. between this training service and a run service which handles multiple training services.
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

        # Used for performance monitoring (not essential for algorithm).
        self.training_times = []
        self.reward_list = []

    def run(self):
        """Run the simulation"""

        print("Running simulation")

        # Write the first line of the master log-file for the Control Center
        with tf.Session() as self.sess:
            if self.load_model:
                print(f"Loading Model at {self.output_location}")
                checkpoint = tf.train.get_checkpoint_state(self.output_location)
                print(checkpoint)
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            else:
                self.sess.run(self.init)

            update_target(self.target_ops, self.sess)  # Set the target network to be equal to the primary network.
            self.writer = tf.summary.FileWriter(f"{self.output_location}/logs/", tf.get_default_graph())

            for e_number in range(self.params["num_episodes"]):
                self.episode_loop(episode_number=e_number)

    def load_configuration(self):
        """Load the configuration files for the environment and agent parameters."""

        print("Loading configuration...")
        with open(f"{self.configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{self.configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def check_for_model(self):
        """Check whether a model for the environment and trial number exists. If not, create output file location"""

        print("Checking for existing model...")
        if not os.path.exists(self.output_location):
            os.makedirs(self.output_location)
            os.makedirs(f"{self.output_location}/episodes")
            os.makedirs(f"{self.output_location}/logs")
            return False
        else:
            return True

    def create_networks(self):
        print("Creating networks...")
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        cell_t = tf.nn.rnn_cell.LSTMCell(num_units=self.params['rnn_dim'], state_is_tuple=True)
        main_QN = QNetwork(self.simulation, self.params['rnn_dim'], cell, 'main', self.params['num_actions'],
                           learning_rate=self.params['learning_rate'])
        target_QN = QNetwork(self.simulation, self.params['rnn_dim'], cell_t, 'target', self.params['num_actions'],
                             learning_rate=self.params['learning_rate'])
        return main_QN, target_QN

    def episode_loop(self, episode_number):
        # TODO: Rename all parameters given in the episode and step loops.
        t0 = time()
        episode_buffer = []
        environment_frames = []  # TODO:What was this for?

        rnn_state = (np.zeros([1, self.main_QN.rnn_dim]), np.zeros([1, self.main_QN.rnn_dim]))  # Reset RNN hidden state
        self.simulation.reset()
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        sv = np.zeros((1, 128))  # Placeholder for the state value stream

        # Take the first simulation step, with a capture action.
        s, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=3,
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
            s, a, r, internal_state, s1, d, rnn_state_2 = self.step_loop(s=s, internal_state=internal_state,
                                                                         a=a, rnn_state=rnn_state)
            all_actions.append(a)
            episode_buffer.append(np.reshape(np.array([s, a, r, internal_state, s1, d]), [1, 6]))
            total_episode_reward += r
            s = s1
            rnn_state = rnn_state_2
            if self.total_steps > self.params['pre_train_steps']:
                self.train_networks()
            if d:
                break

        # Add the episode to the experience buffer
        self.save_episode(episode_start_t=t0,
                          episode_number=episode_number,
                          all_actions=all_actions,
                          total_episode_reward=total_episode_reward,
                          episode_buffer=episode_buffer)

        # Print saved metrics
        # print(f"Total training time: {sum(self.training_times)}")
        # print(f"Total reward: {sum(self.reward_list)}")

    def save_episode(self, episode_start_t, episode_number, all_actions, total_episode_reward, episode_buffer):
        print(f"episode {str(episode_number)}: num steps = {str(self.simulation.num_steps)}", flush=True)
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
        if episode_number % self.params['summaryLength'] == 0 and episode_number != 0:
            print(f"mean time: {np.mean(self.training_times)}")

            self.saver.save(self.sess, f"{self.output_location}/model-{str(episode_number)}.cptk")
            print("Saved Model")
            print(self.total_steps, np.mean(self.reward_list[-50:]), self.e)
            print(self.frame_buffer[0].shape)
            make_gif(self.frame_buffer, f"{self.output_location}/episodes/episode-{str(episode_number)}.gif",
                     duration=len(self.frame_buffer) * self.params['time_per_step'], true_image=True)
            self.frame_buffer = []
            self.save_frames = False

        if (episode_number + 1) % self.params['summaryLength'] == 0:
            print('starting to save frames', flush=True)
            self.save_frames = True

    def step_loop(self, s, internal_state, a, rnn_state):
        """
        Returns the outputs of the step, which follows runnnig of the graphs.
        :param
        session:
        :return:
        s:
        internal_state: The internal state of the agent - whether it is in light, and whether it is hungry.
        a:
        state:
        """

        # Generate actions and corresponding steps.
        if np.random.rand(1) < self.e or self.total_steps < self.params['pre_train_steps']:
            [state1, sa, sv] = self.sess.run([self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                                             feed_dict={self.main_QN.observation: s,
                                                        self.main_QN.internal_state: internal_state,
                                                        self.main_QN.prev_actions: [a], self.main_QN.trainLength: 1,
                                                        self.main_QN.state_in: rnn_state, self.main_QN.batch_size: 1,
                                                        self.main_QN.exp_keep: 1.0})
            a = np.random.randint(0, self.params['num_actions'])
        else:
            a, state1, sa, sv = self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: s,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a], self.main_QN.trainLength: 1,
                           self.main_QN.state_in: rnn_state, self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
            a = a[0]

        # Simulation step
        s1, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=a,
                                                                                      frame_buffer=self.frame_buffer,
                                                                                      save_frames=self.save_frames,
                                                                                      activations=(sa,))
        self.total_steps += 1
        return s, a, r, internal_state, s1, d, state1

    def train_networks(self):
        if self.e > self.params['endE']:
            self.e -= self.step_drop

        if self.total_steps % (self.params['update_freq']) == 0:
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
                self.main_QN.internal_state: np.vstack(train_batch[:, 3]), self.main_QN.state_in: state_train,
                self.main_QN.batch_size: self.params['batch_size'], self.main_QN.exp_keep: 1.0})

            Q2 = self.sess.run(self.target_QN.Q_out, feed_dict={
                self.target_QN.observation: np.vstack(train_batch[:, 4]),
                self.target_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
                self.target_QN.trainLength: self.params['trace_length'],
                self.target_QN.internal_state: np.vstack(train_batch[:, 3]), self.target_QN.state_in: state_train,
                self.target_QN.batch_size: self.params['batch_size'], self.target_QN.exp_keep: 1.0})

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
