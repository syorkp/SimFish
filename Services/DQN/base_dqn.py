import numpy as np

import tensorflow.compat.v1 as tf

from Networks.DQN.q_network import QNetwork
from Tools.graph_functions import update_target


class BaseDQN:

    def __init__(self):

        # Placeholders present in service base classes (overwritten by MRO)
        self.learning_params = None
        self.environment_params = None
        self.total_steps = None
        self.simulation = None
        self.buffer = None
        self.sess = None
        self.batch_size = None
        self.trace_length = None
        self.target_ops = None
        self.pre_train_steps = None
        self.epsilon = None
        self.step_drop = None

        self.frame_buffer = None
        self.save_frames = None

        # Networks
        self.main_QN = None
        self.target_QN = None

        # To check if is assay or training
        self.assay = None

        # Allows use of same episode method
        self.current_episode_max_duration = None
        self.total_episode_reward = 0  # Total reward over episode

        self.init_rnn_state = None  # Reset RNN hidden state
        self.init_rnn_state_ref = None

    def init_states(self):
        # Init states for RNN TODO: Check not being changed. MOve to parent
        self.init_rnn_state = (
            np.zeros([1, self.main_QN.rnn_dim]),
            np.zeros([1, self.main_QN.rnn_dim]))
        self.init_rnn_state_ref = (
            np.zeros([1, self.main_QN.rnn_dim]),
            np.zeros([1, self.main_QN.rnn_dim]))

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        internal_states = sum([1 for x in [self.environment_params['hunger'], self.environment_params['stress']] if x is True]) + 1
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim'], state_is_tuple=True)
        cell_t = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim'], state_is_tuple=True)
        self.main_QN = QNetwork(self.simulation, self.learning_params['rnn_dim'], cell, 'main', self.learning_params['num_actions'], internal_states=internal_states,
                           learning_rate=self.learning_params['learning_rate'], extra_layer=self.learning_params['extra_rnn'])
        self.target_QN = QNetwork(self.simulation, self.learning_params['rnn_dim'], cell_t, 'target', self.learning_params['num_actions'], internal_states=internal_states,
                             learning_rate=self.learning_params['learning_rate'], extra_layer=self.learning_params['extra_rnn'])

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
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
        while step_number < self.learning_params["max_epLength"]:
            step_number += 1
            o, a, r, internal_state, o1, d, rnn_state = self.step_loop(o=o, internal_state=internal_state,
                                                                       a=a, rnn_state=rnn_state)
            all_actions.append(a)
            episode_buffer.append(np.reshape(np.array([o, a, r, internal_state, o1, d]), [1, 6]))
            total_episode_reward += r
            o = o1
            if self.total_steps > self.pre_train_steps:
                if self.epsilon > self.learning_params['endE']:
                    self.epsilon -= self.step_drop
                if self.total_steps % (self.learning_params['update_freq']) == 0:
                    self.train_networks()
            if d:
                break
        # Add the episode to the experience buffer
        return all_actions, total_episode_reward, episode_buffer

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
        if np.random.rand(1) < self.epsilon or self.total_steps < self.pre_train_steps:
            [updated_rnn_state, sa, sv] = self.sess.run(
                [self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.rnn_state_in: rnn_state,
                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
            chosen_a = np.random.randint(0, self.learning_params['num_actions'])
        else:
            chosen_a, updated_rnn_state, sa, sv = self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state, self.main_QN.streamA, self.main_QN.streamV],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.rnn_state_in: rnn_state,
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

    def _assay_step_loop(self, o, internal_state, a, rnn_state):
        chosen_a, updated_rnn_state, rnn2_state, sa, sv, conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r, o2 = \
            self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state, self.main_QN.rnn_state2, self.main_QN.streamA,
                 self.main_QN.streamV,
                 self.main_QN.conv1l, self.main_QN.conv2l, self.main_QN.conv3l, self.main_QN.conv4l,
                 self.main_QN.conv1r, self.main_QN.conv2r, self.main_QN.conv3r, self.main_QN.conv4r,
                 [self.main_QN.ref_left_eye, self.main_QN.ref_right_eye],
                 ],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: [a],
                           self.main_QN.trainLength: 1,
                           self.main_QN.rnn_state_in: rnn_state,
                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0})
        chosen_a = chosen_a[0]
        o1, given_reward, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=chosen_a,
                                                                                                 frame_buffer=self.frame_buffer,
                                                                                                 save_frames=True,
                                                                                                 activations=(sa,))
        fish_angle = self.simulation.fish.body.angle

        if not self.simulation.sand_grain_bodies:
            sand_grain_positions = [self.simulation.sand_grain_bodies[i].position for i, b in
                                    enumerate(self.simulation.sand_grain_bodies)]
            sand_grain_positions = [[i[0], i[1]] for i in sand_grain_positions]
        else:
            sand_grain_positions = [[10000, 10000]]

        if self.simulation.prey_bodies:
            # TODO: Note hacky fix which may want to clean up later.
            prey_positions = [prey.position for prey in self.simulation.prey_bodies]
            prey_positions = [[i[0], i[1]] for i in prey_positions]
            while True:
                if len(prey_positions) < self.last_position_dim:
                    prey_positions = np.append(prey_positions, [[10000, 10000]], axis=0)
                else:
                    break

            self.last_position_dim = len(prey_positions)

        else:
            prey_positions = np.array([[10000, 10000]])

        if self.simulation.predator_body is not None:
            predator_position = self.simulation.predator_body.position
            predator_position = np.array([predator_position[0], predator_position[1]])
        else:
            predator_position = np.array([10000, 10000])

        if self.simulation.vegetation_bodies is not None:
            vegetation_positions = [self.simulation.vegetation_bodies[i].position for i, b in
                                    enumerate(self.simulation.vegetation_bodies)]
            vegetation_positions = [[i[0], i[1]] for i in vegetation_positions]
        else:
            vegetation_positions = [[10000, 10000]]

        if not self.learning_params["extra_rnn"]:
            rnn2_state = [0.0]

        # Saving step data
        possible_data_to_save = self.package_output_data(o1, o2, chosen_a, sa, updated_rnn_state,
                                                         rnn2_state,
                                                         self.simulation.fish.body.position,
                                                         self.simulation.prey_consumed_this_step,
                                                         self.simulation.predator_body,
                                                         conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r,
                                                         prey_positions,
                                                         predator_position,
                                                         sand_grain_positions,
                                                         vegetation_positions,
                                                         fish_angle,
                                                         )
        for key in self.assay_output_data_format:
            self.output_data[key].append(possible_data_to_save[key])
        self.output_data["step"].append(self.step_number)

        return o, chosen_a, given_reward, internal_state, o1, d, updated_rnn_state

    def train_networks(self):
        """
        Trains the two networks, copying over the target network
        :return:
        """
        update_target(self.target_ops, self.sess)
        # Reset the recurrent layer's hidden state
        state_train = (np.zeros([self.learning_params['batch_size'], self.main_QN.rnn_dim]),
                       np.zeros([self.learning_params['batch_size'], self.main_QN.rnn_dim]))

        # Get a random batch of experiences: ndarray 1024x6, with the six columns containing o, a, r, i_s, o1, d
        train_batch = self.buffer.sample(self.learning_params['batch_size'], self.learning_params['trace_length'])

        # Below we perform the Double-DQN update to the target Q-values
        Q1 = self.sess.run(self.main_QN.predict, feed_dict={
            self.main_QN.observation: np.vstack(train_batch[:, 4]),
            self.main_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
            self.main_QN.trainLength: self.learning_params['trace_length'],
            self.main_QN.internal_state: np.vstack(train_batch[:, 3]),
            self.main_QN.rnn_state_in: state_train,
            self.main_QN.batch_size: self.learning_params['batch_size'],
            self.main_QN.exp_keep: 1.0})

        Q2 = self.sess.run(self.target_QN.Q_out, feed_dict={
            self.target_QN.observation: np.vstack(train_batch[:, 4]),
            self.target_QN.prev_actions: np.hstack(([0], train_batch[:-1, 1])),
            self.target_QN.trainLength: self.learning_params['trace_length'],
            self.target_QN.internal_state: np.vstack(train_batch[:, 3]),
            self.target_QN.rnn_state_in: state_train,
            self.target_QN.batch_size: self.learning_params['batch_size'],
            self.target_QN.exp_keep: 1.0})

        end_multiplier = -(train_batch[:, 5] - 1)

        double_Q = Q2[range(self.learning_params['batch_size'] * self.learning_params['trace_length']), Q1]
        target_Q = train_batch[:, 2] + (self.learning_params['y'] * double_Q * end_multiplier)
        # Update the network with our target values.
        self.sess.run(self.main_QN.updateModel,
                      feed_dict={self.main_QN.observation: np.vstack(train_batch[:, 0]),
                                 self.main_QN.targetQ: target_Q,
                                 self.main_QN.actions: train_batch[:, 1],
                                 self.main_QN.internal_state: np.vstack(train_batch[:, 3]),
                                 self.main_QN.prev_actions: np.hstack(([3], train_batch[:-1, 1])),
                                 self.main_QN.trainLength: self.learning_params['trace_length'],
                                 self.main_QN.rnn_state_in: state_train,
                                 self.main_QN.batch_size: self.learning_params['batch_size'],
                                 self.main_QN.exp_keep: 1.0})
