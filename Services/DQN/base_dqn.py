import numpy as np
import copy
import os
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Networks.DQN.q_network_dynamic import QNetworkDynamic
from Networks.DQN.graph_functions import update_target
from Networks.DQN.q_network import QNetwork


class BaseDQN:

    def __init__(self):

        # Placeholders present in service base classes (overwritten by MRO)
        self.learning_params = None
        self.environment_params = None
        self.total_steps = None
        self.simulation = None
        self.experience_buffer = None
        self.sess = None
        self.batch_size = None
        self.trace_length = None
        self.target_ops = None
        self.pre_train_steps = None
        self.initial_exploration_steps = None
        self.epsilon = None
        self.step_drop = None

        self.debug = False
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
        self.use_static = True  # TODO: REMOVE TEST HERE

        # Add attributes only if don't exist yet (prevents errors thrown).
        if not hasattr(self, "get_positions"):
            self.get_positions = None
        if not hasattr(self, "buffer"):
            self.buffer = None
        if not hasattr(self, "output_data"):
            self.output_data = None
        if not hasattr(self, "assay_output_data_format"):
            self.assay_output_data_format = None
        if not hasattr(self, "step_number"):
            self.step_number = None
        if not hasattr(self, "get_internal_state_order"):
            self.get_internal_state_order = None
        if not hasattr(self, "save_environmental_data"):
            self.save_environmental_data = None
        if not hasattr(self, "episode_buffer"):
            self.episode_buffer = None
        if not hasattr(self, "last_position_dim"):
            self.last_position_dim = None
        if not hasattr(self, "package_output_data"):
            self.package_output_data = None
        if not hasattr(self, "using_gpu"):
            self.using_gpu = None
        if not hasattr(self, "model_location"):
            self.model_location = None
        if not hasattr(self, "episode_number"):
            self.episode_number = None

        if not self.assay:
            self.full_efference_copy = True

    def init_states(self):
        # Init states for RNN
        print("Loading states")
        # IF SAVE PRESENT
        if os.path.isfile(f"{self.model_location}/rnn_state-{self.episode_number}.json"):
            with open(f"{self.model_location}/rnn_state-{self.episode_number}.json", 'r') as f:
                print("Successfully loaded previous state.")
                data = json.load(f)
                num_rnns = len(data.keys()) / 4
                self.init_rnn_state = tuple(
                    (np.array(data[f"rnn_state_{shape}_1"]), np.array(data[f"rnn_state_{shape}_2"])) for
                    shape in range(int(num_rnns)))
                self.init_rnn_state_ref = tuple(
                    (np.array(data[f"rnn_state_{shape}_ref_1"]), np.array(data[f"rnn_state_{shape}_ref_2"]))
                    for shape in range(int(num_rnns)))
        else:
            if self.use_static:
                rnn_state_shapes = [512]
            else:
                rnn_state_shapes = self.main_QN.get_rnn_state_shapes()

            self.init_rnn_state = tuple(
                (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)
            self.init_rnn_state_ref = tuple(
                (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        internal_states = sum(
            [1 for x in [self.environment_params['stress'],
                         self.environment_params['energy_state'], self.environment_params['in_light'],
                         self.environment_params['salt']] if x is True])
        internal_states = max(internal_states, 1)
        internal_state_names = self.get_internal_state_order()

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)
        cell_t = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)

        if "reuse_eyes" in self.learning_params:
            reuse_eyes = self.learning_params['reuse_eyes']
        else:
            reuse_eyes = False

        if self.use_static:
            self.main_QN = QNetwork(simulation=self.simulation,
                                    rnn_dim=512,
                                    rnn_cell=cell,
                                    my_scope="main",
                                    num_actions=self.learning_params["num_actions"],
                                    internal_states=internal_states,
                                    learning_rate=self.learning_params["learning_rate"]
                                    )
            self.target_QN = QNetwork(simulation=self.simulation,
                                      rnn_dim=512,
                                      rnn_cell=cell_t,
                                      my_scope="target",
                                      num_actions=self.learning_params["num_actions"],
                                      internal_states=internal_states,
                                      learning_rate=self.learning_params["learning_rate"]
                                      )
        else:
            self.main_QN = QNetworkDynamic(simulation=self.simulation,
                                           my_scope='main',
                                           internal_states=internal_states,
                                           internal_state_names=internal_state_names,
                                           num_actions=self.learning_params['num_actions'],
                                           base_network_layers=self.learning_params[
                                               'base_network_layers'],
                                           modular_network_layers=self.learning_params[
                                               'modular_network_layers'],
                                           ops=self.learning_params['ops'],
                                           connectivity=self.learning_params[
                                               'connectivity'],
                                           reflected=self.learning_params['reflected'],
                                           reuse_eyes=reuse_eyes,
                                           )
            self.target_QN = QNetworkDynamic(simulation=self.simulation,

                                             my_scope='target',
                                             internal_states=internal_states,
                                             internal_state_names=internal_state_names,
                                             num_actions=self.learning_params['num_actions'],
                                             base_network_layers=self.learning_params[
                                                 'base_network_layers'],
                                             modular_network_layers=self.learning_params[
                                                 'modular_network_layers'],
                                             ops=self.learning_params['ops'],
                                             connectivity=self.learning_params[
                                                 'connectivity'],
                                             reflected=self.learning_params['reflected'],
                                             reuse_eyes=reuse_eyes,
                                             )

    def episode_loop(self):
        """
        Loops over an episode, which involves initialisation of the environment and RNN state, then iteration over the
        steps in the episode. The relevant values are then saved to the experience buffer.
        """
        experience = []

        rnn_state = copy.copy(self.init_rnn_state)
        rnn_state_ref = copy.copy(self.init_rnn_state_ref)
        self.simulation.reset()

        # Take the first simulation step, with a capture action. Assigns observation, reward, internal state, done, and
        o, r, internal_state, d, full_masked_image = self.simulation.simulation_step(action=3)

        # For benchmarking each episode.
        all_actions = []
        total_episode_reward = 0  # Total reward over episode

        step_number = 0  # To allow exit after maximum steps.
        a = 3  # Initialise action for episode.

        if self.full_efference_copy:
            efference_copy = [a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]
        else:
            efference_copy = a
        if self.debug:
            fig, ax = plt.subplots(figsize=(4, 3))
            moviewriter = FFMpegWriter(fps=15)
            moviewriter.setup(fig, 'debug.mp4', dpi=500)
        while step_number < self.learning_params["max_epLength"]:
            print(step_number)
            step_number += 1
            o, a, r, i_s, o1, d, rnn_state, rnn_state_ref, full_masked_image = self.step_loop(o=o,
                                                                                              internal_state=internal_state,
                                                                                              a=efference_copy,
                                                                                              rnn_state=rnn_state,
                                                                                              rnn_state_ref=rnn_state_ref)
            if self.debug:
                if self.using_gpu:
                    full_masked_image = full_masked_image.get()
                full_masked_image = np.clip(full_masked_image / self.environment_params['light_gain'], 0, 1)
                ax.imshow(full_masked_image)
                moviewriter.grab_frame()
                ax.clear()
            all_actions.append(a[0])
            experience.append(np.reshape(np.array([o, np.array(a), r, internal_state, o1, d, i_s]), [1, 7]))
            total_episode_reward += r
            efference_copy = a
            internal_state = i_s

            o = o1
            if self.total_steps > self.pre_train_steps:
                if self.epsilon > self.learning_params['endE']:
                    self.epsilon -= self.step_drop
                if self.total_steps % (self.learning_params['update_freq']) == 0 and \
                        len(self.experience_buffer.buffer) > self.batch_size:
                    self.train_networks()
            if d:
                if self.use_static:
                    self.init_rnn_state = [rnn_state]
                    self.init_rnn_state_ref = [rnn_state_ref]
                else:
                    self.init_rnn_state = rnn_state
                    self.init_rnn_state_ref = rnn_state_ref
                break
        # Add the episode to the experience buffer
        if self.debug:
            moviewriter.finish()
            self.debug = False
            fig.clf()

        return all_actions, total_episode_reward, experience

    def step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
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
        feed_dict = {self.main_QN.observation: o,
                     self.main_QN.internal_state: internal_state,
                     self.main_QN.prev_actions: [a],
                     self.main_QN.train_length: 1,
                     self.main_QN.rnn_state_in: rnn_state,
                     self.main_QN.rnn_state_in_ref: rnn_state_ref,
                     self.main_QN.batch_size: 1,
                     self.main_QN.exp_keep: 1.0,
                     self.main_QN.learning_rate: self.learning_params["learning_rate"],
                     }

        if np.random.rand(1) < self.epsilon or self.total_steps < self.initial_exploration_steps:
            [updated_rnn_state, updated_rnn_state_ref] = self.sess.run(
                [self.main_QN.rnn_state_shared, self.main_QN.rnn_state_ref], feed_dict=feed_dict)
            chosen_a = np.random.randint(0, self.learning_params['num_actions'])
        else:
            chosen_a, updated_rnn_state, updated_rnn_state_ref = self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state_shared, self.main_QN.rnn_state_ref],
                feed_dict=feed_dict)
            chosen_a = chosen_a[0]

        # Simulation step
        o1, given_reward, internal_state, d, full_masked_image = self.simulation.simulation_step(action=chosen_a)

        efference_copy = [chosen_a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]
        if self.debug:
            pass
        else:
            full_masked_image = None

        if self.save_environmental_data:
            sand_grain_positions, prey_positions, predator_position = self.get_positions()
            prey_orientations = np.array([p.angle for p in self.simulation.prey_bodies]).astype(np.float32)
            try:
                predator_orientation = self.simulation.predator_body.angle
            except:
                predator_orientation = 0
            prey_ages = np.array(self.simulation.prey_ages)
            prey_gait = np.array(self.simulation.paramecia_gaits)

            self.buffer.save_environmental_positions(chosen_a,
                                                     self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     self.simulation.fish.body.angle,
                                                     self.simulation.fish.salt_health,
                                                     efference_copy=a,
                                                     prey_orientation=prey_orientations,
                                                     predator_orientation=predator_orientation,
                                                     prey_age=prey_ages,
                                                     prey_gait=prey_gait
                                                     )
            efference_copy = [chosen_a, self.simulation.fish.prev_action_impulse,
                              self.simulation.fish.prev_action_angle]

            # Update buffer
            self.buffer.add_training(observation=o1,
                                     internal_state=internal_state,
                                     action=efference_copy,
                                     reward=given_reward,
                                     rnn_state=updated_rnn_state,
                                     rnn_state_ref=updated_rnn_state_ref,
                                     )

        self.total_steps += 1
        return o, efference_copy, given_reward, internal_state, o1, d, updated_rnn_state, updated_rnn_state_ref, full_masked_image

    def assay_step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
        chosen_a, updated_rnn_state, updated_rnn_state_ref, network_layers = \
            self.sess.run(
                [self.main_QN.predict, self.main_QN.rnn_state_shared, self.main_QN.rnn_state_ref,
                 self.main_QN.network_graph],
                feed_dict={self.main_QN.observation: o,
                           self.main_QN.internal_state: internal_state,
                           self.main_QN.prev_actions: a,
                           self.main_QN.train_length: 1,
                           self.main_QN.rnn_state_in: rnn_state,
                           self.main_QN.rnn_state_in_ref: rnn_state_ref,

                           self.main_QN.batch_size: 1,
                           self.main_QN.exp_keep: 1.0,
                           })

        chosen_a = chosen_a[0]
        o1, given_reward, internal_state1, d, full_masked_image = self.simulation.simulation_step(action=chosen_a)
        sand_grain_positions, prey_positions, predator_position = self.get_positions()

        efference_copy = [chosen_a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]

        # Update buffer
        self.buffer.add_training(observation=o1,
                                 internal_state=internal_state1,
                                 action=efference_copy,
                                 reward=given_reward,
                                 rnn_state=updated_rnn_state,
                                 rnn_state_ref=updated_rnn_state_ref,
                                 )

        # Saving step data
        if "environmental positions" in self.buffer.recordings:
            prey_orientations = np.array([p.angle for p in self.simulation.prey_bodies]).astype(np.float32)
            try:
                predator_orientation = self.simulation.predator_body.angle
            except:
                predator_orientation = 0
            prey_ages = self.simulation.prey_ages
            prey_gait = self.simulation.paramecia_gaits

            self.buffer.save_environmental_positions(chosen_a,
                                                     self.simulation.fish.body.position,
                                                     self.simulation.prey_consumed_this_step,
                                                     self.simulation.predator_body,
                                                     prey_positions,
                                                     predator_position,
                                                     sand_grain_positions,
                                                     self.simulation.fish.body.angle,
                                                     self.simulation.fish.salt_health,
                                                     efference_copy=a,
                                                     prey_orientation=prey_orientations,
                                                     predator_orientation=predator_orientation,
                                                     prey_age=prey_ages,
                                                     prey_gait=prey_gait
                                                     )
        self.buffer.make_desired_recordings(network_layers)

        return o, efference_copy, given_reward, internal_state1, o1, d, updated_rnn_state

    def train_networks(self):
        """
        Trains the two networks, copying over the target network
        :return:
        """
        update_target(self.target_ops, self.sess)

        # Reset the recurrent layer's hidden state
        rnn_state_shapes = [512]  # self.main_QN.get_rnn_state_shapes()

        # Load the latest saved states... Note is technically incorrect.
        state_train = copy.copy(self.init_rnn_state)
        state_train = tuple(
            (np.tile(state_train[i][0], (self.learning_params['batch_size'], 1)),
             np.tile(state_train[i][1], (self.learning_params['batch_size'], 1)))
            for i, shape in enumerate(rnn_state_shapes))

        state_train_ref = copy.copy(self.init_rnn_state_ref)
        state_train_ref = tuple(
            (np.tile(state_train_ref[i][0], (self.learning_params['batch_size'], 1)),
             np.tile(state_train_ref[i][1], (self.learning_params['batch_size'], 1)))
            for i, shape in enumerate(rnn_state_shapes))

        # Get a random batch of experiences: ndarray 1024x6, with the six columns containing o, a, r, i_s, o1, d
        train_batch = self.experience_buffer.sample(self.learning_params['batch_size'],
                                                    self.learning_params['trace_length'])

        # Below we perform the Double-DQN update to the target Q-values
        Q1 = self.sess.run(self.main_QN.predict, feed_dict={
            self.main_QN.observation: np.vstack(train_batch[:, 4]),
            self.main_QN.prev_actions: np.vstack(train_batch[:, 1]),  # Previous actions (t+1?)
            self.main_QN.train_length: self.learning_params['trace_length'],
            self.main_QN.internal_state: np.vstack(train_batch[:, 6]),  # Internal states (t+1?)
            self.main_QN.rnn_state_in: state_train,
            self.main_QN.rnn_state_in_ref: state_train_ref,
            self.main_QN.batch_size: self.learning_params['batch_size'],
            self.main_QN.exp_keep: 1.0,
            self.main_QN.learning_rate: self.learning_params["learning_rate"],
        })

        Q2 = self.sess.run(self.target_QN.Q_out, feed_dict={
            self.target_QN.observation: np.vstack(train_batch[:, 4]),
            self.target_QN.prev_actions: np.vstack(train_batch[:, 1]),  # Previous actions (t+1?)
            self.target_QN.train_length: self.learning_params['trace_length'],
            self.target_QN.internal_state: np.vstack(train_batch[:, 6]),
            self.target_QN.rnn_state_in: state_train,
            self.target_QN.rnn_state_in_ref: state_train_ref,
            self.target_QN.batch_size: self.learning_params['batch_size'],
            self.target_QN.exp_keep: 1.0,
            self.main_QN.learning_rate: self.learning_params["learning_rate"],
        })

        end_multiplier = -(train_batch[:, 5] - 1)

        double_Q = Q2[range(self.learning_params['batch_size'] * self.learning_params['trace_length']), Q1]
        target_Q = train_batch[:, 2] + (self.learning_params[
                                            'y'] * double_Q * end_multiplier)  # target_Q = r + y*Q(s',argmax(Q(s',a)))        # Update the network with our target values.
        self.sess.run(self.main_QN.updateModel,
                      feed_dict={self.main_QN.observation: np.vstack(train_batch[:, 0]),  # Observations (t)
                                 self.main_QN.targetQ: target_Q,
                                 self.main_QN.actions: np.vstack(train_batch[:, 1])[:, 0],
                                 self.main_QN.internal_state: np.vstack(train_batch[:, 3]),  # Internal states (t?)
                                 self.main_QN.prev_actions: np.vstack(
                                     (np.array([[6, 0, 0]]), np.vstack(train_batch[:-1, 1]))),  # Previous actions (t?)
                                 self.main_QN.train_length: self.learning_params['trace_length'],
                                 self.main_QN.rnn_state_in: state_train,
                                 self.main_QN.rnn_state_in_ref: state_train_ref,
                                 self.main_QN.batch_size: self.learning_params['batch_size'],
                                 self.main_QN.exp_keep: 1.0,
                                 self.main_QN.learning_rate: self.learning_params["learning_rate"],
                                 })
