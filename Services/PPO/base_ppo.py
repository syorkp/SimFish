import copy
import numpy as np

import tensorflow.compat.v1 as tf

from Networks.PPO.proximal_policy_optimizer_critic import PPONetworkCritic


class BasePPO:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("Base PPO Constructor called")

        # Placeholders present in service base classes (overwritten by MRO)
        self.learning_params = None
        self.environment_params = None
        self.total_steps = None
        self.simulation = None
        self.buffer = None
        self.sess = None
        self.batch_size = None
        self.trace_length = None
        self.output_dimensions = None
        self.continuous = None

        self.frame_buffer = None
        self.save_frames = None

        # Networks
        self.actor_network = None
        self.critic_network = None

        # To check if is assay or training
        self.assay = None

        # Allows use of same episode method
        self.current_episode_max_duration = None
        self.total_episode_reward = 0  # Total reward over episode

        self.init_rnn_state_actor = None  # Reset RNN hidden state
        self.init_rnn_state_actor_ref = None
        self.init_rnn_state_critic = None
        self.init_rnn_state_critic_ref = None

    def init_states(self):
        # Init states for RNN
        self.init_rnn_state_actor = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))
        self.init_rnn_state_actor_ref = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))

        self.init_rnn_state_critic = (
            np.zeros([1, self.critic_network.rnn_dim]),
            np.zeros([1, self.critic_network.rnn_dim]))
        self.init_rnn_state_critic_ref = (
            np.zeros([1, self.critic_network.rnn_dim]),
            np.zeros([1, self.critic_network.rnn_dim]))

    def create_network(self):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        internal_states = sum(
            [1 for x in [self.environment_params['hunger'], self.environment_params['stress']] if x is True]) + 1

        actor_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)
        critic_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)

        output_dimension = self.output_dimensions

        self.critic_network = PPONetworkCritic(simulation=self.simulation,
                                               rnn_dim=self.learning_params['rnn_dim_shared'],
                                               rnn_cell=critic_cell,
                                               my_scope='critic',
                                               internal_states=internal_states,
                                               outputs_per_step=output_dimension,
                                               )

        return actor_cell, internal_states

    def _episode_loop(self, a):
        rnn_state_actor = copy.copy(self.init_rnn_state_actor)
        rnn_state_actor_ref = copy.copy(self.init_rnn_state_actor_ref)
        rnn_state_critic = copy.copy(self.init_rnn_state_critic)
        rnn_state_critic_ref = copy.copy(self.init_rnn_state_critic_ref)

        self.simulation.reset()
        sa = np.zeros((1, 128))  # Kept for GIFs.

        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=a,
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=self.save_frames,
                                                                                     activations=(sa,))

        self.total_episode_reward = 0  # Total reward over episode

        self.buffer.reset()
        self.buffer.action_buffer.append(a)  # Add to buffer for loading of previous actions

        self.step_number = 0
        while self.step_number < self.current_episode_max_duration:
            if self.assay is not None:
                if self.assay["reset"] and self.step_number % self.assay["reset interval"] == 0:
                    rnn_state_actor = copy.copy(self.init_rnn_state_actor)
                    rnn_state_actor_ref = copy.copy(self.init_rnn_state_actor_ref)
                    rnn_state_critic = copy.copy(self.init_rnn_state_critic)
                    rnn_state_critic_ref = copy.copy(self.init_rnn_state_critic_ref)

            self.step_number += 1

            r, internal_state, o, d, rnn_state_actor, rnn_state_actor_ref, rnn_state_critic, rnn_state_critic_ref = self.step_loop(
                o=o,
                internal_state=internal_state,
                a=a,
                rnn_state_actor=rnn_state_actor,
                rnn_state_actor_ref=rnn_state_actor_ref,
                rnn_state_critic=rnn_state_critic,
                rnn_state_critic_ref=rnn_state_critic_ref
            )

            self.total_episode_reward += r
            if d:
                break

        self.buffer.tidy()

    def compute_rnn_states(self, rnn_key_points, observation_buffer, internal_state_buffer, previous_action_buffer):
        num_actions = self.output_dimensions
        batch_size = len(rnn_key_points)

        if self.learning_params["rnn_state_computation"]:
            observation_buffer = np.vstack(observation_buffer)
            internal_state_buffer = np.vstack(internal_state_buffer)
            previous_action_buffer = np.reshape(previous_action_buffer, (observation_buffer.shape[0], num_actions))

            rnn_state_actor = copy.copy(self.init_rnn_state_actor)
            rnn_state_actor_ref = copy.copy(self.init_rnn_state_actor_ref)
            rnn_state_critic = copy.copy(self.init_rnn_state_critic)
            rnn_state_critic_ref = copy.copy(self.init_rnn_state_critic_ref)

            actor_rnn_state_buffer = ([rnn_state_actor[0][0]], [rnn_state_actor[1][0]])
            actor_rnn_state_ref_buffer = ([rnn_state_actor_ref[0][0]], [rnn_state_actor_ref[1][0]])
            critic_rnn_state_buffer = ([rnn_state_critic[0][0]], [rnn_state_critic[1][0]])
            critic_rnn_state_ref_buffer = ([rnn_state_critic_ref[0][0]], [rnn_state_critic_ref[1][0]])

            for step in range(max(rnn_key_points)):
                rnn_state_critic, rnn_state_critic_ref = self.sess.run(
                    [self.critic_network.rnn_state_shared, self.critic_network.rnn_state_ref],
                    feed_dict={self.critic_network.observation: observation_buffer[step],
                               self.critic_network.prev_actions: previous_action_buffer[step].reshape(1, num_actions),
                               self.critic_network.internal_state: internal_state_buffer[step].reshape(1, 2),

                               self.critic_network.rnn_state_in: rnn_state_critic,
                               self.critic_network.rnn_state_in_ref: rnn_state_critic_ref,

                               self.critic_network.trainLength: 1,
                               self.critic_network.batch_size: 1,
                               })
                rnn_state_actor, rnn_state_actor_ref = self.sess.run(
                    [self.actor_network.rnn_state_shared, self.actor_network.rnn_state_ref],
                    feed_dict={self.actor_network.observation: observation_buffer[step],
                               self.actor_network.prev_actions: previous_action_buffer[step].reshape(1, num_actions),
                               self.actor_network.internal_state: internal_state_buffer[step].reshape(1, 2),

                               self.actor_network.rnn_state_in: rnn_state_actor,
                               self.actor_network.rnn_state_in_ref: rnn_state_actor_ref,

                               self.actor_network.trainLength: 1,
                               self.actor_network.batch_size: 1,
                               })
                if step - 1 in rnn_key_points:
                    actor_rnn_state_buffer[0].append(rnn_state_actor[0][0])
                    actor_rnn_state_buffer[1].append(rnn_state_actor[1][0])

                    actor_rnn_state_ref_buffer[0].append(rnn_state_actor_ref[0][0])
                    actor_rnn_state_ref_buffer[1].append(rnn_state_actor_ref[1][0])

                    critic_rnn_state_buffer[0].append(rnn_state_critic[0][0])
                    critic_rnn_state_buffer[1].append(rnn_state_critic[1][0])

                    critic_rnn_state_ref_buffer[0].append(rnn_state_critic_ref[0][0])
                    critic_rnn_state_ref_buffer[1].append(rnn_state_critic_ref[1][0])

            actor_rnn_state_buffer = (np.array(actor_rnn_state_buffer[0]), np.array(actor_rnn_state_buffer[1]))
            actor_rnn_state_ref_buffer = (np.array(actor_rnn_state_ref_buffer[0]), np.array(actor_rnn_state_ref_buffer[1]))
            critic_rnn_state_buffer = (np.array(critic_rnn_state_buffer[0]), np.array(critic_rnn_state_buffer[1]))
            critic_rnn_state_ref_buffer = (
                np.array(critic_rnn_state_ref_buffer[0]), np.array(critic_rnn_state_ref_buffer[1]))
        else:
            actor_rnn_state_buffer = (
                np.zeros([batch_size, self.actor_network.rnn_dim]),
                np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
            actor_rnn_state_ref_buffer = (
                np.zeros([batch_size, self.actor_network.rnn_dim]),
                np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
            critic_rnn_state_buffer = (
                np.zeros([batch_size, self.actor_network.rnn_dim]),
                np.zeros([batch_size, self.actor_network.rnn_dim]))  # Reset RNN hidden state
            critic_rnn_state_ref_buffer = (
                np.zeros([batch_size, self.actor_network.rnn_dim]),
                np.zeros([batch_size, self.actor_network.rnn_dim]))

        return actor_rnn_state_buffer, actor_rnn_state_ref_buffer, critic_rnn_state_buffer, critic_rnn_state_ref_buffer
