import numpy as np

import tensorflow.compat.v1 as tf

from Network.proximal_policy_optimizer_critic import PPONetworkCritic


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

        self.frame_buffer = None
        self.save_frames = None

        # Network
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
        # Init states for RNN TODO: Check not being changed. MOve to parent
        self.init_rnn_state_actor = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state
        self.init_rnn_state_actor_ref = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state

        self.init_rnn_state_critic = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))  # Reset RNN hidden state
        self.init_rnn_state_critic_ref = (
            np.zeros([1, self.actor_network.rnn_dim]),
            np.zeros([1, self.actor_network.rnn_dim]))

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
                                               outputs_per_step=output_dimension
                                               )

        return actor_cell, internal_states
