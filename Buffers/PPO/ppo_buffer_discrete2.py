import numpy as np

from Buffers.PPO.base_ppo_buffer import BasePPOBuffer


class PPOBufferDiscrete2(BasePPOBuffer):
    """Buffer for full episode for PPO training, and logging."""

    def __init__(self, gamma, lmbda, batch_size, train_length, assay, debug=False, use_dynamic_network=False):

        super().__init__(gamma, lmbda, batch_size, train_length, assay, debug, use_dynamic_network)

        # Buffer for training
        self.log_action_probability_buffer = []

        self.actor_loss_buffer = []

        # For assay saving (Make more elegant)
        self.multivariate = True

    def reset(self):
        super().reset()
        # Buffer for training

        self.log_action_probability_buffer = []

    def add_training(self, observation, internal_state, action, reward, value, l_p_action, actor_rnn_state,
                     actor_rnn_state_ref, critic_rnn_state, critic_rnn_state_ref):
        super().add_training(observation, internal_state, reward, action, value, actor_rnn_state,
                             actor_rnn_state_ref, critic_rnn_state, critic_rnn_state_ref)

        self.log_action_probability_buffer.append(l_p_action)
        self.action_buffer.append(int(action))

    def add_logging(self):
        print("Additional logs not set up for discrete.")

    def add_loss(self, action_loss, critic_loss):
        self.actor_loss_buffer.append(action_loss)
        self.critic_loss_buffer.append(critic_loss)

    def tidy(self):
        super().tidy()
        self.log_action_probability_buffer = np.array(self.log_action_probability_buffer)

    def get_episode_buffer(self):
        """Returns the episodes buffer, formatted as: X(Variable).TraceLength.DataDim, with zero padding for incomplete
        traces"""
        episode_duration = len(self.observation_buffer)
        available_steps = episode_duration - 1
        slice_steps = range(0, available_steps, self.trace_length)
        observation_batch = []
        internal_state_batch = []
        action_batch = []
        previous_action_batch = []
        log_action_probability_batch = []
        advantage_batch = []
        return_batch = []
        value_batch = []
        for slice in slice_steps:
            if slice == slice_steps[-1]:
                observation_slice, internal_state_slice, action_slice, previous_action_slice, \
                log_action_probability_slice, advantage_slice, return_slice, value_slice = self.get_batch(final_batch=True)
            else:
                observation_slice, internal_state_slice, action_slice, previous_action_slice, \
                log_action_probability_slice, advantage_slice, return_slice, value_slice = self.get_batch(final_batch=False)

            observation_batch.append(observation_slice)
            internal_state_batch.append(internal_state_slice)
            action_batch.append(action_slice)
            previous_action_batch.append(previous_action_slice)
            log_action_probability_batch.append(log_action_probability_slice)
            advantage_batch.append(advantage_slice)
            return_batch.append(return_slice)
            value_batch.append(value_slice)

        return np.array(observation_batch), np.array(internal_state_batch), np.array(action_batch), \
               np.array(previous_action_batch), np.array(log_action_probability_batch), np.array(
            advantage_batch), np.array(return_batch), np.array(value_batch), slice_steps

    def get_batch(self, final_batch):
        """Gets a trace worth of data (or batch, as used previously)"""
        if final_batch:
            observation_slice = self.pad_slice(self.observation_buffer[self.pointer:-1, :], self.trace_length)
            internal_state_slice = self.pad_slice(self.internal_state_buffer[self.pointer:-1, :], self.trace_length)
            action_slice = self.pad_slice(self.action_buffer[self.pointer + 1:-1], self.trace_length)
            previous_action_slice = self.pad_slice(self.action_buffer[self.pointer:-2], self.trace_length)
            # reward_slice = self.reward_buffer[self.pointer:-1], self.trace_length)
            value_slice = self.pad_slice(self.value_buffer[self.pointer:-2], self.trace_length)
            log_action_probability_slice = self.pad_slice(self.log_action_probability_buffer[self.pointer:-1],
                                                          self.trace_length)
            advantage_slice = self.pad_slice(self.advantage_buffer[self.pointer:], self.trace_length)
            return_slice = self.pad_slice(self.return_buffer[self.pointer:], self.trace_length)
            # actor_rnn_state_slice = self.actor_rnn_state_buffer[self.pointer:-1]
            # actor_rnn_state_ref_slice = self.actor_rnn_state_ref_buffer[self.pointer:-1]
            # critic_rnn_state_slice = self.critic_rnn_state_buffer[self.pointer:-1]
            # critic_rnn_state_ref_slice = self.critic_rnn_state_ref_buffer[self.pointer:-1]

        else:
            observation_slice = self.observation_buffer[self.pointer:self.pointer + self.trace_length, :]
            internal_state_slice = self.internal_state_buffer[self.pointer:self.pointer + self.trace_length, :]
            action_slice = self.action_buffer[self.pointer + 1:self.pointer + self.trace_length + 1]
            previous_action_slice = self.action_buffer[self.pointer:self.pointer + self.trace_length]
            # reward_slice = self.reward_buffer[self.pointer:self.pointer + self.trace_length, ]
            if self.pointer == 0:
                value_slice = self.value_buffer[self.pointer:self.pointer + self.trace_length-1, ]
                value_slice = np.concatenate((np.array([0]), value_slice))
            else:
                value_slice = self.value_buffer[self.pointer-1:self.pointer + self.trace_length-1, ]
            log_action_probability_slice = self.log_action_probability_buffer[
                                           self.pointer:self.pointer + self.trace_length]
            advantage_slice = self.advantage_buffer[self.pointer:self.pointer + self.trace_length]
            return_slice = self.return_buffer[self.pointer:self.pointer + self.trace_length]
            # actor_rnn_state_slice = self.actor_rnn_state_buffer[self.pointer:self.pointer + self.trace_length]
            # actor_rnn_state_ref_slice = self.actor_rnn_state_ref_buffer[self.pointer:self.pointer + self.trace_length]
            # critic_rnn_state_slice = self.critic_rnn_state_buffer[self.pointer:self.pointer + self.trace_length]
            # critic_rnn_state_ref_slice = self.critic_rnn_state_ref_buffer[self.pointer:self.pointer + self.trace_length]

        self.pointer += self.trace_length

        return observation_slice, internal_state_slice, action_slice, previous_action_slice, \
               log_action_probability_slice, advantage_slice, return_slice, value_slice,\
            # actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, critic_rnn_state_ref_slice

    def check_buffers(self):
        # Check for NaN
        print("Checking Buffers")
        buffers = [self.advantage_buffer, self.reward_buffer, self.observation_buffer, self.action_buffer,
                   self.return_buffer, self.value_buffer, self.log_action_probability_buffer]
        if np.isnan(np.sum(np.sum(buffer) for buffer in buffers)):
            print("NaN Detected")

        print("Buffers fine")

        # TODO: Add methods for detecting values outside of range.

    def save_assay_data(self, assay_id, data_save_location, assay_configuration_id):
        hdf5_file, assay_group = BasePPOBuffer.save_assay_data(self, assay_id, data_save_location, assay_configuration_id)

        hdf5_file.close()

    def get_rnn_batch(self, key_points, batch_size):
        actor_rnn_state_batch = []
        actor_rnn_state_batch_ref = []

        for point in key_points:
            actor_rnn_state_batch.append(self.actor_rnn_state_buffer[point])
            actor_rnn_state_batch_ref.append(self.actor_rnn_state_ref_buffer[point])

        actor_rnn_state_batch = np.reshape(np.array(actor_rnn_state_batch), (batch_size, 2, 512))
        actor_rnn_state_batch_ref = np.reshape(np.array(actor_rnn_state_batch_ref), (batch_size, 2, 512))

        actor_rnn_state_batch = (np.array(actor_rnn_state_batch[:, 0, :]), np.array(actor_rnn_state_batch[:, 1, :]))
        actor_rnn_state_batch_ref = (np.array(actor_rnn_state_batch_ref[:, 0, :]), np.array(actor_rnn_state_batch_ref[:, 1, :]))

        return actor_rnn_state_batch, actor_rnn_state_batch_ref

    def calculate_advantages_and_returns(self, normalise_advantage=True):
        # Advantages = returns-values. Then scaled
        # mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        # mb_returns = mb_advs + mb_values


        # delta = self.reward_buffer[:-1] + self.gamma * self.value_buffer[1:] - self.value_buffer[:-1]
        # advantage = self.discount_cumsum(delta, self.gamma * self.lmbda)
        # if normalise_advantage:
        #     advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        # returns = self.discount_cumsum(self.reward_buffer, self.gamma)[:-1]
        # self.advantage_buffer = advantage
        # self.return_buffer = returns
        last_values = self.value_buffer
        advantages = np.zeros_like(self.reward_buffer)
        last_gae_lam = 0
        for step in reversed(range(len(self.observation_buffer))):
            if step == len(self.observation_buffer) - 1:
                nextnonterminal = 1.0# - self.dones
                nextvalues = 0
            else:
                nextnonterminal = 1.0# - mb_dones[step + 1]
                nextvalues = self.value_buffer[step + 1]
            delta = self.reward_buffer[step] + self.gamma * nextvalues * nextnonterminal - self.value_buffer[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.lmbda * nextnonterminal * last_gae_lam
        returns = advantages + self.value_buffer
        self.advantage_buffer = advantages
        self.return_buffer = returns

        if self.debug:
            self.check_buffers()