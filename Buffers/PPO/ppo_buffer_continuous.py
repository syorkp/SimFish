from Buffers.PPO.base_ppo_buffer import BasePPOBuffer
import numpy as np


class PPOBufferContinuousMultivariate2(BasePPOBuffer):
    """Buffer for full episode for PPO training, and logging."""

    def __init__(self, gamma, lmbda, batch_size, train_length, assay, debug=False):
        super().__init__(gamma, lmbda, batch_size, train_length, assay, debug)

        # Buffer for training
        self.log_action_probability_buffer = []

        # Buffer purely for logging
        self.mu_i_buffer = []
        self.si_i_buffer = []
        self.mu_a_buffer = []
        self.si_a_buffer = []

        self.mu1_buffer = []
        self.mu1_ref_buffer = []
        self.mu_a1_buffer = []
        self.mu_a_ref_buffer = []

        self.impulse_loss_buffer = []
        self.angle_loss_buffer = []
        self.entropy_loss_buffer = []

        self.prediction_error_buffer = []
        self.target_output_buffer = []

        self.action_consequences_buffer = []

        # For assay saving
        self.multivariate = True

        self.efference_copy_buffer = []

    def reset(self):
        super().reset()
        # Buffer for training
        self.log_action_probability_buffer = []

        # Buffer purely for logging
        self.mu_i_buffer = []
        self.si_i_buffer = []
        self.mu_a_buffer = []
        self.si_a_buffer = []

        self.mu1_buffer = []
        self.mu1_ref_buffer = []
        self.mu_a1_buffer = []
        self.mu_a_ref_buffer = []

        self.impulse_loss_buffer = []
        self.angle_loss_buffer = []
        self.entropy_loss_buffer = []

        self.prediction_error_buffer = []
        self.target_output_buffer = []

        self.action_consequences_buffer = []
        self.efference_copy_buffer = []

    def add_training(self, observation, internal_state, action, reward, value, l_p_action, actor_rnn_state,
                     actor_rnn_state_ref, critic_rnn_state=None, critic_rnn_state_ref=None, prediction_error=None,
                     target_output=None):
        self.observation_buffer.append(observation)
        self.internal_state_buffer.append(internal_state)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)

        self.actor_rnn_state_buffer.append(actor_rnn_state)
        self.actor_rnn_state_ref_buffer.append(actor_rnn_state_ref)
        self.log_action_probability_buffer.append(l_p_action)
        self.action_buffer.append(action)

        if critic_rnn_state is not None:
            self.critic_rnn_state_buffer.append(critic_rnn_state)
            self.critic_rnn_state_ref_buffer.append(critic_rnn_state_ref)

    def add_logging(self, mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref):
        self.mu_i_buffer.append(mu_i)
        self.si_i_buffer.append(si_i)
        self.mu_a_buffer.append(mu_a)
        self.si_a_buffer.append(si_a)
        self.mu1_buffer.append(mu1)
        self.mu1_ref_buffer.append(mu1_ref)
        self.mu_a1_buffer.append(mu_a1)
        self.mu_a_ref_buffer.append(mu_a_ref)

    def add_loss(self, impulse_loss, angle_loss, critic_loss, entropy_loss):
        self.impulse_loss_buffer.append(impulse_loss)
        self.angle_loss_buffer.append(angle_loss)
        self.critic_loss_buffer.append(critic_loss)
        self.entropy_loss_buffer.append(entropy_loss)

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
                log_action_probability_slice, advantage_slice, return_slice, value_slice = self.get_batch(
                    final_batch=True)
            else:
                observation_slice, internal_state_slice, action_slice, previous_action_slice, \
                log_action_probability_slice, advantage_slice, return_slice, value_slice = self.get_batch(
                    final_batch=False)

            observation_batch.append(observation_slice)
            internal_state_batch.append(internal_state_slice)
            action_batch.append(np.array(action_slice))
            previous_action_batch.append(previous_action_slice)
            log_action_probability_batch.append(log_action_probability_slice)
            advantage_batch.append(advantage_slice)
            return_batch.append(return_slice)
            value_batch.append(value_slice)

        return np.array(observation_batch), np.array(internal_state_batch), np.array(action_batch), \
               np.array(previous_action_batch), np.array(log_action_probability_batch), \
               np.array(advantage_batch), np.array(return_batch), np.array(value_batch), slice_steps

    def get_batch(self, final_batch):
        """Gets a trace worth of data (or batch, as used previously)"""
        if final_batch:
            observation_slice = self.pad_slice(self.observation_buffer[self.pointer:-1, :], self.trace_length, "o")
            internal_state_slice = self.pad_slice(self.internal_state_buffer[self.pointer:-1, :], self.trace_length, "i")
            action_slice = self.pad_slice(self.action_buffer[self.pointer + 1:-1, :2], self.trace_length, "a")
            previous_action_slice = self.pad_slice(self.action_buffer[self.pointer:-2, :], self.trace_length, "p")
            value_slice = self.pad_slice(self.value_buffer[self.pointer:-2], self.trace_length, "v")
            log_action_probability_slice = self.pad_slice(self.log_action_probability_buffer[self.pointer:-1], self.trace_length, "la")
            advantage_slice = self.pad_slice(self.advantage_buffer[self.pointer:self.pointer + 50], self.trace_length, "ad")
            return_slice = self.pad_slice(self.return_buffer[self.pointer:self.pointer + 50], self.trace_length, "re")
            # actor_rnn_state_slice = self.actor_rnn_state_buffer[self.pointer:-1]
            # actor_rnn_state_ref_slice = self.actor_rnn_state_ref_buffer[self.pointer:-1]
            # critic_rnn_state_slice = self.critic_rnn_state_buffer[self.pointer:-1]
            # critic_rnn_state_ref_slice = self.critic_rnn_state_ref_buffer[self.pointer:-1]

        else:
            observation_slice = self.observation_buffer[self.pointer:self.pointer + self.trace_length, :]
            internal_state_slice = self.internal_state_buffer[self.pointer:self.pointer + self.trace_length, :]
            action_slice = self.action_buffer[self.pointer + 1:self.pointer + self.trace_length + 1, :2]
            previous_action_slice = self.action_buffer[self.pointer:self.pointer + self.trace_length, :]
            # reward_slice = self.reward_buffer[self.pointer:self.pointer + self.trace_length, ]
            if self.pointer == 0:
                value_slice = self.value_buffer[self.pointer:self.pointer + self.trace_length - 1, ]
                value_slice = np.concatenate((np.array([0]), value_slice))
            else:
                value_slice = self.value_buffer[self.pointer - 1:self.pointer + self.trace_length - 1, ]
            try:
                log_action_probability_slice = self.log_action_probability_buffer[
                                               self.pointer:self.pointer + self.trace_length, :]
            except ValueError:
                print(f"Pointer: {self.pointer}")
                print(f"Trace length: {self.trace_length}")
                print(f"Shape log action prob buffer: {self.log_action_probability_buffer.shape}")

            advantage_slice = self.advantage_buffer[self.pointer:self.pointer + self.trace_length]
            return_slice = self.return_buffer[self.pointer:self.pointer + self.trace_length]
            # actor_rnn_state_slice = self.actor_rnn_state_buffer[self.pointer:self.pointer + self.trace_length]
            # actor_rnn_state_ref_slice = self.actor_rnn_state_ref_buffer[self.pointer:self.pointer + self.trace_length]
            # critic_rnn_state_slice = self.critic_rnn_state_buffer[self.pointer:self.pointer + self.trace_length]
            # critic_rnn_state_ref_slice = self.critic_rnn_state_ref_buffer[self.pointer:self.pointer + self.trace_length]

            target_output_slice = np.array(self.target_output_buffer[self.pointer:self.pointer + self.trace_length])

        self.pointer += self.trace_length

        return observation_slice, internal_state_slice, action_slice, previous_action_slice, \
               log_action_probability_slice, advantage_slice, return_slice, value_slice
        # actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, critic_rnn_state_ref_slice

    def save_assay_data(self, assay_id, data_save_location, assay_configuration_id, sediment, internal_state_order=None, salt_location=None):
        hdf5_file, assay_group = BasePPOBuffer.save_assay_data(self, assay_id=assay_id, data_save_location=data_save_location,
                                                               assay_configuration_id=assay_configuration_id, sediment=sediment,
                                                               internal_state_order=internal_state_order,
                                                               salt_location=salt_location)

        # self.create_data_group("mu_impulse", np.array(self.mu_i_buffer)[:, 0], assay_group)
        # self.create_data_group("mu_angle", np.array(self.mu_a_buffer)[:, 0], assay_group)
        # self.create_data_group("sigma_impulse", np.array(self.si_i_buffer)[:, 0], assay_group)
        # self.create_data_group("sigma_angle", np.array(self.si_a_buffer)[:, 0], assay_group)

        print(f"{assay_id} Data Saved")
        hdf5_file.close()

    def check_buffers(self):
        # Check for NaN
        print("Checking Buffers")
        buffers = [self.advantage_buffer, self.reward_buffer, self.observation_buffer, self.action_buffer,
                   self.return_buffer, self.value_buffer, self.log_action_probability_buffer]
        if np.isnan(np.sum(np.sum(buffer) for buffer in buffers)):
            print("NaN Detected")

        print("Buffers fine")

        # TODO: Add methods for detecting values outside of range.

    def get_rnn_batch(self, key_points, batch_size, critic=False):
        actor_rnn_state_batch = []
        actor_rnn_state_batch_ref = []

        for point in key_points:
            if critic:
                actor_rnn_state_batch.append(self.critic_rnn_state_buffer[point])
                actor_rnn_state_batch_ref.append(self.critic_rnn_state_ref_buffer[point])
            else:
                actor_rnn_state_batch.append(self.actor_rnn_state_buffer[point])
                actor_rnn_state_batch_ref.append(self.actor_rnn_state_ref_buffer[point])

        n_rnns = np.array(actor_rnn_state_batch).shape[1]
        n_units = np.array(actor_rnn_state_batch).shape[-1]

        actor_rnn_state_batch = np.reshape(np.array(actor_rnn_state_batch), (n_rnns, batch_size, 2, n_units))
        actor_rnn_state_batch_ref = np.reshape(np.array(actor_rnn_state_batch_ref),
                                               (n_rnns, batch_size, 2, n_units))

        actor_rnn_state_batch = tuple(
            (np.array(actor_rnn_state_batch[i, :, 0, :]), np.array(actor_rnn_state_batch[i, :, 1, :])) for i in
            range(n_rnns))
        actor_rnn_state_batch_ref = tuple(
            (np.array(actor_rnn_state_batch_ref[i, :, 0, :]), np.array(actor_rnn_state_batch_ref[i, :, 1, :])) for i
            in range(n_rnns))

        return actor_rnn_state_batch, actor_rnn_state_batch_ref

    def update_rewards_rnd(self):
        # reward_std = np.std(self.return_buffer)
        # normalised_prediction_error = np.array(self.prediction_error_buffer) * reward_std
        # normalised_prediction_error = np.sum(normalised_prediction_error, axis=2)
        # normalised_prediction_error = normalised_prediction_error.reshape(self.return_buffer.shape[0])
        normalised_prediction_error = np.sum(np.array(self.prediction_error_buffer), axis=2) * 100
        normalised_prediction_error = normalised_prediction_error.reshape(self.return_buffer.shape[0])
        self.return_buffer += normalised_prediction_error

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
        value_buffer = np.array([v[0, 0] for v in self.value_buffer])
        last_values = self.value_buffer
        advantages = np.zeros_like(self.reward_buffer)
        last_gae_lam = 0
        for step in reversed(range(len(self.observation_buffer))):
            if step == len(self.observation_buffer) - 1:
                nextnonterminal = 1.0  # - self.dones
                nextvalues = 0
            else:
                nextnonterminal = 1.0  # - mb_dones[step + 1]
                nextvalues = self.value_buffer[step + 1]
            delta = self.reward_buffer[step] + self.gamma * nextvalues * nextnonterminal - self.value_buffer[step]
            advantages[step] = last_gae_lam = delta + self.gamma * self.lmbda * nextnonterminal * last_gae_lam
        returns = advantages + value_buffer
        self.advantage_buffer = advantages
        self.return_buffer = returns

        if self.debug:
            self.check_buffers()
