import numpy as np
import scipy.signal as sig
import h5py


class PPOBufferDiscrete:
    """Buffer for full episode for PPO training, and logging."""

    def __init__(self, gamma, lmbda, batch_size, train_length, assay, debug=False):
        self.gamma = gamma
        self.lmbda = lmbda
        self.batch_size = batch_size
        self.trace_length = train_length
        self.pointer = 0
        self.debug = debug
        self.assay = assay
        self.recordings = None

        # Buffer for training
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []
        self.log_action_probability_buffer = []
        self.advantage_buffer = []
        self.return_buffer = []
        self.actor_rnn_state_buffer = []
        self.actor_rnn_state_ref_buffer = []
        self.critic_rnn_state_buffer = []
        self.critic_rnn_state_ref_buffer = []

        self.actor_loss_buffer = []
        self.critic_loss_buffer = []

        if assay:
            self.fish_position_buffer = []
            self.prey_consumed_buffer = []
            self.predator_presence_buffer = []
            self.prey_positions_buffer = []
            self.predator_position_buffer = []
            self.sand_grain_position_buffer = []
            self.vegetation_position_buffer = []
            self.fish_angle_buffer = []

            self.actor_conv1l_buffer = []
            self.actor_conv2l_buffer = []
            self.actor_conv3l_buffer = []
            self.actor_conv4l_buffer = []
            self.actor_conv1r_buffer = []
            self.actor_conv2r_buffer = []
            self.actor_conv3r_buffer = []
            self.actor_conv4r_buffer = []

            self.critic_conv1l_buffer = []
            self.critic_conv2l_buffer = []
            self.critic_conv3l_buffer = []
            self.critic_conv4l_buffer = []
            self.critic_conv1r_buffer = []
            self.critic_conv2r_buffer = []
            self.critic_conv3r_buffer = []
            self.critic_conv4r_buffer = []

    def reset(self):
        # Buffer for training
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []
        self.log_action_probability_buffer = []
        self.actor_rnn_state_buffer = []
        self.actor_rnn_state_ref_buffer = []
        self.critic_rnn_state_buffer = []
        self.critic_rnn_state_ref_buffer = []

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
        self.critic_loss_buffer = []

        if self.assay:
            self.fish_position_buffer = []
            self.prey_consumed_buffer = []
            self.predator_presence_buffer = []
            self.prey_positions_buffer = []
            self.predator_position_buffer = []
            self.sand_grain_position_buffer = []
            self.vegetation_position_buffer = []
            self.fish_angle_buffer = []

            self.actor_conv1l_buffer = []
            self.actor_conv2l_buffer = []
            self.actor_conv3l_buffer = []
            self.actor_conv4l_buffer = []
            self.actor_conv1r_buffer = []
            self.actor_conv2r_buffer = []
            self.actor_conv3r_buffer = []
            self.actor_conv4r_buffer = []

            self.critic_conv1l_buffer = []
            self.critic_conv2l_buffer = []
            self.critic_conv3l_buffer = []
            self.critic_conv4l_buffer = []
            self.critic_conv1r_buffer = []
            self.critic_conv2r_buffer = []
            self.critic_conv3r_buffer = []
            self.critic_conv4r_buffer = []

        self.pointer = 0

    def add_training(self, observation, internal_state, action, reward, value, l_p_action, actor_rnn_state,
                     actor_rnn_state_ref, critic_rnn_state, critic_rnn_state_ref):
        self.observation_buffer.append(observation)
        self.internal_state_buffer.append(internal_state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.log_action_probability_buffer.append(l_p_action)
        self.actor_rnn_state_buffer.append(actor_rnn_state)
        self.actor_rnn_state_ref_buffer.append(actor_rnn_state_ref)
        self.critic_rnn_state_buffer.append(critic_rnn_state)
        self.critic_rnn_state_ref_buffer.append(critic_rnn_state_ref)

    def add_logging(self, mu_i, si_i, mu_a, si_a, mu1, mu1_ref, mu_a1, mu_a_ref):
        # TODO: Redo for discreet
        self.mu_i_buffer.append(mu_i)
        self.si_i_buffer.append(si_i)
        self.mu_a_buffer.append(mu_a)
        self.si_a_buffer.append(si_a)
        self.mu1_buffer.append(mu1)
        self.mu1_ref_buffer.append(mu1_ref)
        self.mu_a1_buffer.append(mu_a1)
        self.mu_a_ref_buffer.append(mu_a_ref)

    def add_loss(self, action_loss, critic_loss):
        self.actor_loss_buffer.append(action_loss)
        self.critic_loss_buffer.append(critic_loss)

    def save_environmental_positions(self, fish_position, prey_consumed, predator_present, prey_positions,
                                     predator_position, sand_grain_positions, vegetation_positions, fish_angle):
        self.fish_position_buffer.append(fish_position)
        self.prey_consumed_buffer.append(prey_consumed)
        self.predator_presence_buffer.append(predator_present)
        self.prey_positions_buffer.append(prey_positions)
        self.predator_position_buffer.append(predator_position)
        self.sand_grain_position_buffer.append(sand_grain_positions)
        self.vegetation_position_buffer.append(vegetation_positions)
        self.fish_angle_buffer.append(fish_angle)

    def save_conv_states(self, actor_conv1l, actor_conv2l, actor_conv3l, actor_conv4l, actor_conv1r, actor_conv2r,
                         actor_conv3r, actor_conv4r,
                         critic_conv1l, critic_conv2l, critic_conv3l, critic_conv4l, critic_conv1r, critic_conv2r,
                         critic_conv3r, critic_conv4r):
        self.actor_conv1l_buffer.append(actor_conv1l)
        self.actor_conv2l_buffer.append(actor_conv2l)
        self.actor_conv3l_buffer.append(actor_conv3l)
        self.actor_conv4l_buffer.append(actor_conv4l)
        self.actor_conv1r_buffer.append(actor_conv1r)
        self.actor_conv2r_buffer.append(actor_conv2r)
        self.actor_conv3r_buffer.append(actor_conv3r)
        self.actor_conv4r_buffer.append(actor_conv4r)
        self.critic_conv1l_buffer.append(critic_conv1l)
        self.critic_conv2l_buffer.append(critic_conv2l)
        self.critic_conv3l_buffer.append(critic_conv3l)
        self.critic_conv4l_buffer.append(critic_conv4l)
        self.critic_conv1r_buffer.append(critic_conv1r)
        self.critic_conv2r_buffer.append(critic_conv2r)
        self.critic_conv3r_buffer.append(critic_conv3r)
        self.critic_conv4r_buffer.append(critic_conv4r)

    def tidy(self):
        self.observation_buffer = np.array(self.observation_buffer)
        self.action_buffer = np.array(self.action_buffer)
        self.reward_buffer = np.array(self.reward_buffer)
        self.value_buffer = np.array(self.value_buffer).flatten()
        self.internal_state_buffer = np.array(self.internal_state_buffer)
        self.log_action_probability_buffer = np.array(self.log_action_probability_buffer)
        self.actor_rnn_state_buffer = np.array(self.actor_rnn_state_buffer)
        self.actor_rnn_state_ref_buffer = np.array(self.actor_rnn_state_ref_buffer)
        self.critic_rnn_state_buffer = np.array(self.critic_rnn_state_buffer)
        self.critic_rnn_state_ref_buffer = np.array(self.critic_rnn_state_ref_buffer)

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
        for slice in slice_steps:
            if slice == slice_steps[-1]:
                observation_slice, internal_state_slice, action_slice, previous_action_slice, \
                log_action_probability_slice, advantage_slice, return_slice = self.get_batch(final_batch=True)
            else:
                observation_slice, internal_state_slice, action_slice, previous_action_slice, \
                log_action_probability_slice, advantage_slice, return_slice = self.get_batch(final_batch=False)

            observation_batch.append(observation_slice)
            internal_state_batch.append(internal_state_slice)
            action_batch.append(action_slice)
            previous_action_batch.append(previous_action_slice)
            log_action_probability_batch.append(log_action_probability_slice)
            advantage_batch.append(advantage_slice)
            return_batch.append(return_slice)

        return np.array(observation_batch), np.array(internal_state_batch), np.array(action_batch), \
            np.array(previous_action_batch), np.array(log_action_probability_batch), np.array(advantage_batch), np.array(return_batch), slice_steps

    def get_batch(self, final_batch):
        """Gets a trace worth of data (or batch, as used previously)"""
        if final_batch:
            observation_slice = self.pad_slice(self.observation_buffer[self.pointer:-1, :], self.trace_length)
            internal_state_slice = self.pad_slice(self.internal_state_buffer[self.pointer:-1, :], self.trace_length)
            action_slice = self.pad_slice(self.action_buffer[self.pointer + 1:-1, :], self.trace_length)
            previous_action_slice = self.pad_slice(self.action_buffer[self.pointer:-2, :], self.trace_length)
            # reward_slice = self.reward_buffer[self.pointer:-1], self.trace_length)
            # value_slice = self.pad_slice(self.value_buffer[self.pointer:-1], self.trace_length)
            log_action_probability_slice = self.pad_slice(self.log_action_probability_buffer[self.pointer:-1], self.trace_length)
            advantage_slice = self.pad_slice(self.advantage_buffer[self.pointer:], self.trace_length)
            return_slice = self.pad_slice(self.return_buffer[self.pointer:], self.trace_length)
            # actor_rnn_state_slice = self.actor_rnn_state_buffer[self.pointer:-1]
            # actor_rnn_state_ref_slice = self.actor_rnn_state_ref_buffer[self.pointer:-1]
            # critic_rnn_state_slice = self.critic_rnn_state_buffer[self.pointer:-1]
            # critic_rnn_state_ref_slice = self.critic_rnn_state_ref_buffer[self.pointer:-1]

        else:
            observation_slice = self.observation_buffer[self.pointer:self.pointer + self.trace_length, :]
            internal_state_slice = self.internal_state_buffer[self.pointer:self.pointer + self.trace_length, :]
            action_slice = self.action_buffer[self.pointer + 1:self.pointer + self.trace_length + 1, :]
            previous_action_slice = self.action_buffer[self.pointer:self.pointer + self.trace_length, :]
            # reward_slice = self.reward_buffer[self.pointer:self.pointer + self.trace_length, ]
            # value_slice = self.value_buffer[self.pointer:self.pointer + self.trace_length, ]
            log_action_probability_slice = self.log_action_probability_buffer[
                                            self.pointer:self.pointer + self.trace_length, :]
            advantage_slice = self.advantage_buffer[self.pointer:self.pointer + self.trace_length]
            return_slice = self.return_buffer[self.pointer:self.pointer + self.trace_length]
            # actor_rnn_state_slice = self.actor_rnn_state_buffer[self.pointer:self.pointer + self.trace_length]
            # actor_rnn_state_ref_slice = self.actor_rnn_state_ref_buffer[self.pointer:self.pointer + self.trace_length]
            # critic_rnn_state_slice = self.critic_rnn_state_buffer[self.pointer:self.pointer + self.trace_length]
            # critic_rnn_state_ref_slice = self.critic_rnn_state_ref_buffer[self.pointer:self.pointer + self.trace_length]

        self.pointer += self.trace_length

        return observation_slice, internal_state_slice, action_slice, previous_action_slice, \
               log_action_probability_slice, advantage_slice, return_slice, \
               #actor_rnn_state_slice, actor_rnn_state_ref_slice, critic_rnn_state_slice, critic_rnn_state_ref_slice

    @staticmethod
    def pad_slice(buffer, desired_length):
        """Zero pads a trace so all are same length"""
        shape_of_data = buffer.shape[1:]
        extra_pads = desired_length - buffer.shape[0]
        padding_shape = (extra_pads, ) + shape_of_data
        padding = np.zeros(padding_shape, dtype=float)
        buffer = np.concatenate((buffer, padding), axis=0)
        return buffer

    def calculate_advantages_and_returns(self, normalise_advantage=True):
        delta = self.reward_buffer[:-1] + self.gamma * self.value_buffer[1:] - self.value_buffer[:-1]
        advantage = self.discount_cumsum(delta, self.gamma * self.lmbda)
        if normalise_advantage:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)
        returns = self.discount_cumsum(self.reward_buffer, self.gamma)[:-1]
        self.advantage_buffer = advantage
        self.return_buffer = returns

        if self.debug:
            self.check_buffers()

    @staticmethod
    def create_data_group(key, data, assay_group):
        # TODO: Compress data.
        try:
            assay_group.create_dataset(key, data=data)
        except RuntimeError:
            del assay_group[key]
            assay_group.create_dataset(key, data=data)

    def save_assay_data(self, assay_id, data_save_location, assay_configuration_id):
        hdf5_file = h5py.File(f"{data_save_location}/{assay_configuration_id}.h5", "a")

        try:
            assay_group = hdf5_file.create_group(assay_id)
        except ValueError:
            assay_group = hdf5_file.get(assay_id)

        # TODO: Save step

        if "observation" in self.recordings:
            self.create_data_group("observation", np.array(self.observation_buffer), assay_group)

        if "rnn state" in self.recordings:
            self.create_data_group("rnn_state_actor", np.array(self.actor_rnn_state_buffer), assay_group)
            self.create_data_group("rnn_state_critic", np.array(self.critic_rnn_state_buffer), assay_group)

        if "environmental positions" in self.recordings:
            self.create_data_group("impulse", np.array(self.action_buffer)[:, 0], assay_group)
            self.create_data_group("angle", np.array(self.action_buffer)[:, 1], assay_group)
            self.create_data_group("fish_position", np.array(self.fish_position_buffer), assay_group)
            self.create_data_group("fish_angle", np.array(self.fish_angle_buffer), assay_group)
            self.create_data_group("consumed", np.array(self.prey_consumed_buffer), assay_group)
            self.predator_presence_buffer = [0 if i is None else 1 for i in self.predator_presence_buffer]
            self.create_data_group("predator_presence", np.array(self.predator_presence_buffer), assay_group)
            self.create_data_group("prey_positions", np.array(self.prey_positions_buffer), assay_group)
            self.create_data_group("predator_positions", np.array(self.predator_position_buffer), assay_group)
            self.create_data_group("sand_grain_positions", np.array(self.sand_grain_position_buffer), assay_group)
            self.create_data_group("vegetation_positions", np.array(self.vegetation_position_buffer), assay_group)

        if "convolutional layers" in self.recordings:
            self.create_data_group("actor_conv1l", np.array(self.actor_conv1l_buffer), assay_group)
            self.create_data_group("actor_conv2l", np.array(self.actor_conv2l_buffer), assay_group)
            self.create_data_group("actor_conv3l", np.array(self.actor_conv3l_buffer), assay_group)
            self.create_data_group("actor_conv4l", np.array(self.actor_conv4l_buffer), assay_group)
            self.create_data_group("actor_conv1r", np.array(self.actor_conv1r_buffer), assay_group)
            self.create_data_group("actor_conv2r", np.array(self.actor_conv2r_buffer), assay_group)
            self.create_data_group("actor_conv3r", np.array(self.actor_conv3r_buffer), assay_group)
            self.create_data_group("actor_conv4r", np.array(self.actor_conv4r_buffer), assay_group)
            self.create_data_group("critic_conv1l", np.array(self.critic_conv1l_buffer), assay_group)
            self.create_data_group("critic_conv2l", np.array(self.critic_conv2l_buffer), assay_group)
            self.create_data_group("critic_conv3l", np.array(self.critic_conv3l_buffer), assay_group)
            self.create_data_group("critic_conv4l", np.array(self.critic_conv4l_buffer), assay_group)
            self.create_data_group("critic_conv1r", np.array(self.critic_conv1r_buffer), assay_group)
            self.create_data_group("critic_conv2r", np.array(self.critic_conv2r_buffer), assay_group)
            self.create_data_group("critic_conv3r", np.array(self.critic_conv3r_buffer), assay_group)
            self.create_data_group("critic_conv4r", np.array(self.critic_conv4r_buffer), assay_group)

        if "reward assessments" in self.recordings:
            self.create_data_group("reward", np.array(self.reward_buffer), assay_group)
            self.create_data_group("advantage", np.array(self.advantage_buffer), assay_group)
            self.create_data_group("value", np.array(self.value_buffer), assay_group)
            self.create_data_group("returns", np.array(self.return_buffer), assay_group)

        hdf5_file.close()

    @staticmethod
    def discount_cumsum(x, discount):
        """
        magic from rllab for computing discounted cumulative sums of vectors.
        input:
            vector x,
            [x0,
             x1,
             x2]
        output:
            [x0 + discount * x1 + discount^2 * x2,
             x1 + discount * x2,
             x2]
        """
        return sig.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def check_buffers(self):
        # Check for NaN
        print("Checking Buffers")
        buffers = [self.advantage_buffer, self.reward_buffer, self.observation_buffer, self.action_buffer,
                   self.return_buffer, self.value_buffer, self.log_action_probability_buffer]
        if np.isnan(np.sum(np.sum(buffer) for buffer in buffers)):
            print("NaN Detected")

        print("Buffers fine")

        # TODO: Add methods for detecting values outside of range.

    def compute_rewards_to_go(self):
        # NOT USED
        rewards_to_go = []
        current_discounted_reward = 0
        for i, reward in enumerate(reversed(self.reward_buffer)):
            current_discounted_reward = reward + current_discounted_reward * self.gamma
            rewards_to_go.insert(0, current_discounted_reward)
        return rewards_to_go

    def compute_advantages(self):
        """
        According to GAE
        """
        # NOT USED
        g = 0
        lmda = 0.95
        returns = []
        for i in reversed(range(1, len(self.reward_buffer))):
            delta = self.reward_buffer[i - 1] + self.gamma * self.value_buffer[i] - self.value_buffer[i - 1]
            g = delta + self.gamma * lmda * g
            returns.append(g + self.value_buffer[i - 1])
        returns.reverse()
        adv = np.array(returns, dtype=np.float32) - self.value_buffer[:-1]
        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-10)
        returns = np.array(returns, dtype=np.float32)
        return returns, adv
