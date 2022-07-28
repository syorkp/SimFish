import h5py
import scipy.signal as sig
import numpy as np


class DQNAssayBuffer:
    """
    Class to save assay data for DQN.

    NOTE: Class is NOT integrated with experience buffer as is the case with PPO.
    """

    def __init__(self, use_dynamic_network=False):
        self.assay = True
        self.recordings = None
        self.use_dynamic_network = use_dynamic_network

        # Buffer for training
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.advantage_buffer = []
        self.return_buffer = []
        self.rnn_state_buffer = []
        self.rnn_state_ref_buffer = []

        self.loss_buffer = []

        self.fish_position_buffer = []
        self.prey_consumed_buffer = []
        self.predator_presence_buffer = []
        self.prey_positions_buffer = []
        self.predator_position_buffer = []
        self.sand_grain_position_buffer = []
        self.vegetation_position_buffer = []
        self.fish_angle_buffer = []

        self.rnn_layer_names = []

    def reset(self):
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []

        self.rnn_state_buffer = []
        self.rnn_state_ref_buffer = []

        self.loss_buffer = []

        if self.assay:
            self.fish_position_buffer = []
            self.prey_consumed_buffer = []
            self.predator_presence_buffer = []
            self.prey_positions_buffer = []
            self.predator_position_buffer = []
            self.sand_grain_position_buffer = []
            self.vegetation_position_buffer = []
            self.fish_angle_buffer = []

    def add_training(self, observation, internal_state, reward, action, rnn_state, rnn_state_ref):
        self.observation_buffer.append(observation)
        self.internal_state_buffer.append(internal_state)
        self.reward_buffer.append(reward)

        self.rnn_state_buffer.append(rnn_state)
        self.rnn_state_ref_buffer.append(rnn_state_ref)

    def save_environmental_positions(self, action, fish_position, prey_consumed, predator_present, prey_positions,
                                     predator_position, sand_grain_positions, vegetation_positions, fish_angle):
        self.action_buffer.append(action)
        self.fish_position_buffer.append(fish_position)
        self.prey_consumed_buffer.append(prey_consumed)
        self.predator_presence_buffer.append(predator_present)
        self.prey_positions_buffer.append(prey_positions)
        self.predator_position_buffer.append(predator_position)
        self.sand_grain_position_buffer.append(sand_grain_positions)
        self.vegetation_position_buffer.append(vegetation_positions)
        self.fish_angle_buffer.append(fish_angle)

    @staticmethod
    def create_data_group(key, data, assay_group):
        # TODO: Compress data.
        try:
            assay_group.create_dataset(key, data=data)
        except (RuntimeError, OSError) as exception:
            del assay_group[key]
            assay_group.create_dataset(key, data=data)

    def init_assay_recordings(self, recordings, network_recordings):
        self.recordings = recordings
        self.unit_recordings = {i: [] for i in network_recordings}

    def make_desired_recordings(self, network_layers):
        for l in self.unit_recordings.keys():
            self.unit_recordings[l].append(network_layers[l][0])

    def fix_prey_position_buffer(self):
        """Run in the event of prey reproduction to prevent dim errors."""
        new_prey_buffer = []
        number_to_add_each_step = 0
        max_prey_num = 0
        for p in self.prey_positions_buffer:
            if np.array(p).shape[0] > max_prey_num:
                max_prey_num = np.array(p).shape[0]

        for i, p in enumerate(self.prey_positions_buffer):
            missing_values = max_prey_num - np.array(p).shape[0]
            new_entries = np.array([[15000, 15000] for i in range(missing_values)])
            new_prey_buffer.append(np.concatenate((np.array(self.prey_positions_buffer[i]), new_entries), axis=0))

        new_prey_buffer = np.array(new_prey_buffer)
        print(new_prey_buffer.shape)
        return new_prey_buffer

    def save_assay_data(self, assay_id, data_save_location, assay_configuration_id, internal_state_order, salt_location=None):
        hdf5_file = h5py.File(f"{data_save_location}/{assay_configuration_id}.h5", "a")

        try:
            assay_group = hdf5_file.create_group(assay_id)
        except ValueError:
            assay_group = hdf5_file.get(assay_id)

        # self.create_data_group("step", np.array([i for i in range(len(self.observation_buffer))]), assay_group)

        if "observation" in self.recordings:
            self.create_data_group("observation", np.array(self.observation_buffer), assay_group)

        if "rnn state" in self.unit_recordings:
            self.create_data_group("rnn_state_actor", np.array(self.rnn_state_buffer), assay_group)

        if self.use_dynamic_network:
            for layer in self.unit_recordings.keys():
                self.create_data_group(layer, np.array(self.unit_recordings[layer]), assay_group)

        if "internal state" in self.unit_recordings:
            self.internal_state_buffer = np.array(self.internal_state_buffer)
            self.internal_state_buffer = np.reshape(self.internal_state_buffer, (-1, len(internal_state_order)))
            # Get internal state names and save each.
            for i, state in enumerate(internal_state_order):
                self.create_data_group(state, np.array(self.internal_state_buffer[:, i]), assay_group)
                if state == "salt":
                    if salt_location is None:
                        salt_location = [150000, 150000]
                    self.create_data_group("salt_location", np.array(salt_location), assay_group)

        if "environmental positions" in self.recordings:
            self.create_data_group("action", np.array(self.action_buffer), assay_group)
            self.create_data_group("fish_position", np.array(self.fish_position_buffer), assay_group)
            self.create_data_group("fish_angle", np.array(self.fish_angle_buffer), assay_group)
            self.create_data_group("consumed", np.array(self.prey_consumed_buffer), assay_group)
            self.predator_presence_buffer = [0 if i is None else 1 for i in self.predator_presence_buffer]
            self.create_data_group("predator_presence", np.array(self.predator_presence_buffer), assay_group)

            try:
                self.create_data_group("prey_positions", np.array(self.prey_positions_buffer), assay_group)
            except:
                self.fix_prey_position_buffer()
                self.create_data_group("prey_positions", np.array(self.prey_positions_buffer), assay_group)

            self.create_data_group("predator_positions", np.array(self.predator_position_buffer), assay_group)

            self.create_data_group("sand_grain_positions", np.array(self.sand_grain_position_buffer), assay_group)
            self.create_data_group("vegetation_positions", np.array(self.vegetation_position_buffer), assay_group)

        if "convolutional layers" in self.unit_recordings:
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

        return hdf5_file, assay_group

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
