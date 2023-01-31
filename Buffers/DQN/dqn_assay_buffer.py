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

        self.salt_health_buffer = []
        self.rnn_layer_names = []

        self.conv_layer_buffer = []
        self.efference_copy_buffer = []

        # Extra buffers (needed for perfect reloading of states)
        self.prey_orientation_buffer = []
        self.predator_orientation_buffer = []
        self.prey_age_buffer = []
        self.prey_gait_buffer = []


    def reset(self):
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.value_buffer = []

        self.rnn_state_buffer = []
        self.rnn_state_ref_buffer = []

        self.loss_buffer = []
        self.efference_copy_buffer = []

        if self.assay:
            self.fish_position_buffer = []
            self.prey_consumed_buffer = []
            self.predator_presence_buffer = []
            self.prey_positions_buffer = []
            self.predator_position_buffer = []
            self.sand_grain_position_buffer = []
            self.vegetation_position_buffer = []
            self.fish_angle_buffer = []
            self.salt_health_buffer = []

            self.conv_layer_buffer = []

            # Extra buffers (needed for perfect reloading of states)
            self.prey_orientation_buffer = []
            self.predator_orientation_buffer = []
            self.prey_age_buffer = []
            self.prey_gait_buffer = []

    def add_training(self, observation, internal_state, reward, action, rnn_state, rnn_state_ref):
        self.observation_buffer.append(observation)
        self.internal_state_buffer.append(internal_state)
        self.reward_buffer.append(reward)

        self.rnn_state_buffer.append(rnn_state)
        self.rnn_state_ref_buffer.append(rnn_state_ref)

    def save_cnn_data(self, cnn_layers):
        self.conv_layer_buffer.append(cnn_layers)

    def save_environmental_positions(self, action, fish_position, prey_consumed, predator_present, prey_positions,
                                     predator_position, sand_grain_positions, vegetation_positions, fish_angle,
                                     salt_health, efference_copy,
                                     prey_orientation, predator_orientation, prey_age, prey_gait):

        self.action_buffer.append(action)
        self.fish_position_buffer.append(fish_position)
        self.prey_consumed_buffer.append(prey_consumed)
        self.predator_presence_buffer.append(predator_present)
        self.prey_positions_buffer.append(prey_positions)
        self.predator_position_buffer.append(predator_position)
        self.sand_grain_position_buffer.append(sand_grain_positions)
        self.vegetation_position_buffer.append(vegetation_positions)
        self.fish_angle_buffer.append(fish_angle)
        self.salt_health_buffer.append(salt_health)

        self.efference_copy_buffer.append(efference_copy)

        # Extra buffers (needed for perfect reloading of states)
        self.prey_orientation_buffer.append(prey_orientation)
        self.predator_orientation_buffer.append(predator_orientation)
        self.prey_age_buffer.append(prey_age)
        self.prey_gait_buffer.append(prey_gait)

    def create_data_group(self, key, data, assay_group):
        # TODO: Compress data.
        try:
            assay_group.create_dataset(key, data=data)
        except (RuntimeError, OSError) as exception:
            print(f"Failed saving {key}. Attempting to delete existing data.")
            print(exception)
            del assay_group[key]
            self.create_data_group(key, data, assay_group)
            # assay_group.create_dataset(key, data=data)

    def init_assay_recordings(self, recordings, network_recordings):
        self.recordings = recordings
        self.unit_recordings = {i: [] for i in network_recordings}

    def make_desired_recordings(self, network_layers):
        for l in self.unit_recordings.keys():
            self.unit_recordings[l].append(network_layers[l][0])

    def fix_prey_position_buffer(self):
        """Run in the event of prey reproduction to prevent dim errors."""
        # OLD - Creates overlapping array. Complicates analysis too much.
        # new_prey_buffer = []
        # max_prey_num = 0
        # for p in self.prey_positions_buffer:
        #     if np.array(p).shape[0] > max_prey_num:
        #         max_prey_num = np.array(p).shape[0]
        #
        # for i, p in enumerate(self.prey_positions_buffer):
        #     missing_values = max_prey_num - np.array(p).shape[0]
        #     if missing_values > 0:
        #         new_entries = np.array([[15000, 15000] for i in range(missing_values)])
        #         new_prey_buffer.append(np.concatenate((np.array(self.prey_positions_buffer[i]), new_entries), axis=0))
        #     else:
        #         new_prey_buffer.append(np.array(self.prey_positions_buffer[i]))
        #
        # new_prey_buffer = np.array(new_prey_buffer)
        # self.prey_positions_buffer = new_prey_buffer

        # NEW - Has an entire column for each prey that exists at any given time
        # For each step, shift the array of positions across until aligned (should be the min difference with above
        # values).
        num_steps = len(self.prey_positions_buffer)
        num_prey_init = len(self.prey_positions_buffer[0])
        overly_large_position_array = np.ones((num_steps, num_prey_init * 100, 2)) * 10000
        min_index = 0
        total_prey_existing = num_prey_init

        for i, prey_position_slice in enumerate(self.prey_positions_buffer):
            # Ensure one of the arrays is available to accept a new prey.
            overly_large_position_array[i:, total_prey_existing:total_prey_existing + 4] = 1000

            if i == 0:
                overly_large_position_array[i, :num_prey_init] = np.array(self.prey_positions_buffer[i])
            else:
                num_prey = len(self.prey_positions_buffer[i])
                num_prey_previous = len(overly_large_position_array[i - 1])

                prey_position_slice_expanded = np.repeat(np.expand_dims(prey_position_slice, 1), num_prey_previous, 1)
                prey_position_slice_previous_expanded = np.repeat(np.expand_dims(overly_large_position_array[i - 1], 0),
                                                                  num_prey, 0)

                prey_positions_differences = prey_position_slice_expanded - prey_position_slice_previous_expanded
                prey_positions_differences_total = (prey_positions_differences[:, :, 0] ** 2 +
                                                    prey_positions_differences[:, :, 1] ** 2) ** 0.5

                forbidden_index = 0

                for prey in range(prey_positions_differences_total.shape[0]):
                    differences_to_large_array = prey_positions_differences_total[prey]
                    differences_to_large_array[:max([min_index, forbidden_index])] *= 1000
                    order_of_size = np.argsort(differences_to_large_array)
                    forbidden_index = order_of_size[0]
                    if forbidden_index >= total_prey_existing - 1:
                        total_prey_existing += 1
                    overly_large_position_array[i, forbidden_index] = prey_position_slice[prey]
                    forbidden_index += 1

        # Remove columns with only [1000., 1000] or [10000, 10000] (or just both).
        just_1000 = np.sum(
            ((overly_large_position_array[:, :, 0] == 1000.) * (overly_large_position_array[:, :, 1] == 1000.)), axis=0)
        just_10000 = np.sum(
            ((overly_large_position_array[:, :, 0] == 10000.) * (overly_large_position_array[:, :, 1] == 10000.)),
            axis=0)

        whole_just_1000 = (just_1000 == num_steps) * 1
        whole_just_10000 = (just_10000 == num_steps) * 1
        only_both = (just_1000 + just_10000 == num_steps) * 1

        to_delete = whole_just_1000 + whole_just_10000 + only_both
        to_delete = [i for i, d in enumerate(to_delete) if d > 0]

        new_prey_position_array = np.delete(overly_large_position_array, to_delete, axis=1)
        self.prey_positions_buffer = new_prey_position_array

    def organise_conv_recordings(self):
        new_conv_layers = [[] for i in range(8)]
        for timepoint in self.conv_layer_buffer:
            for i in range(8):
                new_conv_layers[i].append(timepoint[i][0])
        for j in range(8):
            new_conv_layers[j] = np.array(new_conv_layers[j])
        self.conv_layer_buffer = new_conv_layers

    def save_assay_data(self, assay_id, data_save_location, assay_configuration_id, internal_state_order,
                        background, salt_location=None):
        hdf5_file = h5py.File(f"{data_save_location}/{assay_configuration_id}.h5", "a")

        try:
            assay_group = hdf5_file.create_group(assay_id)
        except ValueError:
            assay_group = hdf5_file.get(assay_id)

        # self.create_data_group("step", np.array([i for i in range(len(self.observation_buffer))]), assay_group)

        if "observation" in self.recordings:
            self.create_data_group("observation", np.array(self.observation_buffer), assay_group)

        if self.use_dynamic_network:
            for layer in self.unit_recordings.keys():
                self.create_data_group(layer, np.array(self.unit_recordings[layer]).squeeze(), assay_group)
            self.create_data_group("rnn_state_actor", np.array(self.rnn_state_buffer).squeeze(), assay_group)
            self.create_data_group("rnn_state_actor_ref", np.array(self.rnn_state_ref_buffer).squeeze(), assay_group)

            self.internal_state_buffer = np.array(self.internal_state_buffer)
            self.internal_state_buffer = np.reshape(self.internal_state_buffer, (-1, len(internal_state_order)))
            # Get internal state names and save each.
            for i, state in enumerate(internal_state_order):
                self.create_data_group(state, np.array(self.internal_state_buffer[:, i]), assay_group)
                if state == "salt":
                    if salt_location is None:
                        salt_location = [150000, 150000]
                    self.create_data_group("salt_location", np.array(salt_location), assay_group)
                    self.create_data_group("salt_health", np.array(self.salt_health_buffer), assay_group)

            self.efference_copy_buffer = np.array(self.efference_copy_buffer)
            self.create_data_group("efference_copy", self.efference_copy_buffer, assay_group)

        else:
            if "rnn state" in self.unit_recordings:
                self.create_data_group("rnn_state_actor", np.array(self.rnn_state_buffer).squeeze(), assay_group)
                self.create_data_group("rnn_state_actor_ref", np.array(self.rnn_state_ref_buffer).squeeze(), assay_group)

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
                        self.create_data_group("salt_health", np.array(self.salt_health_buffer), assay_group)

                self.efference_copy_buffer = np.array(self.efference_copy_buffer)
                self.create_data_group("efference_copy", self.efference_copy_buffer, assay_group)

            if "convolutional layers" in self.unit_recordings:
                self.organise_conv_recordings()
                self.create_data_group("conv1l", np.array(self.conv_layer_buffer[0]).squeeze(), assay_group)
                self.create_data_group("conv2l", np.array(self.conv_layer_buffer[1]).squeeze(), assay_group)
                self.create_data_group("conv3l", np.array(self.conv_layer_buffer[2]).squeeze(), assay_group)
                self.create_data_group("conv4l", np.array(self.conv_layer_buffer[3]).squeeze(), assay_group)
                self.create_data_group("conv1r", np.array(self.conv_layer_buffer[4]).squeeze(), assay_group)
                self.create_data_group("conv2r", np.array(self.conv_layer_buffer[5]).squeeze(), assay_group)
                self.create_data_group("conv3r", np.array(self.conv_layer_buffer[6]).squeeze(), assay_group)
                self.create_data_group("conv4r", np.array(self.conv_layer_buffer[7]).squeeze(), assay_group)

        if "environmental positions" in self.recordings:
            self.create_data_group("action", np.array(self.action_buffer), assay_group)
            self.create_data_group("impulse", np.array(self.efference_copy_buffer)[:, 0, 1], assay_group)
            self.create_data_group("angle", np.array(self.efference_copy_buffer)[:, 0, 2], assay_group)
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

            self.create_data_group("background", np.array(background), assay_group)

            # Extra buffers (needed for perfect reloading of states)
            self.create_data_group("prey_orientations", np.array(self.prey_orientation_buffer), assay_group)
            self.create_data_group("predator_orientation", np.array(self.predator_orientation_buffer), assay_group)
            self.create_data_group("prey_ages", np.array(self.prey_age_buffer), assay_group)
            self.create_data_group("prey_gaits", np.array(self.prey_gait_buffer), assay_group)

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
