import numpy as np

import h5py


class BaseBuffer:

    def __init__(self):
        self.assay = False
        self.recordings = None
        self.rnn_layer_names = []

        # Buffer for training
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.advantage_buffer = []
        self.value_buffer = []
        self.advantage_buffer_ref = []
        self.value_buffer_ref = []
        self.return_buffer = []
        self.rnn_state_buffer = []
        self.rnn_state_ref_buffer = []
        self.efference_copy_buffer = []

        # Environment log buffer
        self.fish_position_buffer = []
        self.prey_consumed_buffer = []
        self.predator_presence_buffer = []
        self.prey_positions_buffer = []
        self.predator_position_buffer = []
        self.sand_grain_position_buffer = []
        self.fish_angle_buffer = []
        self.salt_health_buffer = []

        # Extra buffers (needed for perfect reloading of states)
        self.prey_identifiers_buffer = []
        self.prey_orientation_buffer = []
        self.predator_orientation_buffer = []
        self.prey_age_buffer = []
        self.prey_gait_buffer = []
        self.switch_step = None  # Tracking switch step

    def _reset(self):
        """Reset all buffers"""

        # Buffer for training
        self.action_buffer = []
        self.observation_buffer = []
        self.reward_buffer = []
        self.internal_state_buffer = []
        self.advantage_buffer = []
        self.value_buffer = []
        self.advantage_buffer_ref = []
        self.value_buffer_ref = []
        self.return_buffer = []
        self.rnn_state_buffer = []
        self.rnn_state_ref_buffer = []
        self.efference_copy_buffer = []

        # Environment log buffer
        self.fish_position_buffer = []
        self.prey_consumed_buffer = []
        self.predator_presence_buffer = []
        self.prey_positions_buffer = []
        self.predator_position_buffer = []
        self.sand_grain_position_buffer = []
        self.fish_angle_buffer = []
        self.salt_health_buffer = []

        # Extra buffers (needed for perfect reloading of states)
        self.prey_identifiers_buffer = []
        self.prey_orientation_buffer = []
        self.predator_orientation_buffer = []
        self.prey_age_buffer = []
        self.prey_gait_buffer = []
        self.switch_step = None  # Tracking switch step

    def _add_training(self, observation, internal_state, reward, rnn_state, rnn_state_ref, value, value_ref,
                      advantage, advantage_ref):
        self.observation_buffer.append(observation)
        self.internal_state_buffer.append(internal_state)
        self.reward_buffer.append(reward)
        self.value_buffer.append(value)
        self.value_buffer_ref.append(value_ref)
        self.advantage_buffer.append(advantage)
        self.advantage_buffer_ref.append(advantage_ref)

        self.rnn_state_buffer.append(rnn_state)
        self.rnn_state_ref_buffer.append(rnn_state_ref)

    def _save_environmental_positions(self, fish_position, prey_consumed, predator_present, prey_positions,
                                      predator_position, sand_grain_positions, fish_angle, salt_health, efference_copy,
                                      prey_orientation, predator_orientation, prey_age, prey_gait, prey_identifiers):
        self.fish_position_buffer.append(fish_position)
        self.prey_consumed_buffer.append(prey_consumed)
        self.predator_presence_buffer.append(predator_present)
        self.predator_position_buffer.append(predator_position)
        self.sand_grain_position_buffer.append(sand_grain_positions)
        self.fish_angle_buffer.append(fish_angle)
        self.salt_health_buffer.append(salt_health)
        self.efference_copy_buffer.append(efference_copy)

        # Buffers that can change shape.
        self.prey_identifiers_buffer.append(prey_identifiers)
        self.prey_positions_buffer.append(prey_positions)

        # Extra buffers (needed for perfect reloading of states)
        self.prey_orientation_buffer.append(prey_orientation)
        self.predator_orientation_buffer.append(predator_orientation)
        self.prey_age_buffer.append(prey_age)
        self.prey_gait_buffer.append(prey_gait)

    def fix_prey_buffers(self):
        """Uses the prey identifiers buffer to reformat the prey buffers (which are likely jagged), so that single
        columns correspond to individual prey."""
        num_steps = len(self.prey_identifiers_buffer)
        total_distinct_prey = int(max([item for sublist in self.prey_identifiers_buffer for item in sublist]) + 1)   #[max(step) for step in self.prey_identifiers_buffer]) + 1)

        new_prey_position_buffer = np.ones((num_steps, total_distinct_prey, 2)) * 15000
        new_prey_orientation_buffer = np.ones((num_steps, total_distinct_prey)) * 15000
        new_prey_age_buffer = np.ones((num_steps, total_distinct_prey)) * 15000
        new_prey_gait_buffer = np.ones((num_steps, total_distinct_prey)) * 15000
        new_prey_identity_buffer = np.ones((num_steps, total_distinct_prey)) * 15000

        for step, prey_identities in enumerate(self.prey_identifiers_buffer):
            for p, prey_i in enumerate(prey_identities):
                new_prey_position_buffer[step, prey_i, :] = self.prey_positions_buffer[step][p]
                new_prey_orientation_buffer[step, prey_i] = self.prey_orientation_buffer[step][p]
                new_prey_age_buffer[step, prey_i] = self.prey_age_buffer[step][p]
                new_prey_gait_buffer[step, prey_i] = self.prey_gait_buffer[step][p]
                new_prey_identity_buffer[step, prey_i] = self.prey_identifiers_buffer[step][p]

        self.prey_positions_buffer = new_prey_position_buffer
        self.prey_orientation_buffer = new_prey_orientation_buffer
        self.prey_age_buffer = new_prey_age_buffer
        self.prey_gait_buffer = new_prey_gait_buffer
        self.prey_identifiers_buffer = new_prey_identity_buffer.astype(int)

    def create_data_group(self, key, data, assay_group):
        try:
            assay_group.create_dataset(key, data=data)
        except (RuntimeError, OSError):
            # In the event to dataset already exists, delete it and run again.
            del assay_group[key]
            self.create_data_group(key, data, assay_group)

    def _save_assay_data(self, data_save_location, assay_configuration_id, assay_id, sediment, internal_state_order,
                         salt_location):
        hdf5_file = h5py.File(f"{data_save_location}/{assay_configuration_id}.h5", "a")

        try:
            assay_group = hdf5_file.create_group(assay_id)
        except ValueError:
            assay_group = hdf5_file.get(assay_id)

        self.create_data_group("observation", np.array(self.observation_buffer), assay_group)

        try:
            rnn_buffer = np.array(self.rnn_state_buffer).squeeze()
            rnn_ref_buffer = np.array(self.rnn_state_ref_buffer).squeeze()

            self.create_data_group("rnn_state", rnn_buffer, assay_group)
            self.create_data_group("rnn_state_ref", rnn_ref_buffer, assay_group)
        except TypeError:
            # Fixes jagged array error that occurs in split assay mode.
            for i, rnn in enumerate(self.rnn_state_buffer):
                if np.array(rnn).shape[0] == 2:
                    self.rnn_state_buffer[i] = np.array([np.array(rnn)]).astype(np.float64)
                else:
                    self.rnn_state_buffer[i] = np.array(rnn).astype(np.float64)

            for i, rnn in enumerate(self.rnn_state_ref_buffer):
                if np.array(rnn).shape[0] == 2:
                    self.rnn_state_ref_buffer[i] = np.array([np.array(rnn)]).astype(np.float64)
                else:
                    self.rnn_state_ref_buffer[i] = np.array(rnn).astype(np.float64)

            self.create_data_group("rnn_state", self.rnn_state_buffer, assay_group)
            self.create_data_group("rnn_state_ref", self.rnn_state_ref_buffer, assay_group)

        self.internal_state_buffer = np.array(self.internal_state_buffer)

        self.internal_state_buffer = np.reshape(self.internal_state_buffer, (-1, len(internal_state_order)))

        self.create_data_group("internal_state", np.array(self.internal_state_buffer), assay_group)

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

        self.create_data_group("fish_position", np.array(self.fish_position_buffer), assay_group)
        self.create_data_group("fish_angle", np.array(self.fish_angle_buffer), assay_group)
        self.create_data_group("consumed", np.array(self.prey_consumed_buffer), assay_group)

        self.predator_presence_buffer = [0 if i is None else 1 for i in self.predator_presence_buffer]
        self.create_data_group("predator_presence", np.array(self.predator_presence_buffer), assay_group)
        self.create_data_group("predator_positions", np.array(self.predator_position_buffer), assay_group)
        self.create_data_group("sand_grain_positions", np.array(self.sand_grain_position_buffer), assay_group)

        if len([item for sublist in self.prey_identifiers_buffer for item in sublist]) > 0:
            self.fix_prey_buffers()
        self.create_data_group("prey_positions", np.array(self.prey_positions_buffer), assay_group)

        self.create_data_group("sediment", np.array(sediment), assay_group)

        if self.switch_step != None:
            self.create_data_group("switch_step", np.array([self.switch_step]), assay_group)

        # Extra buffers (needed for perfect reloading of states)
        if self.assay:
            self.create_data_group("prey_identifiers", self.prey_identifiers_buffer, assay_group)
            self.create_data_group("prey_orientations", np.array(self.prey_orientation_buffer), assay_group)
            self.create_data_group("predator_orientation", np.array(self.predator_orientation_buffer), assay_group)
            self.create_data_group("prey_ages", np.array(self.prey_age_buffer), assay_group)
            self.create_data_group("prey_gaits", np.array(self.prey_gait_buffer), assay_group)

        self.create_data_group("reward", np.array(self.reward_buffer), assay_group)
        self.create_data_group("advantage", np.array(self.advantage_buffer), assay_group)
        self.create_data_group("value", np.array(self.value_buffer), assay_group)
        self.create_data_group("advantage_ref", np.array(self.advantage_buffer_ref), assay_group)
        self.create_data_group("value_ref", np.array(self.value_buffer_ref), assay_group)
        self.create_data_group("returns", np.array(self.return_buffer), assay_group)

        return hdf5_file, assay_group

    def pad_buffer(self, buffer):
        max_dim = 0
        for b in buffer:
            if len(b) > max_dim:
                max_dim = len(b)
        for b in buffer:
            b = list(b)
            if len(b) < max_dim:
                b.append(0)
        return buffer
