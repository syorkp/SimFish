import h5py
import scipy.signal as sig
import numpy as np

from Buffers.base_buffer import BaseBuffer


class DQNAssayBuffer(BaseBuffer):
    """
    Class to save assay data for DQN.

    NOTE: Class is NOT integrated with experience buffer as is the case with PPO.
    """

    def __init__(self):
        super().__init__()

        self.loss_buffer = []
        self.assay = True  # This class is always for assay mode

    def reset(self):
        self._reset()

        self.loss_buffer = []

    def add_training(self, observation, internal_state, reward, action, rnn_state, rnn_state_ref):
        self._add_training(observation, internal_state, reward, rnn_state, rnn_state_ref)

    def save_environmental_positions(self, action, fish_position, prey_consumed, predator_present, prey_positions,
                                     predator_position, sand_grain_positions, fish_angle,
                                     salt_health, efference_copy,
                                     prey_orientation=None, predator_orientation=None, prey_age=None, prey_gait=None):
        self._save_environmental_positions(fish_position, prey_consumed, predator_present, prey_positions,
                                      predator_position, sand_grain_positions, fish_angle, salt_health, efference_copy,prey_orientation, predator_orientation, prey_age, prey_gait)

        self.action_buffer.append(action)

    def save_assay_data(self, assay_id, data_save_location, assay_configuration_id, internal_state_order,
                        sediment, salt_location=None):
        hdf5_file, assay_group = self._save_assay_data(data_save_location, assay_configuration_id, assay_id, sediment, internal_state_order, salt_location)

        self.create_data_group("action", np.array(self.action_buffer), assay_group)
        self.efference_copy_buffer = np.squeeze(np.array(self.efference_copy_buffer))

        self.create_data_group("impulse", self.efference_copy_buffer[:, 1], assay_group)
        self.create_data_group("angle", self.efference_copy_buffer[:, 2], assay_group)

        print(f"{assay_id} Data Saved")
        hdf5_file.close()