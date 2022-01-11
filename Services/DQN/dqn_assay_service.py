import json
import h5py
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf

from Services.assay_service import AssayService
from Services.DQN.base_dqn import BaseDQN
from Tools.make_gif import make_gif

tf.logging.set_verbosity(tf.logging.ERROR)


def assay_target(trial, total_steps, episode_number, memory_fraction):
    service = DQNAssayService(model_name=trial["Model Name"],
                              trial_number=trial["Trial Number"],
                              total_steps=total_steps,
                              episode_number=episode_number,
                              monitor_gpu=trial["monitor gpu"],
                              using_gpu=trial["Using GPU"],
                              memory_fraction=memory_fraction,
                              config_name=trial["Environment Name"],
                              realistic_bouts=trial["Realistic Bouts"],
                              continuous_actions=trial["Continuous Actions"],

                              assays=trial["Assays"],
                              set_random_seed=trial["set random seed"],
                              assay_config_name=trial["Assay Configuration Name"],
                              )

    service.run()


class DQNAssayService(AssayService, BaseDQN):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, assays, set_random_seed, assay_config_name):
        """
        Runs a set of assays provided by the run configuraiton.
        """

        super().__init__(model_name=model_name, trial_number=trial_number,
                         total_steps=total_steps, episode_number=episode_number,
                         monitor_gpu=monitor_gpu, using_gpu=using_gpu,
                         memory_fraction=memory_fraction, config_name=config_name,
                         realistic_bouts=realistic_bouts,
                         continuous_environment=continuous_actions, assays=assays, set_random_seed=set_random_seed,
                         assay_config_name=assay_config_name)

        # Hacky fix for h5py problem:
        self.last_position_dim = self.environment_params["prey_num"]
        self.stimuli_data = []
        self.output_data = {}

    def _run(self):
        self.create_network()
        self.init_states()
        AssayService._run(self)

    def perform_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["recordings"]}

        rnn_state = (
            np.zeros([1, self.main_QN.rnn_dim]),
            np.zeros([1, self.main_QN.rnn_dim]))
        self.assay_output_data_format = {key: None for key in assay["recordings"]}

        self.simulation.reset()

        sa = np.zeros((1, 128))

        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=3,
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=True,
                                                                                     activations=(sa,))
        a = 0
        self.step_number = 0
        while self.step_number < assay["duration"]:
            if assay["reset"] and self.step_number % assay["reset interval"] == 0:
                rnn_state = (
                np.zeros([1, self.main_QN.rnn_dim]), np.zeros([1, self.main_QN.rnn_dim]))  # Reset RNN hidden state
            self.step_number += 1

            o, a, r, internal_state, o1, d, rnn_state = self.step_loop(o=o, internal_state=internal_state,
                                                                       a=a, rnn_state=rnn_state)
            o = o1

            if d:
                break

    def step_loop(self, o, internal_state, a, rnn_state):
        return BaseDQN._assay_step_loop(self, o, internal_state, a, rnn_state)

    def save_hdf5_data(self, assay):
        if assay["save frames"]:
            make_gif(self.frame_buffer,
                     f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        self.frame_buffer = []

        # absolute_path = '/home/sam/PycharmProjects/SimFish/Assay-Output/new_differential_prey_ref-3' + f'/{self.assay_configuration_id}.h5'
        # hdf5_file = h5py.File(absolute_path, "a")
        hdf5_file = h5py.File(f"{self.data_save_location}/{self.assay_configuration_id}.h5", "a")

        try:
            assay_group = hdf5_file.create_group(assay['assay id'])
        except ValueError:
            assay_group = hdf5_file.get(assay['assay id'])

        if "prey_positions" in self.assay_output_data_format.keys():
            self.output_data["prey_positions"] = np.stack(self.output_data["prey_positions"])

        for key in self.output_data:
            try:
                # print(self.output_data[key])
                assay_group.create_dataset(key, data=np.array(self.output_data[key]))  # TODO: Compress data.
            except RuntimeError:
                del assay_group[key]
                assay_group.create_dataset(key, data=np.array(self.output_data[key]))  # TODO: Compress data.
        hdf5_file.close()

    def save_episode_data(self):
        self.episode_summary_data = {
            "Prey Caught": self.simulation.prey_caught,
            "Predators Avoided": self.simulation.predator_attacks_avoided,
            "Sand Grains Bumped": self.simulation.sand_grains_bumped,
            "Steps Near Vegetation": self.simulation.steps_near_vegetation
        }
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-summary_data.json", "w") as output_file:
            json.dump(self.episode_summary_data, output_file)
        self.episode_summary_data = None

    def package_output_data(self, observation, rev_observation, action, advantage_stream, rnn_state, rnn2_state,
                            position, prey_consumed, predator_body,
                            conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r,
                            prey_positions, predator_position, sand_grain_positions, vegetation_positions, fish_angle):
        """

        :param action:
        :param advantage_stream:
        :param rnn_state:
        :param position:
        :param prey_consumed:
        :param predator_body: A boolean to say whether consumed this step.
        :param conv1l:
        :param conv2l:
        :param conv3l:
        :param conv4l:
        :param conv1r:
        :param conv2r:
        :param conv3r:
        :param conv4r:
        :param prey_positions:
        :param predator_position:
        :param sand_grain_positions:
        :param vegetation_positions:
        :return:
        """
        # Make output data JSON serializable
        action = int(action)
        advantage_stream = advantage_stream.tolist()
        rnn_state = rnn_state.c.tolist()
        position = list(position)
        # observation = observation.tolist()

        data = {
            "behavioural choice": action,
            "rnn state": rnn_state,
            "rnn 2 state": rnn2_state,
            "advantage stream": advantage_stream,
            "position": position,
            "observation": observation,
            "rev_observation": rev_observation,
            "left_conv_1": conv1l,
            "left_conv_2": conv2l,
            "left_conv_3": conv3l,
            "left_conv_4": conv4l,
            "right_conv_1": conv1r,
            "right_conv_2": conv2r,
            "right_conv_3": conv3r,
            "right_conv_4": conv4r,
            "prey_positions": prey_positions,
            "predator_position": predator_position,
            "sand_grain_positions": sand_grain_positions,
            "vegetation_positions": vegetation_positions,
            "fish_angle": fish_angle,
            "hunger": self.simulation.fish.hungry,
            "stress": self.simulation.fish.stress,
        }

        if prey_consumed:
            data["consumed"] = 1
        else:
            data["consumed"] = 0
        if predator_body is not None:
            data["predator"] = 1
        else:
            data["predator"] = 0

        stimuli = self.simulation.stimuli_information
        to_save = {}
        for stimulus in stimuli.keys():
            if stimuli[stimulus]:
                to_save[stimulus] = stimuli[stimulus]

        if to_save:
            self.stimuli_data.append(to_save)

        return data

    def make_recordings(self, available_data):
        """No longer used - saves data in JSON"""

        step_data = {i: available_data[i] for i in self.assay_output_data_format}
        for d_type in step_data:
            self.assay_output_data[d_type].append(available_data[d_type])
        step_data["step"] = self.step_number
        self.assay_output_data.append(step_data)

    def save_assay_results(self, assay):
        """No longer used - saves data in JSON"""
        # Saves all the information from the assays in JSON format.
        if assay["save frames"]:
            make_gif(self.frame_buffer, f"{self.data_save_location}/{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'],
                     true_image=True)

        self.frame_buffer = []
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)
