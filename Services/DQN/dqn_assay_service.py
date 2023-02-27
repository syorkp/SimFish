import json
import h5py
from datetime import datetime
import copy

import numpy as np
# noinspection PyUnresolvedReferences
import tensorflow.compat.v1 as tf

from Buffers.DQN.dqn_assay_buffer import DQNAssayBuffer
from Environment.Action_Space.draw_angle_dist import get_modal_impulse_and_angle
from Services.assay_service import AssayService
from Services.DQN.base_dqn import BaseDQN
from Analysis.Video.behaviour_video_construction import draw_episode
from Analysis.load_data import load_data

tf.logging.set_verbosity(tf.logging.ERROR)


def assay_target(trial, total_steps, episode_number, memory_fraction):
    if "monitor gpu" in trial:
        monitor_gpu = trial["monitor gpu"]
    else:
        monitor_gpu = False

    if "Using GPU" in trial:
        using_gpu = trial["Using GPU"]
    else:
        using_gpu = True

    if "Continuous Actions" in trial:
        continuous_actions = trial["Continuous Actions"]
    else:
        continuous_actions = False

    if "Realistic Bouts" in trial:
        realistic_bouts = trial["Realistic Bouts"]
    else:
        realistic_bouts = True

    if "Full Reafference" in trial:
        full_reafference = trial["Full Reafference"]
    else:
        full_reafference = True

    if "Checkpoint" in trial:
        checkpoint = trial["Checkpoint"]
    else:
        checkpoint = None

    if "interventions" in trial:
        interventions = trial["interventions"]
    else:
        interventions = None

    if "set random seed" in trial:
        set_random_seed = trial["set random seed"]
    else:
        set_random_seed = False

    if "behavioural recordings" in trial:
        behavioural_recordings = trial["behavioural recordings"]
    else:
        behavioural_recordings = None

    if "network recordings" in trial:
        network_recordings = trial["network recordings"]
    else:
        network_recordings = None

    # Handling for when using split assay version.
    modification = None
    split_event = None
    if trial["Run Mode"] == "Split-Assay":
        split_event = trial["Split Event"]
        if "Run Index" not in trial:
            run_version = "Original"
        else:
            modification = trial["Modification"]
            run_version = trial["Run Index"]
    else:
        run_version = None

    service = DQNAssayService(model_name=trial["Model Name"],
                              trial_number=trial["Trial Number"],
                              total_steps=total_steps,
                              episode_number=episode_number,
                              monitor_gpu=monitor_gpu,
                              using_gpu=using_gpu,
                              memory_fraction=memory_fraction,
                              config_name=trial["Environment Name"],
                              realistic_bouts=realistic_bouts,
                              continuous_actions=continuous_actions,
                              assays=trial["Assays"],
                              set_random_seed=set_random_seed,
                              assay_config_name=trial["Assay Configuration Name"],
                              checkpoint=checkpoint,
                              full_reafference=full_reafference,
                              behavioural_recordings=behavioural_recordings,
                              network_recordings=network_recordings,
                              interventions=interventions,
                              run_version=run_version,
                              split_event=split_event,
                              modification=modification
                              )

    service.run()


class DQNAssayService(AssayService, BaseDQN):

    def __init__(self, model_name, trial_number, total_steps, episode_number, monitor_gpu, using_gpu, memory_fraction,
                 config_name, realistic_bouts, continuous_actions, assays, set_random_seed,
                 assay_config_name, checkpoint, full_reafference, behavioural_recordings, network_recordings,
                 interventions, run_version, split_event, modification):
        """
        Runs a set of assays provided by the run configuration.
        """

        super().__init__(model_name=model_name, trial_number=trial_number,
                         total_steps=total_steps, episode_number=episode_number,
                         monitor_gpu=monitor_gpu, using_gpu=using_gpu,
                         memory_fraction=memory_fraction, config_name=config_name,
                         realistic_bouts=realistic_bouts,
                         continuous_environment=continuous_actions, assays=assays,
                         set_random_seed=set_random_seed,
                         assay_config_name=assay_config_name,
                         checkpoint=checkpoint,
                         behavioural_recordings=behavioural_recordings,
                         network_recordings=network_recordings,
                         interventions=interventions,
                         run_version=run_version,
                         split_event=split_event,
                         modification=modification
                         )

        # Hacky fix for h5py problem:
        self.last_position_dim = self.environment_params["prey_num"]
        self.stimuli_data = []
        self.output_data = {}

        self.buffer = DQNAssayBuffer()

        self.full_reafference = full_reafference

    def run(self):
        sess = self.create_session()
        with sess as self.sess:
            self.create_network()
            self.init_states()
            AssayService._run(self)

    def perform_assay(self, assay, background=None, energy_state=None):
        # self.assay_output_data_format = {key: None for key in assay["recordings"]}
        # self.buffer.init_assay_recordings(assay["behavioural recordings"], assay["network recordings"])

        if self.rnn_input is not None:
            rnn_state = copy.copy(self.rnn_input[0])
            rnn_state_ref = copy.copy(self.rnn_input[1])
        else:
            rnn_state = copy.copy(self.init_rnn_state)
            rnn_state_ref = copy.copy(self.init_rnn_state_ref)

        if self.run_version == "Original-Completion" or self.run_version == "Modified-Completion":
            print("Loading Simulation")
            o = self.simulation.load_simulation(self.buffer, background, energy_state)
            internal_state = self.buffer.internal_state_buffer[-1]
            a = self.buffer.action_buffer[-1]

            a = np.array([a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle])

            if self.run_version == "Modified-Completion":
                self.simulation.make_modification()

            a, updated_rnn_state, rnn2_state, network_layers, sa, sv = \
                self.sess.run(
                    [self.main_QN.predict, self.main_QN.rnn_state_shared, self.main_QN.rnn_state_ref,
                     self.main_QN.network_graph,

                     self.main_QN.streamA,
                     self.main_QN.streamV,
                     ],
                    feed_dict={self.main_QN.observation: o,
                               self.main_QN.internal_state: internal_state,
                               self.main_QN.prev_actions: np.expand_dims(a, 0),
                               self.main_QN.train_length: 1,
                               self.main_QN.rnn_state_in: rnn_state,
                               self.main_QN.rnn_state_in_ref: rnn_state_ref,

                               self.main_QN.batch_size: 1,
                               self.main_QN.exp_keep: 1.0,
                               # self.main_QN.learning_rate: self.learning_params["learning_rate"],
                               })

            a = a[0]
            self.step_number = len(self.buffer.internal_state_buffer)
        else:
            self.simulation.reset()
            a = 0
            self.step_number = 0

        sa = np.zeros((1, 128))

        o, r, internal_state, d, FOV = self.simulation.simulation_step(action=a, activations=(sa,))

        if self.full_reafference:
            action_reafference = [[a, self.simulation.fish.prev_action_impulse, self.simulation.fish.prev_action_angle]]
        else:
            action_reafference = [a]

        while self.step_number < assay["duration"]:
            # if assay["reset"] and self.step_number % assay["reset interval"] == 0:
            #     rnn_state = (
            #         np.zeros([1, self.main_QN.rnn_dim]), np.zeros([1, self.main_QN.rnn_dim]))  # Reset RNN hidden state
            self.step_number += 1

            # Deal with interventions
            if self.interruptions:
                if self.visual_interruptions is not None:
                    if self.visual_interruptions[self.step_number] == 1:
                        # mean values over all data
                        o[:, 0, :] = 4
                        o[:, 1, :] = 11
                        o[:, 2, :] = 16
                if self.reafference_interruptions is not None:
                    if self.reafference_interruptions[self.step_number] is not False:
                        action = self.reafference_interruptions[self.step_number]
                        if self.full_reafference:
                            i, a = get_modal_impulse_and_angle(action)
                            action_reafference = [[action, i, a]]
                if self.preset_energy_state is not None:
                    if self.preset_energy_state[self.step_number] is not False:
                        self.simulation.fish.energy_level = self.preset_energy_state[self.step_number]
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("energy_state")
                        internal_state[0, index] = self.preset_energy_state[self.step_number]
                if self.in_light_interruptions is not None:
                    if self.in_light_interruptions[self.step_number] == 1:
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("in_light")
                        internal_state[0, index] = self.in_light_interruptions[self.step_number]
                if self.salt_interruptions is not None:
                    if self.salt_interruptions[self.step_number] == 1:
                        internal_state_order = self.get_internal_state_order()
                        index = internal_state_order.index("salt")
                        internal_state[0, index] = self.salt_interruptions[self.step_number]
            self.previous_action = a

            o, a, r, internal_state, o1, d, rnn_state = self.step_loop(o=o,
                                                                       internal_state=internal_state,
                                                                       a=action_reafference,
                                                                       rnn_state=rnn_state,
                                                                       rnn_state_ref=rnn_state_ref)
            o = o1

            if d:
                if self.run_version == "Original":
                    if self.simulation.switch_step != None:
                        self.buffer.switch_step = self.simulation.switch_step
                    else:
                        # If no split occurs, return without saving data.
                        print("No split occurred, as condition never met. Returning without saving data.")
                        return False

                break
            if self.full_reafference:
                # As simulation step returns full action
                action_reafference = [a]
            else:
                action_reafference = [a]

        if self.environment_params["salt"]:
            salt_location = self.simulation.salt_location
        else:
            salt_location = None

        if self.using_gpu:
            background = self.simulation.board.global_background_grating.get()[:, :, 0]
        else:
            background = self.simulation.board.global_background_grating[:, :, 0]
        self.buffer.save_assay_data(assay['assay id'], self.data_save_location, self.assay_configuration_id,
                                    self.internal_state_order, background=background,
                                    salt_location=salt_location)
        self.log_stimuli()
        self.buffer.reset()
        if assay["save frames"]:
            episode_data = load_data(f"{self.model_name}-{self.model_number}", self.assay_configuration_id,
                                     assay['assay id'], training_data=False)
            draw_episode(episode_data, self.config_name, f"{self.model_name}-{self.model_number}", self.continuous_actions,
                         save_id=f"{self.assay_configuration_id}-{assay['assay id']}", training_episode=False)

        print(f"Assay: {assay['assay id']} Completed")
        print("")
        return True

    def step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref):
        return BaseDQN.assay_step_loop(self, o, internal_state, a, rnn_state, rnn_state_ref)

    def save_hdf5_data(self, assay):
        # if assay["save frames"]:
        #     # make_gif(self.frame_buffer,
        #     #          f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
        #     #          duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        #     make_video(self.frame_buffer,
        #                f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.mp4",
        #                duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        # self.frame_buffer = []

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
                assay_group.create_dataset(key, data=np.array(self.output_data[key]))
            except RuntimeError:
                del assay_group[key]
                assay_group.create_dataset(key, data=np.array(self.output_data[key]))
        hdf5_file.close()

    def save_episode_data(self):
        self.episode_summary_data = {
            "Prey Caught": self.simulation.prey_caught,
            "Predators Avoided": self.simulation.predator_attacks_avoided,
            "Sand Grains Bumped": self.simulation.sand_grains_bumped,
        }
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-summary_data.json", "w") as output_file:
            json.dump(self.episode_summary_data, output_file)
        self.episode_summary_data = None

    def package_output_data(self, observation, rev_observation, action, advantage_stream, rnn_state, rnn2_state,
                            position, prey_consumed, predator_body,
                            conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r,
                            prey_positions, predator_position, sand_grain_positions, fish_angle):
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
            "fish_angle": fish_angle,
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
        # if assay["save frames"]:
        #     # make_gif(self.frame_buffer, f"{self.data_save_location}/{assay['assay id']}.gif",
        #     #          duration=len(self.frame_buffer) * self.learning_params['time_per_step'],
        #     #          true_image=True)
        #     make_video(self.frame_buffer, f"{self.data_save_location}/{assay['assay id']}.mp4",
        #                duration=len(self.frame_buffer) * self.learning_params['time_per_step'],
        #                true_image=True)
        # self.frame_buffer = []
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)
