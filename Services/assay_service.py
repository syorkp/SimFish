import json
import h5py
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.controlled_stimulus_environment import ControlledStimulusEnvironment
from Network.q_network import QNetwork
from Tools.make_gif import make_gif

tf.logging.set_verbosity(tf.logging.ERROR)


def assay_target(trial, learning_params, environment_params, total_steps, episode_number, memory_fraction):
    service = AssayService(model_name=trial["Model Name"],
                           trial_number=trial["Trial Number"],
                           assay_config_name=trial["Assay Configuration Name"],
                           learning_params=learning_params,
                           environment_params=environment_params,
                           total_steps=total_steps,
                           episode_number=episode_number,
                           assays=trial["Assays"],
                           realistic_bouts=trial["Realistic Bouts"],
                           memory_fraction=memory_fraction,
                           using_gpu=trial["Using GPU"]
                           )

    service.run()


class AssayService:

    def __init__(self, model_name, trial_number, assay_config_name, learning_params, environment_params, total_steps,
                 episode_number, assays, realistic_bouts, memory_fraction, using_gpu):
        """
        Runs a set of assays provided by the run configuraiton.
        """
        # Names and Directories
        self.model_id = f"{model_name}-{trial_number}"
        self.model_location = f"./Training-Output/{self.model_id}"
        self.data_save_location = f"./Assay-Output/{self.model_id}"

        # Configurations
        self.assay_configuration_id = assay_config_name
        self.learning_params = learning_params
        self.environment_params = environment_params
        self.assays = assays

        # Basic Parameters
        self.using_gpu = using_gpu
        self.realistic_bouts = realistic_bouts
        self.memory_fraction = memory_fraction

        # Network Parameters
        self.saver = None
        self.network = None
        self.init = None
        self.sess = None

        # Simulation
        self.simulation = NaturalisticEnvironment(self.environment_params, self.realistic_bouts)
        self.step_number = 0

        # Data
        self.metadata = {
            "Total Episodes": episode_number,
            "Total Steps": total_steps,
        }
        self.frame_buffer = []
        self.assay_output_data_format = None
        self.assay_output_data = []
        self.output_data = {}
        self.episode_summary_data = None

        # Hacky fix for h5py problem:
        self.last_position_dim = self.environment_params["prey_num"]

    def create_network(self):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim'], state_is_tuple=True)
        network = QNetwork(simulation=self.simulation,
                           rnn_dim=self.learning_params['rnn_dim'],
                           rnn_cell=cell,
                           my_scope='main',
                           num_actions=self.learning_params['num_actions'],
                           learning_rate=self.learning_params['learning_rate'])
        return network

    def create_testing_environment(self, assay):
        """
        Creates the testing environment as specified  by apparatus mode and given assays.
        :return:
        """
        if assay["stimulus paradigm"] == "Projection":
            self.simulation = ControlledStimulusEnvironment(self.environment_params, assay["stimuli"],
                                                            self.realistic_bouts,
                                                            tethered=assay["Tethered"])
        elif assay["stimulus paradigm"] == "Naturalistic":
            self.simulation = NaturalisticEnvironment(self.environment_params, self.realistic_bouts)
        else:
            self.simulation = NaturalisticEnvironment(self.environment_params, self.realistic_bouts)

    def run(self):
        if self.using_gpu:
            options = tf.GPUOptions(per_process_gpu_memory_fraction=self.memory_fraction)
        else:
            options = None

        if options:
            with tf.Session(config=tf.ConfigProto(gpu_options=options)) as self.sess:
                self._run()
        else:
            with tf.Session() as self.sess:
                self._run()

    def _run(self):
        self.network = self.create_network()
        self.saver = tf.train.Saver(max_to_keep=5)
        self.init = tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(self.model_location)
        self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
        print("Model loaded")
        for assay in self.assays:
            self.create_output_data_storage(assay)
            self.create_testing_environment(assay)
            self.perform_assay(assay)
            # self.save_assay_results(assay)
            self.save_hdf5_data(assay)
        self.save_metadata()
        self.save_episode_data()

    def create_output_data_storage(self, assay):
        self.output_data = {key: [] for key in assay["recordings"]}
        self.output_data["step"] = []

    def perform_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["recordings"]}

        self.simulation.reset()
        rnn_state = (np.zeros([1, self.network.rnn_dim]), np.zeros([1, self.network.rnn_dim]))  # Reset RNN hidden state
        sa = np.zeros((1, 128))

        o, r, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=3,
                                                                                     frame_buffer=self.frame_buffer,
                                                                                     save_frames=True,
                                                                                     activations=(sa,))
        a = 0
        self.step_number = 0
        while self.step_number < assay["duration"]:
            self.step_number += 1
            o, a, r, internal_state, o1, d, rnn_state = self.step_loop(o=o, internal_state=internal_state,
                                                                       a=a, rnn_state=rnn_state)
            o = o1
            if d:
                break

    def step_loop(self, o, internal_state, a, rnn_state):
        chosen_a, updated_rnn_state, sa, sv, conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r = \
            self.sess.run(
                [self.network.predict, self.network.rnn_state, self.network.streamA, self.network.streamV,
                 self.network.conv1l, self.network.conv2l, self.network.conv3l, self.network.conv4l,
                 self.network.conv1r, self.network.conv2r, self.network.conv3r, self.network.conv4r
                 ],
                feed_dict={self.network.observation: o,
                           self.network.internal_state: internal_state,
                           self.network.prev_actions: [a],
                           self.network.trainLength: 1,
                           self.network.state_in: rnn_state,
                           self.network.batch_size: 1,
                           self.network.exp_keep: 1.0})
        chosen_a = chosen_a[0]
        o1, given_reward, internal_state, d, self.frame_buffer = self.simulation.simulation_step(action=chosen_a,
                                                                                                 frame_buffer=self.frame_buffer,
                                                                                                 save_frames=True,
                                                                                                 activations=(sa,))

        fish_angle = self.simulation.fish.body.angle

        if not self.simulation.sand_grain_bodies:
            sand_grain_positions = [self.simulation.sand_grain_bodies[i].position for i, b in
                                    enumerate(self.simulation.sand_grain_bodies)]
            sand_grain_positions = [[i[0], i[1]] for i in sand_grain_positions]
        else:
            sand_grain_positions = [[10000, 10000]]

        if self.simulation.prey_bodies:
            # TODO: Note hacky fix which may want to clean up later.
            prey_positions = [prey.position for prey in self.simulation.prey_bodies]
            prey_positions = [[i[0], i[1]] for i in prey_positions]
            while True:
                if len(prey_positions) < self.last_position_dim:
                    prey_positions = np.append(prey_positions, [[10000, 10000]], axis=0)
                else:
                    break

            self.last_position_dim = len(prey_positions)

        else:
            prey_positions = np.array([[10000, 10000]])

        if self.simulation.predator_body is not None:
            predator_position = self.simulation.predator_body.position
            predator_position = np.array([predator_position[0], predator_position[1]])
        else:
            predator_position = np.array([10000, 10000])

        if self.simulation.vegetation_bodies is not None:
            vegetation_positions = [self.simulation.vegetation_bodies[i].position for i, b in enumerate(self.simulation.vegetation_bodies)]
            vegetation_positions = [[i[0], i[1]] for i in vegetation_positions]
        else:
            vegetation_positions = [[10000, 10000]]

        # Saving step data
        possible_data_to_save = self.package_output_data(chosen_a, sa, updated_rnn_state,
                                                         self.simulation.fish.body.position,
                                                         self.simulation.prey_consumed_this_step,
                                                         self.simulation.predator_body,
                                                         conv1l, conv2l, conv3l, conv4l, conv1r, conv2r, conv3r, conv4r,
                                                         prey_positions,
                                                         predator_position,
                                                         sand_grain_positions,
                                                         vegetation_positions,
                                                         fish_angle
                                                         )
        for key in self.assay_output_data_format:
            self.output_data[key].append(possible_data_to_save[key])
        self.output_data["step"].append(self.step_number)

        return o, chosen_a, given_reward, internal_state, o1, d, updated_rnn_state

    def save_hdf5_data(self, assay):
        if assay["save frames"]:
            make_gif(self.frame_buffer,
                     f"{self.data_save_location}/{self.assay_configuration_id}-{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        self.frame_buffer = []

        hdf5_file = h5py.File(f"{self.data_save_location}/{self.assay_configuration_id}.h5", "a")
        try:
            assay_group = hdf5_file.create_group(assay['assay id'])
        except ValueError:
            assay_group = hdf5_file.get(assay['assay id'])

        self.output_data["prey_positions"] = np.stack(self.output_data["prey_positions"])
        for key in self.output_data:
            try:
                assay_group.create_dataset(key, data=np.array(self.output_data[key]))  # TODO: Compress data.
            except RuntimeError:
                del assay_group[key]
                assay_group.create_dataset(key, data=np.array(self.output_data[key]))  # TODO: Compress data.
        hdf5_file.close()

    def save_episode_data(self):
        self.episode_summary_data = {
            "Prey Caught": self.simulation.prey_caught,
            "Predators Avoided": self.simulation.predators_avoided,
            "Sand Grains Bumped": self.simulation.sand_grains_bumped,
            "Steps Near Vegetation": self.simulation.steps_near_vegetation
        }
        with open(f"{self.data_save_location}/{self.assay_configuration_id}-summary_data.json", "w") as output_file:
            json.dump(self.episode_summary_data, output_file)
        self.episode_summary_data = None

    def save_metadata(self):
        self.metadata["Assay Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(f"{self.data_save_location}/{self.assay_configuration_id}.json", "w") as output_file:
            json.dump(self.metadata, output_file)

    def package_output_data(self, action, advantage_stream, rnn_state, position, prey_consumed, predator_body,
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
        observation = self.simulation.fish.get_visual_inputs()
        observation = observation.tolist()

        data = {
            "behavioural choice": action,
            "rnn state": rnn_state,
            "advantage stream": advantage_stream,
            "position": position,
            "observation": observation,
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
            "fish_angle": fish_angle
        }

        if prey_consumed:
            data["consumed"] = 1
        else:
            data["consumed"] = 0
        if predator_body is not None:
            data["predator"] = 1
        else:
            data["predator"] = 0

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
