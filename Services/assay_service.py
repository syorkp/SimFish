import json
import h5py
from datetime import datetime

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.controlled_stimulus_environment import ProjectionEnvironment
from Network.q_network import QNetwork
from Tools.make_gif import make_gif

tf.logging.set_verbosity(tf.logging.ERROR)


def assay_target(trial, learning_params, environment_params, total_steps, episode_number, memory_fraction):
    using_gpu = tf.test.is_gpu_available(cuda_only=True)
    print(using_gpu)
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

        self.model_id = f"{model_name}-{trial_number}"
        self.model_location = f"./Training-Output/{self.model_id}"
        self.data_save_location = f"./Assay-Output/{self.model_id}"

        self.using_gpu = using_gpu

        self.realistic_bouts = realistic_bouts

        self.assay_configuration_id = assay_config_name

        self.learning_params = learning_params
        self.environment_params = environment_params

        self.simulation = NaturalisticEnvironment(self.environment_params, self.realistic_bouts)
        self.metadata = {
            "Total Episodes": episode_number,
            "Total Steps": total_steps,
        }
        # TODO: Consider creating a base service.

        # Create the assays
        self.assays = assays
        self.memory_fraction = memory_fraction

        self.frame_buffer = []

        self.saver = None
        self.network = None
        self.init = None
        self.sess = None

        self.assay_output_data_format = None
        self.assay_output_data = []

        self.output_data = {}

        self.step_number = 0

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
            self.simulation = ProjectionEnvironment(self.environment_params, assay["stimuli"], self.realistic_bouts,
                                                    tethered=assay["fish setup"])
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
        chosen_a, updated_rnn_state, sa, sv = self.sess.run(
            [self.network.predict, self.network.rnn_state, self.network.streamA, self.network.streamV],
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

        # Saving step data
        possible_data_to_save = self.package_output_data(chosen_a, sa, updated_rnn_state,
                                                         self.simulation.fish.body.position)
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

        for key in self.output_data:
            try:
                assay_group.create_dataset(key, data=self.output_data[key])  # TODO: Compress data.
            except RuntimeError:
                del assay_group[key]
                assay_group.create_dataset(key, data=self.output_data[key])  # TODO: Compress data.
        hdf5_file.close()

    def save_metadata(self):
        self.metadata["Assay Date"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        with open(f"{self.data_save_location}/{self.assay_configuration_id}.json", "w") as output_file:
            json.dump(self.metadata, output_file)

    def package_output_data(self, action, advantage_stream, rnn_state, position):
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
        }  # Will work for now but note is inefficient

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
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True) # TODO: Remove 100 to change back to normal.
        self.frame_buffer = []
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)
