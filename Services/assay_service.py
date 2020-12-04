import json

import numpy as np
import tensorflow.compat.v1 as tf

from Environment.naturalistic_environment import NaturalisticEnvironment
from Environment.controlled_stimulus_environment import ProjectionEnvironment
from Network.q_network import QNetwork
from Tools.make_gif import make_gif


class AssayService:

    def __init__(self, model_name, environment_name, trial_number, learning_params, environment_params, apparatus_mode, assays):

        self.model_id = f"{model_name}-{trial_number}"
        self.model_location = f"./Training-Output/{self.model_id}"
        self.data_save_location = f"./Assay-Output/{self.model_id}"

        self.learning_params = learning_params
        self.environment_params = environment_params

        # Create the testing environment TODO: Make individual for different assays.
        self.apparatus_mode = apparatus_mode
        self.simulation = NaturalisticEnvironment(self.environment_params)  # TODO: While this has no effects, is inelegant and should be changed. Potentially pass in eye size directly to netwrok rather than whole environment.

        # Create the assays
        self.assays = assays

        self.frame_buffer = []

        self.saver = None
        self.network = None
        self.init = None
        self.sess = None

        self.assay_output_data_format = None
        self.assay_output_data = []

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
            self.simulation = ProjectionEnvironment(self.environment_params, assay["stimuli"], tethered=assay["ish setup"])
        else:
            self.simulation =  NaturalisticEnvironment(self.environment_params)

    def run(self):
        with tf.Session() as self.sess:
            self.network = self.create_network()
            self.saver = tf.train.Saver(max_to_keep=5)
            self.init = tf.global_variables_initializer()
            checkpoint = tf.train.get_checkpoint_state(self.model_location)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Model loaded")
            for assay in self.assays:
                self.create_testing_environment(assay)
                self.perform_assay(assay)
                self.save_assay_results(assay)

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
        possible_data_to_save = self.package_output_data(chosen_a, sa, updated_rnn_state,
                                                         self.simulation.fish.body.position)
        self.make_recordings(possible_data_to_save)

        return o, chosen_a, given_reward, internal_state, o1, d, updated_rnn_state

    def make_recordings(self, available_data):
        step_data = {i: available_data[i] for i in self.assay_output_data_format}
        step_data["step"] = self.step_number
        self.assay_output_data.append(step_data)

    def package_output_data(self, action, advantage_stream, rnn_state, position):
        # Make output data JSON serializable
        action = int(action)
        advantage_stream = advantage_stream.tolist()
        rnn_state = rnn_state.c.tolist()
        position = list(position)
        # observation = observation.tolist()
        observation = self.simulation.get_visual_inputs()
        observation = observation.tolist()

        data = {
            "behavioural choice": action,
            "rnn state": rnn_state,
            "advantage stream": advantage_stream,
            "position": position,
            "observation": observation,
        }  # Will work for now but note is inefficient

        return data

    def save_assay_results(self, assay):
        # Saves all the information from the assays in JSON format.
        if assay["save frames"]:
            make_gif(self.frame_buffer, f"{self.data_save_location}/{assay['assay id']}.gif",
                     duration=len(self.frame_buffer) * self.learning_params['time_per_step'], true_image=True)
        self.frame_buffer = []
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)
