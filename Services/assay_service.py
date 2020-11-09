import json

import numpy as np
import tensorflow.compat.v1 as tf

from Network.simfish_drqn import QNetwork
from Environment.simfish_env import SimState


class AssayService:

    def __init__(self, environment_name, trial_number, learning_params, environment_params, apparatus_mode, assays):

        self.trial_id = f"{environment_name}_{trial_number}"

        self.model_location = f"./Output/{self.trial_id}_output"
        self.data_save_location = f"./Assay-Output/{self.trial_id}"

        self.learning_params = learning_params
        self.environment_params = environment_params

        # Create the testing environment
        self.apparatus_mode = apparatus_mode
        self.simulation = self.create_testing_environment()

        # Create the assays
        self.assays = assays

        self.saver = None

        self.network = None
        self.init = None
        self.sess = None

        self.assay_output_data_format = None
        self.assay_output_data = []

        self.step_number = 0

    def build_assays(self):
        # Returns a list of assays which can be looped over. This must include information for the stimuli, interactions, and recording settings
        return True

    def create_network(self):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim'], state_is_tuple=True)
        network = QNetwork(simulation=self.simulation,
                           rnn_dim=self.learning_params['rnn_dim'],
                           rnn_cell=cell,
                           my_scope='main',
                           num_actions=self.learning_params['num_actions'],
                           learning_rate=self.learning_params['learning_rate'])
        return network

    def create_testing_environment(self):
        return SimState(self.environment_params)

    def run(self):
        with tf.Session() as self.sess:
            self.network = self.create_network()
            self.saver = tf.train.Saver(max_to_keep=5)
            self.init = tf.global_variables_initializer()
            checkpoint = tf.train.get_checkpoint_state(self.model_location)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Model loaded")
            for assay in self.assays:
                self.run_assay(assay)
                self.save_assay_results(assay)

    def run_assay(self, assay):
        self.assay_output_data_format = {key: None for key in assay["to record"]}

        self.simulation.reset()
        rnn_state = (np.zeros([1, self.network.rnn_dim]), np.zeros([1, self.network.rnn_dim]))  # Reset RNN hidden state
        sa = np.zeros((1, 128))  # Placeholder for the state advantage stream.
        frame_buffer = []  # TODO: Remove as unnecessary
        o, r, internal_state, d, frame_buffer = self.simulation.simulation_step(action=3,
                                                                                frame_buffer=frame_buffer,
                                                                                save_frames=False,
                                                                                activations=(sa,))
        a = 0
        self.step_number = 0
        while self.step_number < self.learning_params["max_epLength"]:
            self.step_number += 1
            o, a, r, internal_state, s1, d, rnn_state = self.step_loop(o=o, internal_state=internal_state,
                                                                       a=a, rnn_state=rnn_state)
            o = s1
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
        frame_buffer = []
        o1, given_reward, internal_state, d, frame_buffer = self.simulation.simulation_step(action=chosen_a,
                                                                                            frame_buffer=frame_buffer,
                                                                                            save_frames=False,
                                                                                            activations=(sa,))

        # Make output data JSON serializable
        action = int(chosen_a)
        advantage_stream = sa.tolist()
        rnn_state = updated_rnn_state.c.tolist()
        print(rnn_state)

        possible_data_to_save = {
            "behavioural choice": action,
            "rnn state": rnn_state,
            "advantage stream": advantage_stream,
        }  # Will work for now but note is inefficient
        self.make_recordings(possible_data_to_save)
        return o, chosen_a, given_reward, internal_state, o1, d, updated_rnn_state

    def make_recordings(self, available_data):
        step_data = {i: available_data[i] for i in self.assay_output_data_format}
        step_data["step"] = self.step_number
        self.assay_output_data.append(step_data)

    def save_assay_results(self, assay):
        # Saves all the information from the assays in JSON format.
        data = [
            {
                "step": 1,
                "action": 4,
                "advantage": 55,
            },
            {
                "step": 2,
                "action": 7,
                "advantage": 40,
            },
        ]  # TODO: Swith to compile real data.
        with open(f"{self.data_save_location}/{assay['assay id']}.json", "w") as output_file:
            json.dump(self.assay_output_data, output_file)
