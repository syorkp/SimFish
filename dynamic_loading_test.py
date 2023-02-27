import json
import numpy as np

import tensorflow.compat.v1 as tf

from Networks.DQN.q_network_dynamic import QNetworkDynamic

from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment

from Tools.graph_functions import update_target_graph, update_target

tf.disable_v2_behavior()
tf.logging.set_verbosity(tf.logging.ERROR)


class DynamicLoadingTest:

    def __init__(self, model_name, trial_number, config_to_load):
        # Name and location
        self.model_id = f"{model_name}-{trial_number}"
        self.model_location = f"./Training-Output/{self.model_id}"

        self.config_name = model_name
        self.configuration_index = config_to_load
        self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"

        self.learning_params, self.environment_params = self.load_configuration_files()
        self.simulation = DiscreteNaturalisticEnvironment(self.environment_params, True, True, False)

        sess = self.create_session()
        with sess as self.sess:
            # Initial loading
            self.create_network()
            self.init_states()
            self.saver = tf.train.Saver(max_to_keep=5)
            self.init = tf.global_variables_initializer()
            self.trainables = tf.trainable_variables()
            checkpoint = tf.train.get_checkpoint_state(self.model_location)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            self.saver.save(self.sess, f"{self.model_location}/model-{str(3000)}.cptk")

        tf.reset_default_graph()
        sess = self.create_session()
        with sess as self.sess:
            # Then switch config and call reload function
            self.configuration_index = config_to_load + 1
            self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"
            self.learning_params, self.environment_params = self.load_configuration_files()
            print("Saved Model")
            self.create_network()
            self.init_states()
            checkpoint = tf.train.get_checkpoint_state(self.model_location)
            # tf.train.import_meta_graph(checkpoint.model_checkpoint_path + ".meta")
            # x = tf.get_default_graph().as_graph_def()

            variables_to_keep = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            variables_to_keep = self.remove_new_variables(variables_to_keep, ["new_dense", "targetaw", "mainaw",
                                                                              "mainvw", "targetvw"])

            self.saver = tf.train.Saver(max_to_keep=5, var_list=variables_to_keep)
            self.init = tf.global_variables_initializer()
            self.trainables = tf.trainable_variables()

            self.target_ops = update_target_graph(self.trainables, self.learning_params['tau'])
            self.sess.run(self.init)
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            # all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # sp = [var for var in all_variables if "_new_dense" in var.name]
            #
            # tf.variables_initializer(sp)
            update_target(self.target_ops, self.sess)
            self.saver = tf.train.Saver(max_to_keep=5)

            # Load possible parameters
            self.saver.save(self.sess, f"{self.model_location}/model-{str(3001)}.cptk")
            print("Saved Model")

        tf.reset_default_graph()
        sess = self.create_session()
        with sess as self.sess:
            # Then switch config and call reload function
            self.configuration_index = self.configuration_index + 1
            self.current_configuration_location = f"./Configurations/Training-Configs/{self.config_name}/{str(self.configuration_index)}"
            self.learning_params, self.environment_params = self.load_configuration_files()
            print("Saved Model")
            self.create_network()
            self.init_states()
            checkpoint = tf.train.get_checkpoint_state(self.model_location)
            # tf.train.import_meta_graph(checkpoint.model_checkpoint_path + ".meta")
            # x = tf.get_default_graph().as_graph_def()

            variables_to_keep = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

            variables_to_keep = self.remove_new_variables(variables_to_keep, ["XYZKDF"])

            self.saver = tf.train.Saver(max_to_keep=5, var_list=variables_to_keep)
            self.init = tf.global_variables_initializer()
            self.trainables = tf.trainable_variables()


            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

            self.saver = tf.train.Saver(max_to_keep=5)

            # Load possible parameters
            self.saver.save(self.sess, f"{self.model_location}/model-{str(8011)}.cptk")
            print("Saved Model")

    def create_session(self):
        print("Creating Session..")

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        if config:
            return tf.Session(config=config)
        else:
            return tf.Session()

    def remove_new_variables(self, var_list, new_var_names):
        filtered_var_list = []
        for var in var_list:
            if any(new_name in var.name for new_name in new_var_names):
                print(f"Found in {var.name}")
            else:
                filtered_var_list.append(var)
        return filtered_var_list

    def load_configuration_files(self):
        with open(f"{self.current_configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{self.current_configuration_location}_env.json", 'r') as f:
            env = json.load(f)
        return params, env

    def get_internal_state_order(self):
        internal_state_order = []
        if self.environment_params['in_light']:
            internal_state_order.append("in_light")
        if self.environment_params['stress']:
            internal_state_order.append("stress")
        if self.environment_params['energy_state']:
            internal_state_order.append("energy_state")
        if self.environment_params['salt']:
            internal_state_order.append("salt")
        return internal_state_order

    def init_states(self):
        # Init states for RNN
        rnn_state_shapes = self.main_QN.get_rnn_state_shapes()
        self.init_rnn_state = tuple(
            (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)
        self.init_rnn_state_ref = tuple(
            (np.zeros([1, shape]), np.zeros([1, shape])) for shape in rnn_state_shapes)

    def create_network(self):
        internal_states = sum(
            [1 for x in [self.environment_params['stress'],
                         self.environment_params['energy_state'], self.environment_params['in_light'],
                         self.environment_params['salt']] if x is True])
        internal_states = max(internal_states, 1)
        internal_state_names = self.get_internal_state_order()

        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)
        cell_t = tf.nn.rnn_cell.LSTMCell(num_units=self.learning_params['rnn_dim_shared'], state_is_tuple=True)

        self.main_QN = QNetworkDynamic(simulation=self.simulation,
                                       my_scope='main',
                                       internal_states=internal_states,
                                       internal_state_names=internal_state_names,
                                       num_actions=self.learning_params['num_actions'],
                                       base_network_layers=self.learning_params[
                                           'base_network_layers'],
                                       modular_network_layers=self.learning_params[
                                           'modular_network_layers'],
                                       ops=self.learning_params['ops'],
                                       connectivity=self.learning_params[
                                           'connectivity'],
                                       reflected=self.learning_params['reflected'],
                                       reuse_eyes=False,
                                       )
        self.target_QN = QNetworkDynamic(simulation=self.simulation,

                                         my_scope='target',
                                         internal_states=internal_states,
                                         internal_state_names=internal_state_names,
                                         num_actions=self.learning_params['num_actions'],
                                         base_network_layers=self.learning_params[
                                             'base_network_layers'],
                                         modular_network_layers=self.learning_params[
                                             'modular_network_layers'],
                                         ops=self.learning_params['ops'],
                                         connectivity=self.learning_params[
                                             'connectivity'],
                                         reflected=self.learning_params['reflected'],
                                         reuse_eyes=False,
                                         )


model_name = "dqn_scaffold_dn_switch"
DynamicLoadingTest(model_name, 1, 1)

