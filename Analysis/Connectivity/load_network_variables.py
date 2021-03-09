
import tensorflow.compat.v1 as tf
import json


from Environment.naturalistic_environment import NaturalisticEnvironment
from Network.q_network import QNetwork

tf.disable_v2_behavior()


def load_configuration_files(environment_name):
    """
    Called by create_trials method, should return the learning and environment configurations in JSON format.
    :param environment_name:
    :return:
    """
    print("Loading configuration...")
    configuration_location = f"../../Configurations/Assay-Configs/{environment_name}"
    with open(f"{configuration_location}_learning.json", 'r') as f:
        params = json.load(f)
    with open(f"{configuration_location}_env.json", 'r') as f:
        env = json.load(f)
    return params, env


def load_network_variables(model_name, conf_name):
    learning, env = load_configuration_files(f"{conf_name}")
    simulation = NaturalisticEnvironment(env, False)
    model_location = f"../../Training-Output/{model_name}"

    with tf.Session() as sess:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=learning["rnn_dim"], state_is_tuple=True)
        internal_states = sum([1 for x in [env['hunger'], env['stress']] if x is True]) + 1
        network = QNetwork(simulation=simulation,
                           rnn_dim=learning["rnn_dim"],
                           rnn_cell=cell,
                           my_scope='main',
                           internal_states=internal_states,
                           num_actions=learning["num_actions"],
                           learning_rate=learning["learning_rate"])
        saver = tf.train.Saver(max_to_keep=5)
        init = tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(model_location)
        saver.restore(sess, checkpoint.model_checkpoint_path)
        vars = tf.trainable_variables()
        vals = sess.run(vars)
        sorted_vars = {}
        for var, val in zip(vars, vals):
            sorted_vars[str(var.name)] = val
        return sorted_vars


v = load_network_variables("test-1", "1")

x = True
