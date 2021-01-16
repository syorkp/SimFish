
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
    configuration_location = f"./Configurations/Assay-Configs/{environment_name}"
    with open(f"{configuration_location}_learning.json", 'r') as f:
        params = json.load(f)
    with open(f"{configuration_location}_env.json", 'r') as f:
        env = json.load(f)
    return params, env

learning, env = load_configuration_files("example")

simulation = NaturalisticEnvironment(env, False)

# Load the model
model_location = f"./Training-Output/conditional_prey_and_static-1"


with tf.Session() as sess:
    cell = tf.nn.rnn_cell.LSTMCell(num_units=256, state_is_tuple=True)
    network = QNetwork(simulation=simulation,
                       rnn_dim=256,
                       rnn_cell=cell,
                       my_scope='main',
                       num_actions=7,
                       learning_rate=0.0001)

    saver =tf.train.Saver(max_to_keep=5)
    init = tf.global_variables_initializer()
    checkpoint = tf.train.get_checkpoint_state(model_location)
    saver.restore(sess, checkpoint.model_checkpoint_path)
    vars = tf.trainable_variables()
    vals = sess.run(vars)
    x = 4
    for var, val in zip(vars, vals):
        print(var.name, val)





