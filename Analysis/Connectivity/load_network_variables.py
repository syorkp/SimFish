import tensorflow.compat.v1 as tf
import json

from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Networks.A2C.advantage_actor_critic import A2CNetwork
from Networks.PPO.proximal_policy_optimizer_continuous import PPONetworkActor

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


def load_network_variables_a2c(model_name, conf_name):
    learning, env = load_configuration_files(f"{conf_name}")
    simulation = DiscreteNaturalisticEnvironment(env, False)
    model_location = f"../../Training-Output/{model_name}"

    with tf.Session() as sess:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=learning["rnn_dim"], state_is_tuple=True)
        internal_states = sum([1 for x in [env['hunger'], env['stress']] if x is True]) + 1
        network = A2CNetwork(simulation=simulation,
                             rnn_dim_shared=learning['rnn_dim'],
                             rnn_dim_critic=learning['rnn_dim'],
                             rnn_dim_actor=learning['rnn_dim'],
                             rnn_cell_shared=cell,
                             rnn_cell_critic=cell,
                             rnn_cell_actor=cell,
                             my_scope='main',
                             internal_states=internal_states,
                             actor_learning_rate_impulse=0.00001,
                             actor_learning_rate_angle=0.00001,
                             critic_learning_rate=0.00001,
                             max_impulse=10.0,
                             max_angle_change=3.0)

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


def load_network_variables_ppo(model_name, conf_name):
    learning, env = load_configuration_files(f"{conf_name}")
    simulation = ContinuousNaturalisticEnvironment(env, False)
    model_location = f"../../Training-Output/{model_name}"

    with tf.Session() as sess:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=learning["rnn_dim"], state_is_tuple=True)
        internal_states = sum([1 for x in [env['hunger'], env['stress']] if x is True]) + 1
        network = PPONetworkActor(simulation=simulation,
                                  rnn_dim=learning['rnn_dim'],
                                  rnn_dim_critic=learning['rnn_dim'],
                                  rnn_dim_actor=learning['rnn_dim'],
                                  rnn_cell=cell,
                                  rnn_cell_critic=cell,
                                  rnn_cell_actor=cell,
                                  my_scope='main',
                                  internal_states=internal_states,
                                  actor_learning_rate_impulse=0.00001,
                                  actor_learning_rate_angle=0.00001,
                                  learning_rate=0.00001,
                                  max_impulse=10.0,
                                  max_angle_change=3.0)

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


v = load_network_variables_ppo("updated_ppo-4", "1")

x = True
