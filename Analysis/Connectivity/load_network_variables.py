import numpy as np
import tensorflow.compat.v1 as tf
import json

from Environment.discrete_naturalistic_environment import DiscreteNaturalisticEnvironment
from Environment.continuous_naturalistic_environment import ContinuousNaturalisticEnvironment
from Networks.PPO.proximal_policy_optimizer_continuous import PPONetworkActor
from Networks.PPO.proximal_policy_optimizer_continuous_multivariate import PPONetworkActorMultivariate
from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator import PPONetworkActorMultivariate2
from Networks.PPO.proximal_policy_optimizer_continuous_beta_sb_emulator import PPONetworkActorMultivariateBetaNormal2
from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator_dynamic import PPONetworkActorMultivariate2Dynamic
from Networks.PPO.proximal_policy_optimizer_continuous_sb_emulator_split_networks import PPONetworkActorMultivariate2A, \
    PPONetworkActorMultivariate2V
from Networks.DQN.q_network import QNetwork
from Networks.DQN.q_network_dynamic import QNetworkDynamic

tf.disable_v2_behavior()


def get_internal_state_order(environment_params):
    internal_state_order = []
    if environment_params['in_light']:
        internal_state_order.append("in_light")
    if environment_params['stress']:
        internal_state_order.append("stress")
    if environment_params['energy_state']:
        internal_state_order.append("energy_state")
    if environment_params['salt']:
        internal_state_order.append("salt")
    return internal_state_order


def load_configuration_files(environment_name):
    """
    Called by create_trials method, should return the learning and environment configurations in JSON format.
    :param environment_name:
    :return:
    """
    print("Loading configuration...")
    try:
        configuration_location = f"../../../Configurations/Assay-Configs/{environment_name}"
        with open(f"{configuration_location}_learning.json", 'r') as f:
            params = json.load(f)
        with open(f"{configuration_location}_env.json", 'r') as f:
            env = json.load(f)
    except FileNotFoundError:
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
        internal_states = sum([1 for x in [env['stress']] if x is True]) + 1
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


def create_ppo_network(simulation, environment_params, learning_params, multivariate=True):
        """
        Create the main and target Q networks, according to the configuration parameters.
        :return: The main network and the target network graphs.
        """
        print("Creating networks...")
        internal_states = sum(
            [1 for x in [environment_params['stress'],
                         environment_params['energy_state'], environment_params['in_light'],
                         environment_params['salt']] if x is True])
        internal_states = max(internal_states, 1)
        internal_state_names = get_internal_state_order(environment_params)

        actor_cell = tf.nn.rnn_cell.LSTMCell(num_units=learning_params['rnn_dim_shared'], state_is_tuple=True)

        if multivariate:
            if "reuse_eyes" in learning_params:
                reuse_eyes = learning_params['reuse_eyes']
            else:
                reuse_eyes = False
            actor_network = PPONetworkActorMultivariate2Dynamic(simulation=simulation,
                                                                     # rnn_dim=learning_params['rnn_dim_shared'],
                                                                     # rnn_cell=actor_cell,
                                                                     my_scope='actor',
                                                                     internal_states=internal_states,
                                                                     internal_state_names=internal_state_names,
                                                                     max_impulse=environment_params[
                                                                         'max_impulse'],
                                                                     max_angle_change=environment_params[
                                                                         'max_angle_change'],
                                                                     clip_param=environment_params[
                                                                         'clip_param'],
                                                                     input_sigmas=learning_params[
                                                                         'input_sigmas'],
                                                                     base_network_layers=learning_params[
                                                                         'base_network_layers'],
                                                                     modular_network_layers=learning_params[
                                                                         'modular_network_layers'],
                                                                     ops=learning_params['ops'],
                                                                     connectivity=learning_params[
                                                                         'connectivity'],
                                                                     reflected=learning_params['reflected'],
                                                                     reuse_eyes=reuse_eyes,
                                                                     )

        print("Created network")
        return actor_network


def load_network_variables_ppo(model_name, conf_name):
    learning, env = load_configuration_files(f"{conf_name}")
    simulation = ContinuousNaturalisticEnvironment(env, False, True, False)
    model_location = f"../../Training-Output/{model_name}"

    with tf.Session() as sess:
        network = create_ppo_network(simulation, env, learning)

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


def load_network_variables_dqn(model_name, conf_name, full_reafference=False):
    learning, env = load_configuration_files(f"{conf_name}")
    simulation = DiscreteNaturalisticEnvironment(env, False, True, False)

    with tf.Session() as sess:
        cell = tf.nn.rnn_cell.LSTMCell(num_units=learning["rnn_dim_shared"], state_is_tuple=True)
        internal_states = sum(
            [1 for x in [env['stress'],
                         env['energy_state'], env['in_light'],
                         env['salt']] if x is True])
        internal_states = max(internal_states, 1)
        internal_state_names = get_internal_state_order(env)

        network = QNetworkDynamic(simulation=simulation,
                                  my_scope='main',
                                  internal_states=internal_states,
                                  internal_state_names=internal_state_names,
                                  num_actions=learning['num_actions'],
                                  base_network_layers=learning[
                                      'base_network_layers'],
                                  modular_network_layers=learning[
                                      'modular_network_layers'],
                                  ops=learning['ops'],
                                  connectivity=learning[
                                      'connectivity'],
                                  reflected=learning['reflected'],
                                  )

        saver = tf.train.Saver(max_to_keep=5)
        init = tf.global_variables_initializer()
        try:
            model_location = f"../Training-Output/{model_name}"
            checkpoint = tf.train.get_checkpoint_state(model_location)
            saver.restore(sess, checkpoint.model_checkpoint_path)

        except AttributeError:
            try:
                model_location = f"../../Training-Output/{model_name}"
                checkpoint = tf.train.get_checkpoint_state(model_location)
                saver.restore(sess, checkpoint.model_checkpoint_path)
            except AttributeError:
                try:
                    model_location = f"../../../Training-Output/{model_name}"
                    checkpoint = tf.train.get_checkpoint_state(model_location)
                    saver.restore(sess, checkpoint.model_checkpoint_path)
                except AttributeError:
                    model_location = f"../../../../Training-Output/{model_name}"
                    checkpoint = tf.train.get_checkpoint_state(model_location)
                    saver.restore(sess, checkpoint.model_checkpoint_path)

        vars = tf.trainable_variables()
        vals = sess.run(vars)
        sorted_vars = {}
        for var, val in zip(vars, vals):
            sorted_vars[str(var.name)] = val
    return sorted_vars


if __name__ == "__main__":
    # v = load_network_variables_ppo("updated_ppo-4", "1")
    # v = load_network_variables_dqn("dqn_scaffold_26-2", "dqn_26_2", full_reafference=True)
    # rnn = v["main_rnn/lstm_cell/kernel:0"]
    # with open('dqn26_2_rnn.npy', 'wb') as f:
    #     np.save(f, rnn)
    v = load_network_variables_ppo("ppo_scaffold_21-2", "ppo_21_2")
    rnn = v["actor_rnn/lstm_cell/kernel:0"]
    with open('ppo21_2_rnn.npy', 'wb') as f:
        np.save(f, rnn)


