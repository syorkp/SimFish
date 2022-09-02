import tensorflow.compat.v1 as tf

from Networks.DQN.q_network import QNetwork
from Networks.DQN.q_network_dynamic import QNetworkDynamic


def get_internal_state_order(environment_params):
    internal_state_order = []
    if environment_params['in_light']:
        internal_state_order.append("in_light")
    if environment_params['hunger']:
        internal_state_order.append("hunger")
    if environment_params['stress']:
        internal_state_order.append("stress")
    if environment_params['energy_state']:
        internal_state_order.append("energy_state")
    if environment_params['salt']:
        internal_state_order.append("salt")
    return internal_state_order


def build_network_dqn(environment_params, learning_params, simulation, full_reafference=True):
    internal_states = sum(
        [1 for x in [environment_params['hunger'], environment_params['stress'],
                     environment_params['energy_state'], environment_params['in_light'],
                     environment_params['salt']] if x is True])
    internal_states = max(internal_states, 1)
    internal_state_names = get_internal_state_order(environment_params)

    cell = tf.nn.rnn_cell.LSTMCell(num_units=learning_params['rnn_dim_shared'], state_is_tuple=True)
    if environment_params["use_dynamic_network"]:
        if "reuse_eyes" in learning_params:
            reuse_eyes = learning_params['reuse_eyes']
        else:
            reuse_eyes = False
        main_QN = QNetworkDynamic(simulation=simulation,
                                  my_scope='main',
                                  internal_states=internal_states,
                                  internal_state_names=internal_state_names,
                                  num_actions=learning_params['num_actions'],
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
    else:
        main_QN = QNetwork(simulation=simulation,
                           rnn_dim=learning_params['rnn_dim_shared'],
                           rnn_cell=cell,
                           my_scope='main',
                           num_actions=learning_params['num_actions'],
                           internal_states=internal_states,
                           learning_rate=learning_params['learning_rate'],
                           extra_layer=learning_params['extra_rnn'],
                           full_reafference=full_reafference)
    return main_QN
