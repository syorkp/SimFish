import copy

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


def flatten_list(t):
    return [item for sublist in t for item in sublist]


class DynamicBaseNetwork:

    def __init__(self, simulation, my_scope, internal_states, internal_state_names, action_dim, num_actions,
                 base_network_layers, modular_network_layers, ops, connectivity, reflected, algorithm, reuse_eyes):

        self.resolve_connectivity(base_network_layers, modular_network_layers, ops, connectivity)

        # Network parameters
        self.photoreceptor_num = simulation.fish.left_eye.observation_size  # Photoreceptor num per channel in each eye
        self.train_length = tf.placeholder(dtype=tf.int32, shape=[], name="train_length")
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.scope = my_scope

        # Network inputs
        if algorithm == "dqn":
            self.prev_actions = tf.placeholder(shape=[None, 3], dtype=tf.float32, name='prev_actions')
            self.prev_action_consequences = self.prev_actions[:, 1:]
            self.prev_action_impulse = self.prev_action_consequences[:, :1]
            self.prev_action_angle = self.prev_action_consequences[:, 1:]
            self.prev_chosen_actions = self.prev_actions[:, 0]
            self.prev_chosen_actions = tf.cast(self.prev_chosen_actions, dtype=tf.int32)
            self.prev_actions_one_hot = tf.one_hot(self.prev_chosen_actions, num_actions, dtype=tf.float32)
        else:
            self.prev_actions = tf.placeholder(shape=[None, action_dim * 2], dtype=tf.float32, name='prev_actions')

        self.internal_state = tf.placeholder(shape=[None, internal_states], dtype=tf.float32, name='internal_state')
        self.observation = tf.placeholder(shape=[None, 3, 2], dtype=tf.float32, name='observation')
        self.reshaped_observation = tf.reshape(self.observation, shape=[-1, self.photoreceptor_num, 3, 2],
                                               name="reshaped_observation")
        if reuse_eyes:
            # If re-using weights from one eye to the other, reflect the observation reaching the right eye.
            left_eye = self.reshaped_observation[:, :, :, 0:1]
            right_eye = self.reshaped_observation[:, :, :, 1:2]
            right_eye = tf.reverse(right_eye, axis=[1])
            self.reshaped_observation = tf.concat((left_eye, right_eye), axis=3)

        # Record of processing step layers
        self.base_network_layers = base_network_layers
        self.modular_network_layers = modular_network_layers
        self.ops = ops
        self.connectivity = connectivity
        self.internal_state_names = internal_state_names

        self.rnn_cells, self.rnn_cell_states, self.rnn_dim = self.create_rnns(base_network_layers,
                                                                              modular_network_layers, reflected)

        # Contains all layers to be referenced.
        if algorithm == "dqn":
            self.network_graph = {"observation": self.reshaped_observation,
                                  "internal_state": self.internal_state,
                                  "prev_actions": self.prev_actions_one_hot,
                                  "prev_action_impulse": self.prev_action_impulse,
                                  "prev_action_angle": self.prev_action_angle,
                                  }
        else:
            self.network_graph = {"observation": self.reshaped_observation,
                                  "internal_state": self.internal_state,
                                  "prev_actions": self.prev_actions}
        self.initialize_network(copy.copy(base_network_layers), copy.copy(modular_network_layers), copy.copy(ops),
                                connectivity, self.network_graph, reuse_eyes=reuse_eyes)

        if reflected:
            print("Building reflected graph")
            if algorithm == "dqn":
                self.reflected_network_graph = {"observation": self.reshaped_observation,
                                                "internal_state": self.internal_state,
                                                "prev_actions": tf.reverse(self.prev_actions_one_hot, [1]),
                                                "prev_action_impulse": self.prev_action_impulse,
                                                "prev_action_angle": self.prev_action_angle,
                                                }
            else:
                self.reflected_network_graph = {"observation": self.reshaped_observation,
                                                "internal_state": self.internal_state,
                                                "prev_actions": tf.reverse(self.prev_actions, [1])}
            self.initialize_network(copy.copy(base_network_layers), copy.copy(modular_network_layers), copy.copy(ops),
                                    connectivity, self.reflected_network_graph, reflected=True, reuse_eyes=reuse_eyes)

        # Name the outputs
        self.processing_network_output = self.network_graph["output_layer"]

        if reflected:
            self.processing_network_output_ref = self.reflected_network_graph["output_layer"]

        self.initialize_rnn_states(reflected)

    def get_rnn_state_shapes(self):
        rnn_layers_base = [layer for layer in self.base_network_layers.keys() if
                           self.base_network_layers[layer][0] == "dynamic_rnn"]
        rnn_layers_modular = [layer for layer in self.modular_network_layers.keys() if
                              self.modular_network_layers[layer][0] == "dynamic_rnn"]
        layer_sizes = [self.base_network_layers[layer][1] for layer in rnn_layers_base] + [
            self.modular_network_layers[layer][1] for layer in rnn_layers_modular]
        return layer_sizes

    def initialize_rnn_states(self, reflected):
        rnn_layers_base = [layer for layer in self.base_network_layers.keys() if
                           self.base_network_layers[layer][0] == "dynamic_rnn"]
        rnn_layers_modular = [layer for layer in self.modular_network_layers.keys() if
                              self.modular_network_layers[layer][0] == "dynamic_rnn"]
        rnn_layers = rnn_layers_base + rnn_layers_modular

        self.rnn_layer_names = rnn_layers

        self.rnn_state_shared = tuple(self.network_graph[layer + "_shared"] for layer in rnn_layers)
        self.rnn_state_in = tuple(self.rnn_cell_states[layer] for layer in rnn_layers)

        if reflected:
            self.rnn_state_ref = tuple(self.reflected_network_graph[layer + "_shared"] for layer in rnn_layers)
            self.rnn_state_in_ref = tuple(self.rnn_cell_states[layer + "_ref"] for layer in rnn_layers)

    def perform_op(self, op, network_graph, reflected):
        if op[0] == "eye_split":
            if reflected:
                # Swap eyes
                network_graph[op[2][0]] = tf.reverse(self.network_graph[op[2][1]], [1])
                network_graph[op[2][1]] = tf.reverse(self.network_graph[op[2][0]], [1])
            else:
                network_graph[op[2][0]] = network_graph[op[1][0]][:, :, :, 0]
                network_graph[op[2][1]] = network_graph[op[1][0]][:, :, :, 1]

        elif op[0] == "flatten":
            network_graph[op[2][0]] = tf.layers.flatten(network_graph[op[1][0]], name=self.scope + "_" + op[2][0])
        elif op[0] == "concatenate":
            if reflected:  # TODO: Get rid of in future, is compatability error..
                network_graph[op[2][0]] = tf.concat([network_graph[op[1][i]] for i in range(len(op[1]))], 1)
            else:
                network_graph[op[2][0]] = tf.concat([network_graph[op[1][i]] for i in range(len(op[1]))], 1,
                                                    name=op[2][0])
        else:
            print(f"Undefined op: {op[0]}")

    def create_layer(self, layer_name, layer_input, connection_type, layer_parameters, network_graph, reflected,
                     reuse_eyes):
        # TODO: build in different levels of connection

        if layer_parameters[0] == "conv1d":
            filters, kernel_size, strides = layer_parameters[1:]
            if reuse_eyes and layer_name[-1] == "r":
                # If set to reuse weights of one eye in the other, checks whether a right layer is specified, and if so,
                # renames it and tells it to reuse the weight.
                layer_name_ = layer_name[:-1] + "l"
                reflected = True
            else:
                layer_name_ = layer_name
            network_graph[layer_name] = tf.layers.conv1d(inputs=network_graph[layer_input],
                                                         filters=filters,
                                                         kernel_size=kernel_size, strides=strides, padding='valid',
                                                         activation=tf.nn.relu, name=self.scope + "_" + layer_name_,
                                                         reuse=reflected)
        elif layer_parameters[0] == "dense":
            units = layer_parameters[1]
            network_graph[layer_name] = tf.layers.dense(network_graph[layer_input], units,
                                                        activation=tf.nn.relu,
                                                        kernel_initializer=tf.orthogonal_initializer,
                                                        trainable=True, name=self.scope + "_" + layer_name,
                                                        reuse=reflected)

        elif layer_parameters[0] == "dynamic_rnn":
            # if reflected:
            #     state_layer_name = layer_name + "_ref"
            # else:
            #     state_layer_name = layer_name
            state_layer_name = layer_name
            units = layer_parameters[1]
            self.network_graph[layer_input + "_inputs"] = tf.layers.dense(network_graph[layer_input], units,
                                                                          activation=tf.nn.relu,
                                                                          kernel_initializer=tf.orthogonal_initializer,
                                                                          trainable=True,
                                                                          name=self.scope + "_" + layer_name + "_in",
                                                                          reuse=reflected)
            if reflected:  # TODO: Remove later, compatability issue now.
                network_graph[layer_input + "_reshaped"] = tf.reshape(self.network_graph[layer_input + "_inputs"],
                                                                      [self.batch_size, self.train_length, units])
            else:
                network_graph[layer_input + "_reshaped"] = tf.reshape(self.network_graph[layer_input + "_inputs"],
                                                                      [self.batch_size, self.train_length, units],
                                                                      name="flattened_shared_" + layer_name + "_in")
            network_graph[layer_name + "_dyn"], network_graph[layer_name + "_shared"] = tf.nn.dynamic_rnn(
                inputs=network_graph[layer_input + "_reshaped"], cell=self.rnn_cells[layer_name],
                dtype=tf.float32,
                initial_state=self.rnn_cell_states[state_layer_name],
                scope=self.scope + "_" + state_layer_name)
            if reflected:  # TODO: Remove later, compatability issue now.
                network_graph[layer_name] = tf.reshape(network_graph[layer_name + "_dyn"], shape=[-1, units])
            else:
                network_graph[layer_name] = tf.reshape(network_graph[layer_name + "_dyn"], shape=[-1, units],
                                                       name="shared_" + layer_name + "_output")
        else:
            print(f"Undefined layer: {layer_parameters[0]}")

    def separate_internal_state_inputs(self, reflected):
        if reflected:
            for i, state in enumerate(self.internal_state_names):
                self.reflected_network_graph[state] = self.reflected_network_graph["internal_state"][:, i]
        else:
            for i, state in enumerate(self.internal_state_names):
                self.network_graph[state] = self.network_graph["internal_state"][:, i]

    def initialize_network(self, layers, modular_layers, ops, connectivity, network_graph, reflected=False,
                           reuse_eyes=False):
        """reuse_eyes, if true, causes all conv layers to be reused."""

        self.separate_internal_state_inputs(reflected)
        layers = {**layers, **modular_layers}
        final_name = None
        while True:

            op_to_remove = None
            for op in ops.keys():
                if set(ops[op][1]).issubset(list(network_graph.keys())):
                    self.perform_op(ops[op], network_graph, reflected)
                    final_name = ops[op][2]
                    op_to_remove = op
                    break
            if op_to_remove is not None:
                del ops[op_to_remove]

            layer_to_remove = None
            break_out = False
            for layer in layers.keys():
                for connection in connectivity:
                    if connection[1][1] == layer:
                        if connection[1][0] in list(network_graph.keys()):
                            self.create_layer(layer, connection[1][0], connection[0], layers[layer], network_graph,
                                              reflected, reuse_eyes)
                            layer_to_remove = layer
                            final_name = layer
                            break_out = True
                            break
                if break_out:
                    break
            if layer_to_remove is not None:
                del layers[layer_to_remove]

            if not layers and not ops:
                if not isinstance(final_name, str):
                    final_name = final_name[0]

                network_graph["output_layer"] = network_graph[final_name]
                return

    @staticmethod
    def resolve_connectivity(base_network_layers, modular_network_layers, ops, connectivity):
        """Through running through indicated connections, ensuring there are no impossible connections and that all
        referenced inputs are indicated. Also display warnings when layers of network have no input on output."""
        all_layer_names = list(base_network_layers.keys()) + list(modular_network_layers.keys()) + [item for l in
                                                                                                    [ops[v][2] for v in
                                                                                                     ops.keys()] for
                                                                                                    item
                                                                                                    in l]

        connectivity_required_layers = flatten_list([[v[1]] for v in connectivity])
        connectivity_required_layers = flatten_list(connectivity_required_layers)

        for con in connectivity_required_layers:
            if con not in all_layer_names:
                raise Exception(f"Error, connectivity requirements dont match given layers: {con} not defined")

    def create_rnns(self, base_network_layers, modular_network_layers, reflected):
        rnn_units = {}
        rnn_cell_states = {}
        rnn_dim = None  # TODO: allow multiple of these to exist...
        layers = {**base_network_layers, **modular_network_layers}

        for layer in layers.keys():
            if layers[layer][0] == "dynamic_rnn":
                rnn_units[layer] = tf.nn.rnn_cell.LSTMCell(num_units=layers[layer][1], state_is_tuple=True)
                rnn_cell_states[layer] = rnn_units[layer].zero_state(self.train_length, tf.float32)
                if reflected:
                    rnn_cell_states[layer + "_ref"] = rnn_units[layer].zero_state(self.train_length,
                                                                                  tf.float32)  # TODO: TEST CHANGE HERE
                rnn_dim = layers[layer][1]
        return rnn_units, rnn_cell_states, rnn_dim
