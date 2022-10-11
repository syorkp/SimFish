import copy
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale

from Analysis.load_data import load_data
from Analysis.load_model_config import load_configuration_files
from Tools.make_gif import make_gif
from Configurations.Networks.original_network import base_network_layers, ops, connectivity
from Tools.make_video import make_video


def get_num_layers_upstream(layer, connectivity_graph):
    for connection in connectivity_graph:
        if layer == connection[1][1]:
            return 1 + get_num_layers_upstream(connection[1][0], connectivity_graph)
    else:
        return 0


def convert_ops_to_graph(operations):
    connections = []
    for op in operations.keys():
        inputs = operations[op][1]
        outputs = operations[op][2]

        if len(inputs) == 1:
            for output in outputs:
                connections.append([op, [inputs[0], output]])
        else:
            for input in inputs:
                connections.append([op, [input, outputs[0]]])

    return connections


def self_normalise_network_activity(neural_activity):
    for unit in range(neural_activity.shape[1]):
        activity_points = neural_activity[:, unit]
        max_activity = np.max(activity_points)
        min_activity = np.min(activity_points)
        difference = max_activity - min_activity
        neural_activity[:, unit] *= 255 / difference
    return neural_activity


def rejig_layers_upstream(layers_upstream):
    layers_upstream2 = copy.copy(layers_upstream)
    for i, l in enumerate(layers_upstream):
        if l - 1 in layers_upstream or l == 0:
            pass
        else:
            layers_upstream2[i] -= 1
            return rejig_layers_upstream(layers_upstream2)
    return layers_upstream


def tidy_layers_upstream(layers_upstream):
    min_layers_upstream = min(layers_upstream)
    layers_upstream = [layer - min_layers_upstream for layer in layers_upstream]
    return rejig_layers_upstream(layers_upstream)


def create_network_video(neural_data, connectivity_graph, model_name, scale=0.25, save_id="placeholder", s_per_frame=0.03):
    layer_space = 100

    layers = neural_data.keys()
    layer_widths = []
    layers_upstream = []
    for i, layer in enumerate(layers):
        units = neural_data[layer].shape[1:]
        units = units[0]
        layer_widths.append(units)
        t = neural_data[layer].shape[0]
        num_upstream = get_num_layers_upstream(layer, connectivity_graph)
        layers_upstream.append(num_upstream)

    layer_order = tidy_layers_upstream(layers_upstream)

    width = max(layer_widths)
    height = len(set(layer_order)) * layer_space

    base_display = np.zeros((t, height, width, 3))

    for l, layer in enumerate(layers):
        position = layer_order[l]
        num_units = layer_widths[l]

        in_parallel = layer_order.count(position)
        first_occurrence = True
        if in_parallel == 2:
            available_width = (width/2) - 2
            in_parallel = True
            positions_up_to = layer_order[:l + 1]
            if positions_up_to.count(position) == 2:
                first_occurrence = False
        else:
            available_width = width
            in_parallel = False

        h_index = int((position * layer_space) + layer_space / 2)

        if in_parallel:
            buffer_each_side = available_width - (available_width // num_units) * num_units # int(((width / 2) - num_units) / 2)
        else:
            buffer_each_side = 0#int((width - num_units) / 2)

        if first_occurrence:
            w_index = int(buffer_each_side)
        else:
            w_index = int((width / 2) + buffer_each_side)

        layer_neural_data = neural_data[layer]
        if "eye" not in layer:
            layer_neural_data = np.expand_dims(layer_neural_data, len(layer_neural_data.shape))
            layer_neural_data = np.repeat(layer_neural_data, 3, axis=len(layer_neural_data.shape) - 1)
        else:
            layer_neural_data = np.concatenate((self_normalise_network_activity(layer_neural_data[:, :, 0:1]),
                                                np.zeros((layer_neural_data.shape[0], layer_neural_data.shape[1], 1)),
                                                self_normalise_network_activity(layer_neural_data[:, :, 1:2])), axis=2)

        num_repeats = available_width // num_units

        if len(layer_neural_data.shape) == 3:
            layer_neural_data = self_normalise_network_activity(layer_neural_data)
            thickness_per_unit = np.clip((layer_space / 2), 1, 1000).astype(int)
            layer_neural_data = np.expand_dims(layer_neural_data, 1)
            layer_neural_data = np.tile(layer_neural_data, (1, thickness_per_unit, 1, 1))

            if num_repeats > 1:
                layer_neural_data = np.repeat(layer_neural_data, num_repeats, axis=2)

            start_index_h = h_index
            end_index_h = h_index + thickness_per_unit

            start_index_w = w_index
            end_index_w = int(w_index + num_repeats * num_units)

            base_display[:, start_index_h: end_index_h, start_index_w: end_index_w, :] = layer_neural_data

        else:
            in_layers = neural_data[layer].shape[-1]
            thickness_per_unit = np.clip((layer_space / 2) / in_layers, 1, 1000).astype(int)

            for sub_layer in range(in_layers):
                desired_data = layer_neural_data[:, :, sub_layer]
                desired_data = self_normalise_network_activity(desired_data)
                desired_data = np.expand_dims(desired_data, 1)
                desired_data = np.tile(desired_data, (1, thickness_per_unit, 1, 1))

                if num_repeats > 1:
                    desired_data = np.repeat(desired_data, num_repeats, axis=2)

                start_index_w = w_index
                end_index_w = int(w_index + num_repeats * num_units)

                start_index_h = h_index + (sub_layer * thickness_per_unit)
                end_index_h = h_index + ((sub_layer+1) * thickness_per_unit)
                base_display[:, start_index_h:end_index_h, start_index_w: end_index_w, :] = desired_data

        if in_parallel:
            base_display[:, int(position * layer_space): int((position * layer_space) + layer_space),
                         int(width / 2 - 2):int(width / 2 + 2)] = (0, 100, 0)

    frames = []
    for i in range(base_display.shape[0]):
        frames.append(rescale(base_display[i], scale, multichannel=True, anti_aliasing=True))

    make_video(frames, f"{model_name}-{save_id}-neural_activity.mp4", duration=len(frames) * s_per_frame, true_image=True)


if __name__ == "__main__":

    # model_name = "parameterised_speed_test_fast-1"
    # model_name = "scaffold_version_4-4"
    model_name = "dqn_scaffold_26-2"
    learning_params, environment_params, base_network_layers, ops, connectivity = load_configuration_files(model_name)

    data = load_data(model_name, "Behavioural-Data-Videos-CONV", "Naturalistic-2")
    base_network_layers["rnn_state_actor"] = base_network_layers["rnn"]
    del base_network_layers["rnn"]
    network_data = {key: data[key] for key in list(base_network_layers.keys())}
    network_data["left_eye"] = data["observation"][:, :, :, 0]
    network_data["right_eye"] = data["observation"][:, :, :, 1]
    network_data["internal_state"] = np.concatenate((np.expand_dims(data["energy_state"], 1),
                                                     np.expand_dims(data["salt"], 1)), axis=1)

    ops = convert_ops_to_graph(ops)
    create_network_video(network_data, connectivity + ops, model_name)

    # plt.plot(range(500), data["internal_state"][:, 0])
    # plt.xlabel("Step")
    # plt.ylabel("Energy Level")
    # plt.show()


