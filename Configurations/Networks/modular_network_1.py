"""
File for example configuration of simple network.
"""

# Whether network is repeated as a reflected version - Currently must be true
reflected = True

# Layers in the base network: Name: [type, parameters]
base_network_layers = {
    "conv1l": ['conv1d', 16, 16, 4],  # Filters, kernel_size, strides.
    "conv2l": ['conv1d', 8, 8, 2],
    "conv3l": ['conv1d', 8, 4, 1],
    "conv4l": ['conv1d', 64, 4, 1],

    "conv1r": ['conv1d', 16, 16, 4],
    "conv2r": ['conv1d', 8, 8, 2],
    "conv3r": ['conv1d', 8, 4, 1],
    "conv4r": ['conv1d', 64, 4, 1],

    "rgc": ["dense", 512],  # Units
}

# Layers in the modular network - Programmatic differentiation is that these layers may be changed during trials.
modular_network_layers = {
    "optic_tectum": ["dynamic_rnn", 512],

}

# Non-processing operations performed on network, with organisation Name: [operation, [required layers], [output layers]]
ops = {
    "eye_split": ["eye_split", ["observation"], ["left_eye", "right_eye"]],
    "flatten1": ["flatten", ["conv4l"], ["conv4l_flat"]],
    "flatten2": ["flatten", ["conv4r"], ["conv4r_flat"]],
    "join_eyes": ["concatenate", ["conv4l_flat", "conv4r_flat"], ["conv_joined"]],
    "rnn_inputs": ["concatenate", ["internal_state", "prev_actions", "conv_joined"], ["rnn_inputs"]],
    "final_layer_inputs": ["concatenate", ["conv_joined", "optic_tectum"], ["final_layer_inputs"]],
}

# Types of connectivity between layers
connectivity = [
    ["full", ["left_eye", "conv1l"]],
    ["full", ["conv1l", "conv2l"]],
    ["full", ["conv2l", "conv3l"]],
    ["full", ["conv3l", "conv4l"]],

    ["full", ["right_eye", "conv1r"]],
    ["full", ["conv1r", "conv2r"]],
    ["full", ["conv2r", "conv3r"]],
    ["full", ["conv3r", "conv4r"]],

    ["full", ["rnn_inputs", "optic_tectum"]],
    ["full", ["final_layer_inputs", "rgc"]],
]
