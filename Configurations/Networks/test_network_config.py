"""
File for example configuration of simple network.
"""

# Define inputs to network  - Not needed, each can be included consistently, and set to zero when not needed.
network_inputs = {
    "observation": (120, 3, 2),
    "internal_state": (4, 1),
    "previous_action": (1, 1),
}

# Whether network is repeated as a reflected version
reflected = True

# Layers in the base network
base_network_layers = {
    "conv1l": ['conv1d', 16, 16, 4],  # Type, filters, kernel_size, strides
    "conv2l": ['conv1d', 8, 8, 2],
    "conv3l": ['conv1d', 8, 4, 1],
    "conv4l": ['conv1d', 64, 4, 1],

    "conv1r": ['conv1d', 16, 16, 4],
    "conv2r": ['conv1d', 8, 8, 2],
    "conv3r": ['conv1d', 8, 4, 1],
    "conv4r": ['conv1d', 64, 4, 1],

    "rgc": ["dense", 512],
    "optic_tectum": ["dynamic_rnn", 512]
}

# Layers in the modular network
modular_network_layers = {

}

# Non-processing operations performed on network, with organisation (operation, [required layers], [output layers])
ops = [
    ["eye_split", ["observation"], ["left_eye", "right_eye"]],
    ["flatten", ["conv4l"], ["conv4l_flat"]],
    ["flatten", ["conv4r"], ["conv4r_flat"]],
    ["concatenate", ["conv4l", "conv4r"], ["conv_with_states"]],
]

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

    ["full", ["conv_with_states", "rgc"]],
    ["full", ["rgc", "optic_tectum"]],

]

