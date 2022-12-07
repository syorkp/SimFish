

def get_conv_weights_and_biases(params, left=True):
    if left:
        layer_names = ['main_conv1l', 'main_conv2l', 'main_conv3l', 'main_conv4l']
    else:
        layer_names = ['main_conv1r', 'main_conv2r', 'main_conv3r', 'main_conv4r']

    kernels = []
    biases = []
    for layer in layer_names:
        for key in params.keys():
            if layer in key:
                if "kernel" in key:
                    kernels.append(params[key])
                elif "bias" in key:
                    biases.append(params[key])
    return kernels, biases

