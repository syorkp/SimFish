import matplotlib.pyplot as plt
import numpy as np

from Analysis.Connectivity.load_network_variables import load_network_variables_dqn

#-------------------------------------------------
#Utility function for displaying filters as images
#-------------------------------------------------

def deprocess_image(x):

    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

#---------------------------------------------------------------------------------------------------
#Utility function for generating patterns for given layer starting from empty input image and then
#applying Stochastic Gradient Ascent for maximizing the response of particular filter in given layer
#---------------------------------------------------------------------------------------------------

def generate_pattern(layer_name, filter_index, params, size=150):

    filter, bias = params[layer_name + "/" + "kernel:0"], params[layer_name + "/" + "bias:0"]
    f_min, f_max = np.min(filter), np.max(filter)
    filter = (filter - f_min) / (f_max - f_min)
    reshaped_filter = np.swapaxes(filter, 1, 2)

    just_visible_parts_filter = np.concatenate((reshaped_filter[:, :, 0:1],
                                                np.zeros((reshaped_filter[:, :, 0:1].shape)),
                                                reshaped_filter[:, :, 1:2]), axis=2)

    plt.imshow(just_visible_parts_filter)
    plt.show()


    # layer_output = model.get_layer(layer_name).output
    # loss = K.mean(layer_output[:, :, :, filter_index])
    # grads = K.gradients(loss, model.input)[0]
    # grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    # iterate = K.function([model.input], [loss, grads])
    # input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    # step = 1.
    # for i in range(80):
    #     loss_value, grads_value = iterate([input_img_data])
    #     input_img_data += grads_value * step
    #
    # img = input_img_data[0]
    # return deprocess_image(img)

#------------------------------------------------------------------------------------------
#Generating convolution layer filters for intermediate layers using above utility functions
#------------------------------------------------------------------------------------------

# params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1")
#
# layer_names = ['main_conv1l', 'main_conv2l', 'main_conv3l', 'main_conv4l',
#                'main_conv1r', 'main_conv2r', 'main_conv3r', 'main_conv4r']

# for layer in layer_names:
#     generate_pattern(layer, 0, params)
# size = 299
# margin = 5
# results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

# for i in range(8):
#     for j in range(8):
#         filter_img = generate_pattern(layer_name, i + (j * 8), params, size=size)
#         horizontal_start = i * size + i * margin
#         horizontal_end = horizontal_start + size
#         vertical_start = j * size + j * margin
#         vertical_end = vertical_start + size
#         results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img
#
# plt.figure(figsize=(20, 20))
# plt.savefig(results)

def compute_filters(kernels, biases, input_filter, strides=4, activation="relu"):
    """
    Kernels have the following dimensions: (filters, channel_number, kernel_size)
    :return:
    """
    input_size = input_filter.shape[0]
    # Create new filter.

    for layer_num in range(len(kernels)):
        new_filter = np.zeros((input_size, 3, kernels[layer_num].shape[-1]))

        for unit in range(kernels[layer_num].shape[-1]):
            kernel_size = kernels[layer_num].shape[0]
            # Convolution of kernel across input
            for i in range(0, input_size, strides):
                if len(new_filter[i:i + kernel_size, 0]) == kernel_size:
                    for c in range(input_filter.shape[1]):
                        if layer_num == 0:
                            new_filter[i:i + kernel_size, c, unit] += input_filter[i:i+kernel_size, c] * kernels[layer_num][:, c, unit]
                        else:
                            input_array = input_filter[i:i+kernel_size, :, c]
                            kernel_array = kernels[layer_num][:, c:c+1, unit]
                            element_by_element_mul = input_array * kernel_array
                            new_filter[i:i + kernel_size, :, unit] += element_by_element_mul
            # Add the bias
            new_filter[:, :, unit] += biases[layer_num][unit]
            # Apply activation function
            if activation == "relu":
                new_filter = np.maximum(new_filter, 0)
            else:
                print("ERROR, activation function not recognised.")

        # Scaling for full range display
        scaling_factor = 1/np.max(new_filter)
        new_filter *= scaling_factor

        input_filter = new_filter

    return new_filter

def display_cnn_filters(kernels, biases, input_size=100):
    """NOTE: Must supply layer weights in correct order, from first to last."""
    # Starting filter
    input_filter = np.ones((input_size, 3))

    for layer_num in range(len(kernels)):
        rel_kernels = kernels[:layer_num+1]
        rel_biases = biases[:layer_num + 1]
        computed_filter = compute_filters(rel_kernels, rel_biases, input_filter)
        computed_filter = np.swapaxes(computed_filter, 1, 2)
        computed_filter = np.swapaxes(computed_filter, 0, 1)
        computed_filter[:, :, 2] *= 0

        # Swap blue location so is RGB
        computed_filter = np.concatenate((computed_filter[:, :, 0:1], computed_filter[:, :, 2:3], computed_filter[:, :, 1:2]), axis=2)

        plt.imshow(computed_filter)
        plt.show()


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


if __name__ == "__main__":
    params = load_network_variables_dqn("dqn_scaffold_18-1", "dqn_18_1", full_reafference=True)
    k, b = get_conv_weights_and_biases(params, left=True)
    display_cnn_filters(k, b)

