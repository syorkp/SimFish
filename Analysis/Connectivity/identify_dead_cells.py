from Analysis.Connectivity.load_network_variables import load_network_variables


def identify_dead_rnn_units(graph_variables):
    rnn_output_weights = graph_variables["Variable:0"]
    rnn_output_weights2 = graph_variables["Variable_1:0"]
    rnn_input_weights = graph_variables["main_rnn/lstm_cell/kernel:0"]
    low_value_neurons = []
    low_input_neurons = []
    for i, neuron in enumerate(rnn_output_weights):
        for a_weight in neuron:
            if -0.1 < a_weight < 0.1:
                low_value_neurons.append(i)
    for i, neuron in enumerate(rnn_output_weights2):
        for a_weight in neuron:
            if -0.1 < a_weight < 0.1:
                low_value_neurons.append(i+256)

    return low_value_neurons

sv = load_network_variables("large_all_features-1", "naturalistic")
lvn = identify_dead_rnn_units(sv)
x = True


# TODO: Note that ouptut weights dont indicate the existence of dead cells - those that are low value are only this way for individual actions..