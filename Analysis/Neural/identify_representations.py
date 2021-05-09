from Analysis.Neural.calculate_vrv import create_full_response_vector, create_full_stimulus_vector


def identify_representations(response_vector, stimulus_vector, leeway=0.1):
    """Returns list of neurons, each with all representation labels attached."""
    basic_response, background_response = response_vector[:242], response_vector[242:]

    representations = []
    for basic_neuron, background_neuron in zip(basic_response, background_response):
        neuron_reps = []
        for stimulus, basic, background in zip(stimulus_vector, basic_neuron, background_neuron):
            if -leeway < basic-background < leeway:
                neuron_reps.append(stimulus)
        representations.append(neuron_reps)

    return representations


full_rv = create_full_response_vector("new_differential_prey_ref-5", background=True)
full_sv = create_full_stimulus_vector("new_differential_prey_ref-5",  background=False)  # False because only need half to compute.
reps = identify_representations(full_rv, full_sv)
x = True
