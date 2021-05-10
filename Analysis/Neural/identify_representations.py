import numpy as np
import matplotlib.pyplot as plt

from Analysis.Neural.calculate_vrv import create_full_response_vector, create_full_stimulus_vector
from Analysis.Visualisation.visualise_response_vectors import display_full_response_vector, normalise_response_vectors, format_func_prey, format_func_pred


def identify_representations(response_vector, stimulus_vector, leeway=0.1):
    """Returns list of neurons, each with all representation labels attached."""
    basic_response, background_response = response_vector[:][:242], response_vector[:][242:]

    representations = []
    for basic_neuron, background_neuron in zip(basic_response, background_response):
        neuron_reps = []
        for stimulus, basic, background in zip(stimulus_vector, basic_neuron, background_neuron):
            if -leeway < basic-background < leeway:
                neuron_reps.append(stimulus)
        representations.append(neuron_reps)

    return representations


def compute_relative_vrv(response_vector):
    response_vector = np.array(response_vector)
    basic_response, background_response = response_vector[:, :int(len(response_vector)/2)], response_vector[:, int(len(response_vector)/2):]  # TODO: Correct to use proper dimensions
    relative_rv = []
    for i, neuron in enumerate(basic_response):
        neuron_vector = []
        for j, value in enumerate(neuron):
            if value > 0.5 or value <-0.5:
                neuron_vector.append(value-background_response[i, j])
            else:
                neuron_vector.append(1)
        relative_rv.append(neuron_vector)
    return relative_rv


def display_full_response_vector_relative(response_vector, stimulus_vector, title, transition_points=None):
    fig, ax = plt.subplots()
    # fig.set_size_inches(18.5, 80)
    fig.set_size_inches(18.5, 20)
    ax.set_title(title, fontsize=45)
    ax.pcolor(response_vector, cmap='RdYlBu')
    ax.grid(True, which='minor', axis='both', linestyle='-', color='k')
    ax.set_xlim(0, 121)
    ax.xaxis.set_major_locator(plt.MultipleLocator(11))
    if "Prey" in title:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_prey))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func_pred))

    ax.tick_params(labelsize=15)
    ax.set_xticks(range(0, len(stimulus_vector), 11), minor=True)
    if transition_points:
        transition_points = [0] + transition_points
        # cluster_labels = [i for i in range(len(transition_points))]

        def format_func_cluster(value, tick):
            for i, tp in enumerate(transition_points):
                if value < tp:
                    return i-1
            return len(transition_points) - 1

        ax.set_yticks(transition_points, minor=True)
        ax2 = ax.secondary_yaxis("right")
        ax2.yaxis.set_major_locator(plt.FixedLocator(transition_points))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func_cluster))
        ax2.set_ylabel("Cluster")

    ax.set_xlabel("Stimulus and Position", fontsize=35)
    ax.set_ylabel("Neuron", fontsize=35)
    ax.xaxis.grid(linewidth=1, color="black")
    # ax.xaxis._axinfo["grid"]['linewidth'] = 3.
    plt.show()


full_rv = create_full_response_vector("new_differential_prey_ref-5", background=True)
full_sv = create_full_stimulus_vector("new_differential_prey_ref-5",  background=False)
full_rv = normalise_response_vectors(full_rv)
# False because only need half to compute.
reps = identify_representations(full_rv, full_sv)
display_full_response_vector(full_rv, full_sv, "real vector")

full_rv = compute_relative_vrv(full_rv)
display_full_response_vector_relative(full_rv, full_sv, "relative vector")
x = True
