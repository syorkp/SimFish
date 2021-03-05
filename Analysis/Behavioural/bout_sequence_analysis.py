import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from Analysis.Behavioural.bout_transition_probabilities import get_third_order_transition_counts_from_sequences, get_transition_probabilities, convert_probabilities_to_vectors, get_second_order_transition_counts, get_third_order_transition_counts
from Analysis.Behavioural.extract_event_action_sequence import get_escape_sequences, get_capture_sequences
from Analysis.load_data import load_data


def plot_tsne_results(transition_probabilities, order=2, title="None"):
    while order > 1:
        transition_probabilities = np.vstack(transition_probabilities)
        order -= 1

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(transition_probabilities)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    tpd['Point'] = ["Blue" for i in range(len(tsne_results[:, 0]))]
    tpd["Point"][0] = "Red"
    plt.figure(figsize=(16, 10))
    plt.title(title)

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # palette=sns.color_palette("hls", 10),
        hue="Point",
        data=tpd,
        legend="full",
        alpha=0.3
    )
    plt.show()


# Trials from data
data = load_data("changed_penalties-1", "Naturalistic", "Naturalistic-1")["behavioural choice"]
data2 = load_data("changed_penalties-1", "Naturalistic", "Naturalistic-2")["behavioural choice"]
data3 = load_data("changed_penalties-1", "Predator", "Predator-1")["behavioural choice"]
data4 = load_data("changed_penalties-1", "Predator", "Predator-2")["behavioural choice"]
data5 = load_data("changed_penalties-1", "Predator", "Predator-3")["behavioural choice"]
data6 = load_data("changed_penalties-1", "Predator", "Predator-4")["behavioural choice"]


data1_counts = get_third_order_transition_counts_from_sequences(data)
data2_counts = get_third_order_transition_counts_from_sequences(data2)
data3_counts = get_third_order_transition_counts_from_sequences(data3)
data4_counts = get_third_order_transition_counts_from_sequences(data4)
data5_counts = get_third_order_transition_counts_from_sequences(data5)
data6_counts = get_third_order_transition_counts_from_sequences(data6)


data1_probabilities = get_transition_probabilities(data1_counts)
data2_probabilities = get_transition_probabilities(data2_counts)
data3_probabilities = get_transition_probabilities(data3_counts)
data4_probabilities = get_transition_probabilities(data4_counts)
data5_probabilities = get_transition_probabilities(data5_counts)
data6_probabilities = get_transition_probabilities(data6_counts)

probabilities = [data1_probabilities, data2_probabilities, data3_probabilities, data4_probabilities, data5_probabilities,
data6_probabilities]

pvectors = convert_probabilities_to_vectors(probabilities)



# plot_tsne_results(capture_probabilities, 3, "Prey Capture")
# plot_tsne_results(escape_probabilities, 3, "Predator Avoidance")
# plot_tsne_results(all_probabilities, 3, "All Actions")

# Getting directly from files:

# t = get_third_order_transition_counts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)
# t2 = get_third_order_transition_counts("changed_penalties-1", "Predator", "Predator", 4)
#
# tp1 = get_transition_probabilities(t)
# tp2 = get_transition_probabilities(t2)
#
# tp_all = get_transition_probabilities(t + t2)
#
# plot_tsne_results(tp1, 3, "Prey Capture")
# plot_tsne_results(tp2, 3, "Predator Avoidance")
#
# plot_tsne_results(tp_all, 3, "All Actions")


