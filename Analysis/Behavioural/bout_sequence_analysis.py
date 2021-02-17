import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from Analysis.Behavioural.bout_transition_probabilities import get_third_order_transition_counts_from_sequences, get_transition_probabilities, get_second_order_transition_counts, get_third_order_transition_counts
from Analysis.Behavioural.extract_event_action_sequence import get_escape_sequences, get_capture_sequences


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


# From sequences:

capture_sequences = get_capture_sequences("changed_penalties-1", "Naturalistic", f"Naturalistic", 2)
escape_sequences = get_escape_sequences("changed_penalties-1", "Predator", f"Predator", 4)

capture_counts = get_third_order_transition_counts_from_sequences(capture_sequences)
escape_counts = get_third_order_transition_counts_from_sequences(escape_sequences)

capture_probabilities = get_transition_probabilities(capture_counts)
escape_probabilities = get_transition_probabilities(escape_counts)

all_probabilities = capture_probabilities + escape_probabilities


plot_tsne_results(capture_probabilities, 3, "Prey Capture")
plot_tsne_results(escape_probabilities, 3, "Predator Avoidance")
plot_tsne_results(all_probabilities, 3, "All Actions")

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


