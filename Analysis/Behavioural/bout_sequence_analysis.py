import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from Analysis.Behavioural.bout_transition_probabilities import get_third_order_transition_counts_from_sequences, create_third_order_transition_count_matrix, get_transition_probabilities, convert_probabilities_to_vectors, get_second_order_transition_counts, get_third_order_transition_counts
from Analysis.Behavioural.extract_event_action_sequence import get_escape_sequences, get_capture_sequences
from Analysis.load_data import load_data


def do_tsne(probability_vectors):
    tsne = TSNE(n_components=2, n_iter=250)
    tsne_results = tsne.fit_transform(probability_vectors)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    tpd['Point'] = ["Blue" for i in range(len(tsne_results[:, 0]))]
    for i in range(14,18):
        tpd["Point"][i] = "Red"
    plt.figure(figsize=(16, 10))
    plt.title("title")

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        # palette=sns.color_palette("hls", 10),
        hue="Point",
        data=tpd,
        legend="full",
        alpha=0.3
    )
    plt.show()


def plot_tsne_results(transition_probabilities, order=2, title="None"):
    while order > 1:
        transition_probabilities = np.vstack(transition_probabilities)
        order -= 1

    tsne = TSNE(n_components=2, n_iter=0)
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
a_data1 = create_third_order_transition_count_matrix(load_data("changed_penalties-1", "Naturalistic", "Naturalistic-1")["behavioural choice"])
a_data2 = create_third_order_transition_count_matrix(load_data("changed_penalties-1", "Naturalistic", "Naturalistic-2")["behavioural choice"])
a_data3 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-1")["behavioural choice"])
a_data4 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-2")["behavioural choice"])
a_data5 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-3")["behavioural choice"])
a_data6 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-4")["behavioural choice"])
a_data7 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-5")["behavioural choice"])
a_data8 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-6")["behavioural choice"])
a_data9 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-7")["behavioural choice"])
a_data10 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-8")["behavioural choice"])
a_data11 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-9")["behavioural choice"])
a_data12 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-10")["behavioural choice"])
a_data13 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-11")["behavioural choice"])
a_data14 = create_third_order_transition_count_matrix(load_data("large_all_features-1", "Naturalistic", "Naturalistic-12")["behavioural choice"])


data3 = create_third_order_transition_count_matrix(load_data("changed_penalties-1", "Predator", "Predator-1")["behavioural choice"])
data4 = create_third_order_transition_count_matrix(load_data("changed_penalties-1", "Predator", "Predator-2")["behavioural choice"])
data5 = create_third_order_transition_count_matrix(load_data("changed_penalties-1", "Predator", "Predator-3")["behavioural choice"])
data6 = create_third_order_transition_count_matrix(load_data("changed_penalties-1", "Predator", "Predator-4")["behavioural choice"])


data1_probabilities = get_transition_probabilities(a_data1)
data2_probabilities = get_transition_probabilities(a_data2)
data3_probabilities = get_transition_probabilities(a_data3)
data4_probabilities = get_transition_probabilities(a_data4)
data5_probabilities = get_transition_probabilities(a_data5)
data6_probabilities = get_transition_probabilities(a_data6)
data7_probabilities = get_transition_probabilities(a_data7)
data8_probabilities = get_transition_probabilities(a_data8)
data9_probabilities = get_transition_probabilities(a_data9)
data10_probabilities = get_transition_probabilities(a_data10)
data11_probabilities = get_transition_probabilities(a_data11)
data12_probabilities = get_transition_probabilities(a_data12)
data13_probabilities = get_transition_probabilities(a_data13)
data14_probabilities = get_transition_probabilities(a_data14)
data15_probabilities = get_transition_probabilities(data3)
data16_probabilities = get_transition_probabilities(data4)
data17_probabilities = get_transition_probabilities(data5)
data18_probabilities = get_transition_probabilities(data6)



probabilities = [data1_probabilities, data2_probabilities, data3_probabilities, data4_probabilities, data5_probabilities,
data6_probabilities,
                 data7_probabilities,
                 data8_probabilities,
                 data9_probabilities,
                 data10_probabilities,
                 data11_probabilities,
                 data12_probabilities,
                 data13_probabilities,
                 data14_probabilities,
                 data15_probabilities,
                 data16_probabilities,
                 data17_probabilities,
                 data18_probabilities,
                 ]


do_tsne(probabilities)

x = True

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


