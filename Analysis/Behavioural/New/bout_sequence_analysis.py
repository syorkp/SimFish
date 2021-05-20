import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from Analysis.Behavioural.New.bout_transition_probabilities import get_third_order_transition_counts_from_sequences, \
    create_third_order_transition_count_matrix, compute_transition_probabilities, convert_probabilities_to_vectors, \
    get_second_order_transition_counts, get_third_order_transition_counts
from Analysis.Behavioural.New.extract_event_action_sequence import get_escape_sequences, get_capture_sequences
from Analysis.load_data import load_data


def do_tsne(probability_vectors, groups):
    for it in range(250, 650, 100):
        for p in range(1, 20, 2):
            tsne = TSNE(n_components=2, perplexity=p, n_iter=it)
            tsne_results = tsne.fit_transform(probability_vectors)

            tpd = {}

            tpd['tsne-2d-one'] = tsne_results[:, 0]
            tpd['tsne-2d-two'] = tsne_results[:, 1]
            tpd['Point'] = ["Blue" for i in range(len(tsne_results[:, 0]))]
            for i in range(len(tsne_results[:, 0])):
                tpd["Point"][i] = groups[i]
            plt.figure(figsize=(16, 10))
            plt.title(f"it {it} p {p}")

            sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                # palette=sns.color_palette("hls", 10),
                hue="Point",
                data=tpd,
                legend="full",
                alpha=1
            )
            plt.show()


action_choices_prey = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-4", "Behavioural-Data-Free", f"Prey-{i}")["behavioural choice"]) for i in range(1, 11)]
action_choices_predator = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-4", "Behavioural-Data-Free", f"Predator-{i}")["behavioural choice"]) for i in range(1, 11)]

action_choices_prey_2 = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-5", "Behavioural-Data-Free", f"Prey-{i}")["behavioural choice"]) for i in range(1, 11)]
action_choices_predator_2 = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-5", "Behavioural-Data-Free-Predator", f"Predator-{i}")["behavioural choice"]) for i in
                             range(1, 11)]

action_choices_prey_slow = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-4", "Behavioural-Data-Free", f"Prey-Slow-{i}")["behavioural choice"]) for i in range(1, 3)]
action_choices_prey_no_jump = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-4", "Behavioural-Data-Free", f"Prey-No-Jump-{i}")["behavioural choice"]) for i in
                               range(1, 3)]
action_choices_prey_differential = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-4", "Behavioural-Data-Free", f"Prey-Differential-{i}")["behavioural choice"]) for i in
                                    range(1, 3)]
action_choices_prey_sparse = [create_third_order_transition_count_matrix(
    load_data("even_prey_ref-4", "Behavioural-Data-Free", f"Prey-Low-Density-{i}")["behavioural choice"]) for i in
                              range(1, 3)]

probabilities = [compute_transition_probabilities(ac) for ac in
                 action_choices_prey + action_choices_predator + action_choices_prey_2 + action_choices_predator_2 + action_choices_prey_slow + action_choices_prey_no_jump +
                 action_choices_prey_differential +
                 action_choices_prey_sparse]
groups = ["1-Prey" for i in range(10)] + \
         ["1-Predator" for i in range(10)] + \
         ["2-Prey" for i in range(10)] + \
         ["2-Predator" for i in range(10)] + \
         ["1-Prey-Slow" for i in range(2)] + \
         ["1-Prey-No-Jump" for i in range(2)]+ \
         ["1-Prey-Differential" for i in range(2)]+\
         ["1-Prey-Sparse" for i in range(2)]


do_tsne(probabilities, groups)
