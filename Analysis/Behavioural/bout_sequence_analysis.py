import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from Analysis.Behavioural.bout_transition_probabilities import get_transition_probabilities, get_second_order_transition_counts, get_third_order_transition_counts


def plot_tsne_results(transition_probabilities, order=2, title="None"):
    while order > 1:
        transition_probabilities = np.vstack(transition_probabilities)
        order -= 1

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(transition_probabilities)

    tpd = {}

    tpd['tsne-2d-one'] = tsne_results[:, 0]
    tpd['tsne-2d-two'] = tsne_results[:, 1]
    plt.figure(figsize=(16, 10))
    plt.title(title)

    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        palette=sns.color_palette("hls", 10),
        data=tpd,
        legend="full",
        alpha=0.3
    )
    plt.show()


t = get_third_order_transition_counts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)
t2 = get_third_order_transition_counts("changed_penalties-1", "Predator", "Predator", 4)

tp1 = get_transition_probabilities(t)
tp2 = get_transition_probabilities(t2)

tp_all = get_transition_probabilities(t + t2)

plot_tsne_results(tp1, 3, "Prey Capture")
plot_tsne_results(tp2, 3, "Predator Avoidance")

plot_tsne_results(tp_all, 3, "All Actions")


