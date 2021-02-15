import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

from Analysis.Behavioural.bout_transition_probabilities import get_transition_probabilities, get_second_order_transition_counts


t = get_second_order_transition_counts("changed_penalties-1", "Naturalistic", "Naturalistic", 2)
tp = get_transition_probabilities(t)
tp2 = np.vstack(tp)

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(tp2)

tpd = {}

tpd['tsne-2d-one'] = tsne_results[:, 0]
tpd['tsne-2d-two'] = tsne_results[:, 1]
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    data=tpd,
    legend="full",
    alpha=0.3
)
plt.show()



