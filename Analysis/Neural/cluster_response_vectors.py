import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector


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



data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")
stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli")


stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")


plot_tsne_results(capture_probabilities, 3, "Prey Capture")
plot_tsne_results(escape_probabilities, 3, "Predator Avoidance")
plot_tsne_results(all_probabilities, 3, "All Actions")






