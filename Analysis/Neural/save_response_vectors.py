import csv
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


from Analysis.load_stimuli_data import load_stimulus_data
from Analysis.load_data import load_data
from Analysis.Neural.calculate_vrv import get_all_neuron_vectors, get_stimulus_vector


data = load_data("large_all_features-1", "Controlled_Visual_Stimuli", "Curved_prey")
stimulus_data = load_stimulus_data("large_all_features-1", "Controlled_Visual_Stimuli")


stimulus_vector = get_stimulus_vector(stimulus_data, "prey 1")
all_vectors = get_all_neuron_vectors(data, "prey 1", stimulus_data, "rnn state")

# import collections
# counter = collections.defaultdict(int)
# for row in all_vectors:
#     counter[row[0]] += 1

writer = csv.writer(open("./vrvs.csv", "w"))
writer.writerow(["Neuron Num"] + stimulus_vector)
for i, row in enumerate(all_vectors):
    writer.writerow([f"Neuron {i}"] + row)



