import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from Analysis.Behavioural.Tools.get_action_name import get_action_name


def display_sequences(sequences):
    plot_dim = len(sequences[0])
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'y', "k", "m", "m", "k"]
    plt.figure(figsize=(5, 15))
    for i, seq in enumerate(reversed(sequences)):
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j + 1), i, i + 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()


def display_all_sequences(sequences, min_length=None, max_length=None):
    sns.set()
    sequences.sort(key=len)
    plot_dim = max([len(seq) for seq in sequences])
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]

    seen = set()
    seen_add = seen.add
    actions_compiled_flattened = np.concatenate(np.array(sequences))
    actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)
    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    # plt.figure(figsize=(5, 15))
    for i, seq in enumerate(sequences):
        if min_length is not None:
            if len(seq) < min_length:
                continue
        if max_length is not None:
            if len(seq) > max_length:
                continue
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            plt.fill_between((j, j+1), i, i+1, color=color_set[a])

    legend_elements = [Patch(facecolor=color_set[a], label=associated_actions[a]) for i, a in enumerate(ordered_actions_present)]# [0], [0], marker="o", color=color_set[i], label=associated_actions[i]) for i in actions_present]

    # plt.legend(legend_elements, associated_actions, bbox_to_anchor=(1.5, 1), borderaxespad=0)#loc='upper right')
    plt.legend(legend_elements, associated_actions, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.axis("scaled")
    # plt.tight_layout()
    plt.show()
