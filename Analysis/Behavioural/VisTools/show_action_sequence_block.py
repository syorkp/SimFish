import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

from Analysis.Behavioural.VisTools.get_action_name import get_action_name


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


def display_all_sequences(sequences, min_length=None, max_length=None, indicate_consumption=False, save_figure=False,
                          figure_save_location=None, alternate_action_names=None):
    if len(sequences) == 0:
        return
    sns.set()
    sequences.sort(key=len)
    plot_dim = max([len(seq) for seq in sequences])
    if alternate_action_names is None:
        color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]
    else:
        color_set = ["b", "g", "r", "y", "c", "m", "black"]

    seen = set()
    seen_add = seen.add
    actions_compiled_flattened = np.concatenate(np.array(sequences))
    actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)
    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    # plt.figure(figsize=(5, 15))
    fig, ax = plt.subplots(figsize=(10, 10))
    used_sequences = []
    for i, seq in enumerate(sequences):
        if min_length is not None:
            if len(seq) < min_length:
                continue
        if max_length is not None:
            if len(seq) > max_length:
                for j, a in enumerate(reversed(seq[:max_length])):
                    j = plot_dim - j
                    ax.fill_between((j, j + 1), i, i + 1, color=color_set[a])
                    used_sequences.append(seq)

                continue
        for j, a in enumerate(reversed(seq)):
            j = plot_dim - j
            ax.fill_between((j, j+1), i, i+1, color=color_set[a])
        used_sequences.append(seq)

    seen = set()
    seen_add = seen.add
    actions_compiled_flattened = np.concatenate(np.array(used_sequences))
    actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)
    if alternate_action_names is not None:
        associated_actions = alternate_action_names
        legend_elements = [Patch(facecolor=color_set[int(a)], label=associated_actions[int(a)]) for i, a in enumerate(ordered_actions_present)]# [0], [0], marker="o", color=color_set[i], label=associated_actions[i]) for i in actions_present]
    else:
        associated_actions = [get_action_name(a) for a in ordered_actions_present]
        legend_elements = [Patch(facecolor=color_set[a], label=associated_actions[i]) for i, a in enumerate(ordered_actions_present)]# [0], [0], marker="o", color=color_set[i], label=associated_actions[i]) for i in actions_present]

    # if indicate_consumption:
    #     # outline = plt.Polygon(np.array([[plot_dim, 0], [plot_dim+1, 0], [plot_dim+1, len(used_sequences)], [plot_dim, len(used_sequences)]]))
    #     # ax.add_patch(outline)
    #     plt.arrow(plot_dim, 0, 0, 1, width=4, color="r")

    # plt.legend(legend_elements, associated_actions, bbox_to_anchor=(1.5, 1), borderaxespad=0)#loc='upper right')
    plt.legend(legend_elements, associated_actions, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
    plt.axis("scaled")
    if save_figure:
        plt.savefig(figure_save_location)
        plt.clf()
        plt.close()
    else:
        plt.show()

    # plt.tight_layout()
