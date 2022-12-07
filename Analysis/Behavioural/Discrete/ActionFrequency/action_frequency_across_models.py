import matplotlib.pyplot as plt
import numpy as np

from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import get_exploration_sequences
from Analysis.Behavioural.Tools.filter_sequences import remove_sCS_heavy


def compare_action_uses_across_models(exploration_sequences_14, exploration_sequences_19, exploration_sequences_18, exploration_sequences_nl):
    actions_proportions = np.zeros((10, 4))
    actions_present = np.array([])

    for i, sequences in enumerate([exploration_sequences_14, exploration_sequences_19, exploration_sequences_18, exploration_sequences_nl]):
        flattened_sequences = np.concatenate((sequences))
        unique, counts = np.unique(flattened_sequences, return_counts=True)
        actions_present = np.concatenate((actions_present, unique))
        x = list(zip(unique, counts))
        for a in x:
            actions_proportions[int(a[0]), i] = a[1] / len(flattened_sequences)
    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]
    actions_present = list(set(actions_present))
    fig, ax = plt.subplots(figsize=(10, 10))
    for a in reversed(list(actions_present)):
        total = sum([actions_proportions[int(i)] for i in range(int(a + 1))])
        ax.bar([i for i in range(4)], total,  color=color_set[int(a)], width=0.4)

    plt.xticks([i for i in range(4)], ["Model 1", "Model 2", "Model 3", "Model 4"], fontsize=20)
    plt.ylabel("Proportion of Actions", fontsize=30)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="x", labelsize=25)
    ax.tick_params(axis="y", labelsize=20)
    # plt.legend([get_action_name(int(a)) for a in reversed(actions_present)])
    plt.savefig("../../Figures/Panels/Panel-5/Model-Action.png")
    plt.show()


exploration_sequences_14 = get_exploration_sequences(f"dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20)
exploration_sequences_19 = get_exploration_sequences(f"dqn_scaffold_19-1", "Behavioural-Data-Free", "Naturalistic", 20)
exploration_sequences_19 = remove_sCS_heavy(exploration_sequences_19, max_sCS=6)
exploration_sequences_18 = get_exploration_sequences(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20)
exploration_sequences_nl = get_exploration_sequences(f"dqn_scaffold_nl_19-1", "Behavioural-Data-Free", "Naturalistic", 20)
exploration_sequences_nl = remove_sCS_heavy(exploration_sequences_nl, max_sCS=6)


compare_action_uses_across_models(exploration_sequences_14, exploration_sequences_19, exploration_sequences_18,
                                  exploration_sequences_nl)
