import matplotlib.pyplot as plt
import seaborn as sns

from Analysis.Behavioural.Tools.extract_capture_sequences import get_capture_sequences
from Analysis.Behavioural.Tools.extract_exploration_sequences import get_exploration_sequences

from Analysis.Behavioural.Tools.show_action_sequence_block import display_all_sequences
from Analysis.Behavioural.Tools.extract_failed_capture_sequences import get_failed_capture_sequences
from Analysis.Behavioural.Tools.extract_event_action_sequence import get_escape_sequences


def display_average_sequence(sequences):
    plot_dim = max([len(seq) for seq in sequences])
    modal_sequence_1 = []
    modal_sequence_2 = []
    modal_sequence_3 = []
    color_set = ['b', 'g', 'g', 'r', 'y', 'y', "k", "m", "m"]
    for index in range(max([len(seq) for seq in sequences])-1):
        all_actions_at_index = [seq[len(seq) - index - 1] for seq in sequences if len(seq) > index+1]
        modal_1 = max(set(all_actions_at_index), key=all_actions_at_index.count)
        remaining = list(set(all_actions_at_index))
        remaining.remove(modal_1)
        modal_2 = max(set(remaining), key=all_actions_at_index.count)
        # remaining.remove(modal_2)
        # modal_3 = max(set(remaining), key=all_actions_at_index.count)
        modal_sequence_1.append(modal_1)
        modal_sequence_2.append(modal_2)
        # modal_sequence_3.append(modal_3)

    for i, a in enumerate(reversed((modal_sequence_1))):
        i = plot_dim - i
        plt.fill_between((i, i + 1), 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()

    for i, a in enumerate(reversed((modal_sequence_2))):
        i = plot_dim - i
        plt.fill_between((i, i + 1), 1, color=color_set[a])
    plt.axis("scaled")
    plt.show()
    #
    # for i, a in enumerate(reversed((modal_sequence_3))):
    #     i = plot_dim - i
    #     plt.fill_between((i, i + 1), 1, color=color_set[a])
    # plt.axis("scaled")
    # plt.show()


def remove_sCS_heavy(sequences, max_sCS=7):
    new_sequences = []
    for sequence in sequences:
        if list(sequence).count(3) > max_sCS:
            pass
        else:
            new_sequences.append(sequence)
    return new_sequences

# VERSION 2

# Exploration DQN 14
exploration_sequences_14 = get_exploration_sequences(f"dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20)
# display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
#                       figure_name="Exploration-dqn_scaffold_14-1")

# Exploration DQN 19
exploration_sequences_19 = get_exploration_sequences(f"dqn_scaffold_19-1", "Behavioural-Data-Free", "Naturalistic", 20)
exploration_sequences_19 = remove_sCS_heavy(exploration_sequences_19, max_sCS=6)
# display_all_sequences(exploration_sequences, min_length=15, max_length=42, save_figure=True,
#                       figure_name="Exploration-dqn_scaffold_19-1")

# Exploration DQN 18
exploration_sequences_18 = get_exploration_sequences(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20)
# display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
#                       figure_name="Exploration-dqn_scaffold_18-1")

# Exploration DQN_nl 19
exploration_sequences_nl = get_exploration_sequences(f"dqn_scaffold_nl_19-1", "Behavioural-Data-Free", "Naturalistic", 20)
exploration_sequences_nl = remove_sCS_heavy(exploration_sequences_nl, max_sCS=6)
# display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
#                       figure_name="Exploration-dqn_scaffold_nl_19-1")

# Comparing action usage across the models
import numpy as np
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


#                                   CAPTURE SEQUENCES
# compiled_capture_sequences = []
# capture_sequences = get_capture_sequences(f"dqn_scaffold_18x-1", "Behavioural-Data-Free", "Naturalistic", 10)
# display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=False, figure_name="Captures-dqn_scaffold_18a-1")
# capture_sequences = remove_sCS_heavy(capture_sequences)
# display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_name="Captures-dqn_scaffold_18a-1")
#
# capture_sequences = get_capture_sequences(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20)
# display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=False, figure_name="Captures-dqn_scaffold_18a-1")
# capture_sequences = remove_sCS_heavy(capture_sequences)
# display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_name="Captures-dqn_scaffold_18b-1")
#
# capture_sequences = get_capture_sequences(f"dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20)
# display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=False, figure_name="Captures-dqn_scaffold_18a-1")
# capture_sequences = remove_sCS_heavy(capture_sequences, max_sCS=4)
# display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_name="Captures-dqn_scaffold_14-1")

# for i in range(1, 2):
#     capture_sequences += get_capture_sequences(f"dqn_scaffold_18-{i}", "Behavioural-Data-Free", "Naturalistic", 20)
#     capture_sequences = remove_sCS_heavy(capture_sequences)
#     # Filter those with too many sCS
#
#     # For each model
#     display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_name="dqn_scaffold_18_captures")
#     # compiled_capture_sequences += capture_sequences

# Combined across models
# display_all_sequences(compiled_capture_sequences)

#                       EXPLORATION SEQUENCES
# compiled_exploration_sequences = []
# exploration_sequences = get_exploration_sequences(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20)
# display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
#                       figure_name="Exploration-dqn_scaffold_18-1")

# compiled_exploration_sequences += exploration_sequences

# display_all_sequences(compiled_exploration_sequences, min_length=50, max_length=150)

# Failed Capture Sequences
# compiled_failed_capture_sequences = []
# for i in range(1, 5):
#     capture_sequences = get_failed_capture_sequences(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", "Naturalistic", 10)
#     # For each model
#     display_all_sequences(capture_sequences)
#     compiled_failed_capture_sequences += capture_sequences


#                           PREDATOR AVOIDANCE SEQUENCES

# escape_sequences = get_escape_sequences("dqn_scaffold_20-1", "Behavioural-Data-Free", "Naturalistic", 40)
# display_all_sequences(escape_sequences)
# escape_sequences = get_escape_sequences("dqn_scaffold_20-2", "Behavioural-Data-Free", "Naturalistic", 40)
# display_all_sequences(escape_sequences)
# escape_sequences = get_escape_sequences("dqn_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 40)
# escape_sequences = remove_sCS_heavy(escape_sequences, 0)
# display_all_sequences(escape_sequences, max_length=20, save_figure=True, figure_name="Avoidance-dqn_scaffold_21-2")
# escape_sequences = get_escape_sequences("dqn_scaffold_22-1", "Behavioural-Data-Free", "Naturalistic", 40)
# display_all_sequences(escape_sequences)


# Combined across models
# display_all_sequences(compiled_failed_capture_sequences)

# VERSION 1

# tp = get_transition_probabilities("changed_penalties-1", "Naturalistic", "Naturalistic", 2, order=1)
# ms = get_modal_sequences(tp, order=5)
# fs_sequences = []
# for j in range(1, 4):
#     for i in range(1, 11):
#         data = load_data("new_differential_prey_ref-3", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
#         fs_sequences += get_free_swimming_sequences(data)
#
# fs_sequences = divide_sequences(fs_sequences)
# fs_sequences.sort(key=len)
#
# display_all_sequences_escape(fs_sequences[-50:])

# capture_sequences = get_capture_sequences("even_prey_ref-3", "Behavioural-Data-Free", "Prey", 10)
# display_all_sequences_capture(capture_sequences[:100])
#
# escape_sequences = get_escape_sequences("even_prey_ref-3", "Behavioural-Data-Free", "Predator", 10)
# display_all_sequences_escape(escape_sequences[:100])


# Ablation groups:
# capture_sequences1 = get_capture_sequences("new_even_prey_ref-4", "Ablation-Indiscriminate-even_prey_only", "Ablated-0", 3)
# capture_sequences2 = get_capture_sequences("new_even_prey_ref-1", "Ablation-Indiscriminate-even_prey_only", "Ablated-100", 3)
# escape_sequences1 = get_escape_sequences("new_even_prey_ref-1", "Ablation-Indiscriminate-even_naturalistic", "Ablated-0", 3)
# escape_sequences2 = get_escape_sequences("new_even_prey_ref-1", "Ablation-Indiscriminate-even_naturalistic", "Ablated-100", 3)
#
# display_all_sequences_escape(capture_sequences1)
# display_all_sequences_escape(capture_sequences2)
# display_all_sequences_escape(escape_sequences1)
# display_all_sequences_escape(escape_sequences2)



# transition_counts = get_fourth_order_transition_counts_from_sequences(capture_sequences)
# tp = compute_transition_probabilities(transition_counts)
# ms = get_modal_sequences(tp, 4)
# display_sequences(ms)
# display_all_sequences_capture(capture_sequences[:70])
# escape_sequences = get_escape_sequences("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)
# display_all_sequences_escape(escape_sequences[:70])
#
# escape_sequences = escape_sequences + get_escape_sequences("even_prey_ref-4", "Behavioural-Data-Free", "Predator", 1)
# # display_all_sequences_escape(escape_sequences[:70])
# escape_sequences = escape_sequences + get_escape_sequences("even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 1)
# display_all_sequences_escape(escape_sequences[:60])

# cs = get_capture_sequences("even_prey_ref-7", "Ablation-Test-Predator-Only", "Prey-Only-Ablated-100", 3)
# display_all_sequences_capture(cs)
#

# tp = get_transition_probabilities("new_even_prey_ref-4", "Behavioural-Data-Free", "Prey", 10, order=4)
# ms = get_modal_sequences(tp, order=4, number=40)
# display_sequences(ms)
#
# x = True


# cs = get_capture_sequences("new_differential_prey_ref-6", "Behavioural-Data-Free-1", "Naturalistic", 10)
# display_all_sequences_capture(cs)
# es = get_escape_sequences("new_differential_prey_ref-6", "Behavioural-Data-Free-1", "Naturalistic", 10)
# display_all_sequences_capture(es)
# cs = get_capture_sequences("new_even_prey_ref-4", "Behavioural-Data-Free", "Prey", 10)
# display_all_sequences_capture(cs)
# es = get_escape_sequences("new_even_prey_ref-4", "Behavioural-Data-Free", "Naturalistic", 10)
# display_all_sequences_capture(es)

# tp = get_transition_probabilities("new_even_prey_ref-1", "Behavioural-Data-Free", "Predator", 10, order=4)
# ms = get_modal_sequences(tp, order=4, number=40)
# display_sequences(ms)

