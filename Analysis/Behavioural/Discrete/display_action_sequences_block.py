import matplotlib.pyplot as plt

from Analysis.load_data import load_data

from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import get_exploration_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_capture_sequences import get_capture_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_event_action_sequence import get_escape_sequences

from Analysis.Behavioural.Tools.filter_sequences import remove_sCS_heavy
from Analysis.Behavioural.VisTools.show_action_sequence_block import display_all_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_sand_grain_interaction_sequences import \
    get_sand_grain_engagement_sequences_multiple_trials
from Analysis.Behavioural.Tools.BehavLabels.extract_paramecia_interaction_sequences import \
    get_paramecia_engagement_sequences_multiple_trials


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


# VERSION 2
if __name__ == "__main__":
    capture_sequences = get_capture_sequences(f"dqn_scaffold_33-2", "Behavioural-Data-Free", "Naturalistic", 20)
    display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_name="Captures-dqn_33-1")

    #                            SAND GRAIN INTERACTION SEQUENCES
    # seq = get_sand_grain_engagement_sequences_multiple_trials("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic",
    #                                                           30, range_for_engagement=30, preceding_steps=20, proceeding_steps=10)
    # seq = [s[:50] for s in seq]
    # display_all_sequences(seq, min_length=50, max_length=60, save_figure=True, indicate_event_point=20,
    #                       figure_save_location="Sand-Grain-Interaction-dqn_scaffold_33-1",)
    #
    # #                            SAND GRAIN INTERACTION SEQUENCES (END)
    # seq = get_sand_grain_engagement_sequences_multiple_trials("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic",
    #                                                           30, range_for_engagement=30, preceding_steps=20, proceeding_steps=20)
    # seq = [s[-50:] for s in seq]
    # display_all_sequences(seq, min_length=20, max_length=60, save_figure=True, indicate_event_point=20,
    #                       figure_save_location="Sand-Grain-Interaction-END-dqn_scaffold_33-1",)

    #                            PREY INTERACTION SEQUENCES
    # seq = get_paramecia_engagement_sequences_multiple_trials("dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic",
    #                                                           30, range_for_engagement=20, preceding_steps=20, proceeding_steps=10)
    # seq = [s for s in seq if len(s) < 50]
    #
    # display_all_sequences(seq, min_length=30, max_length=60, save_figure=True, indicate_event_point=20,
    #                       figure_save_location="Paramecia-Interaction-dqn_scaffold_33-1",)

    #                            PREY CAPTURE SEQUENCES

    # capture_sequences = get_capture_sequences(f"dqn_scaffold_33-1", "Behavioural-Data-Free", "Naturalistic", 30)
    # display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_save_location="Captures-dqn_scaffold_33-1")

    #                            EXPLORATION SEQUENCES
    # capture_sequences_26 = get_capture_sequences(f"dqn_scaffold_26-2", "Behavioural-Data-NaturalisticA", "Naturalistic",
    #                                             40, dur=100)
    # display_all_sequences(capture_sequences_26, min_length=20, max_length=200, save_figure=True,
    #                       figure_save_location="Prey-Capture-dqn_scaffold_26-2",)
    #
    # exploration_sequences_26 = get_exploration_sequences(f"dqn_scaffold_26-2", "Behavioural-Data-NaturalisticA", "Naturalistic", 40)
    # display_all_sequences(exploration_sequences_26, min_length=20, max_length=200, save_figure=True,
    #                       figure_save_location="Exploration-dqn_scaffold_26-2",)
    #
    # escape_sequences_26 = get_escape_sequences(f"dqn_scaffold_26-2", "Behavioural-Data-NaturalisticA", "Naturalistic", 40)
    # display_all_sequences(escape_sequences_26, min_length=20, max_length=200, save_figure=True,
    #                       figure_save_location="Escape-dqn_scaffold_26-2",)

    # Exploration DQN 14
    # exploration_sequences_14 = get_exploration_sequences(f"dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic", 20)
    # # display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
    # #                       figure_name="Exploration-dqn_scaffold_14-1")
    #
    # # Exploration DQN 19
    # exploration_sequences_19 = get_exploration_sequences(f"dqn_scaffold_19-1", "Behavioural-Data-Free", "Naturalistic", 20)
    # exploration_sequences_19 = remove_sCS_heavy(exploration_sequences_19, max_sCS=6)
    # # display_all_sequences(exploration_sequences, min_length=15, max_length=42, save_figure=True,
    # #                       figure_name="Exploration-dqn_scaffold_19-1")
    #
    # # Exploration DQN 18
    # exploration_sequences_18 = get_exploration_sequences(f"dqn_scaffold_18-1", "Behavioural-Data-Free", "Naturalistic", 20)
    # # display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
    # #                       figure_name="Exploration-dqn_scaffold_18-1")
    #
    # # Exploration DQN_nl 19
    # exploration_sequences_nl = get_exploration_sequences(f"dqn_scaffold_nl_19-1", "Behavioural-Data-Free", "Naturalistic", 20)
    # exploration_sequences_nl = remove_sCS_heavy(exploration_sequences_nl, max_sCS=6)
    # # display_all_sequences(exploration_sequences, min_length=23, max_length=42, save_figure=True,
    # #                       figure_name="Exploration-dqn_scaffold_nl_19-1")



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

