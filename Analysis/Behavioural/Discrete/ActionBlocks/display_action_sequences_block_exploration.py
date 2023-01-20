from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import get_exploration_sequences
from Analysis.Behavioural.Discrete.ActionBlocks.display_action_sequences_block import display_all_sequences


if __name__ == "__main__":

    exploration_sequences = get_exploration_sequences("dqn_gamma-1", "Behavioural-Data-Empty", "Naturalistic", 100)
    exploration_sequences = exploration_sequences[:200]
    display_all_sequences(exploration_sequences, min_length=20, max_length=40, save_figure=True,
                          figure_name="Exploration_2-dqn_gamma-1",)

    exploration_sequences = get_exploration_sequences("dqn_gamma-4", "Behavioural-Data-Empty", "Naturalistic", 100)
    exploration_sequences = exploration_sequences[:200]
    display_all_sequences(exploration_sequences, min_length=20, max_length=40, save_figure=True,
                          figure_name="Exploration_2-dqn_gamma-4",)

    exploration_sequences = get_exploration_sequences("dqn_gamma-5", "Behavioural-Data-Empty", "Naturalistic", 100)
    exploration_sequences = exploration_sequences[:200]
    display_all_sequences(exploration_sequences, min_length=20, max_length=40, save_figure=True,
                          figure_name="Exploration_2-dqn_gamma-5",)

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

    #                            EXPLORATION SEQUENCES

    # exploration_sequences_26 = get_exploration_sequences(f"dqn_scaffold_26-2", "Behavioural-Data-NaturalisticA", "Naturalistic", 40)
    # display_all_sequences(exploration_sequences_26, min_length=20, max_length=200, save_figure=True,
    #                       figure_save_location="Exploration-dqn_scaffold_26-2",)
    #
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
