from Analysis.Behavioural.Tools.BehavLabels.extract_event_action_sequence import get_escape_sequences
from Analysis.Behavioural.Discrete.ActionBlocks.display_action_sequences_block import display_all_sequences


if __name__ == "__main__":
    #                           PREDATOR AVOIDANCE SEQUENCES

    escape_sequences = get_escape_sequences("dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 100)
    display_all_sequences(escape_sequences, figure_name="Escapes-dqn_gamma-2", save_figure=True, max_length=25)

    escape_sequences = get_escape_sequences("dqn_gamma-4", "Behavioural-Data-Free", "Naturalistic", 100)
    display_all_sequences(escape_sequences, figure_name="Escapes-dqn_gamma-4", save_figure=True, max_length=25)

    escape_sequences = get_escape_sequences("dqn_gamma-5", "Behavioural-Data-Free", "Naturalistic", 100)
    display_all_sequences(escape_sequences, figure_name="Escapes-dqn_gamma-5", save_figure=True, max_length=25)


    # escape_sequences = get_escape_sequences("dqn_scaffold_20-2", "Behavioural-Data-Free", "Naturalistic", 40)
    # display_all_sequences(escape_sequences)
    # escape_sequences = get_escape_sequences("dqn_scaffold_21-2", "Behavioural-Data-Free", "Naturalistic", 40)
    # escape_sequences = remove_sCS_heavy(escape_sequences, 0)
    # display_all_sequences(escape_sequences, max_length=20, save_figure=True, figure_name="Avoidance-dqn_scaffold_21-2")
    # escape_sequences = get_escape_sequences("dqn_scaffold_22-1", "Behavioural-Data-Free", "Naturalistic", 40)
    # display_all_sequences(escape_sequences)


    # Combined across models
    # display_all_sequences(compiled_failed_capture_sequences)