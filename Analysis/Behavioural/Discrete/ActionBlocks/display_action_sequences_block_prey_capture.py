from Analysis.load_data import load_data

from Analysis.Behavioural.Discrete.ActionBlocks.display_action_sequences_block import display_all_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_capture_sequences import get_capture_sequences
from Analysis.Behavioural.Tools.filter_sequences import remove_sCS_heavy
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps

if __name__ == "__main__":


    d = load_data("dqn_beta-1", "Behavioural-Data-Free", "Naturalistic-2")
    seq = get_hunting_sequences_timestamps(d, True)
    display_all_sequences(seq, indicate_consumption=True, save_figure=True, figure_name="Captures-pedro_method-1")

    seq = get_hunting_sequences_timestamps(d, False)
    display_all_sequences(seq, indicate_consumption=True, save_figure=True, figure_name="Hunts-pedro_method-1")

    # capture_sequences = get_capture_sequences(f"dqn_beta-1", "Behavioural-Data-Free", "Naturalistic", 10)
    # display_all_sequences(capture_sequences, indicate_consumption=True, save_figure=True, figure_save_location="Captures-dqn_scaffold_33-1")
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
