from Analysis.load_data import load_data

from Analysis.Behavioural.Discrete.ActionBlocks.display_action_sequences_block import display_all_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_hunting_sequences import get_hunting_sequences_timestamps, get_hunting_sequences

if __name__ == "__main__":
    # All sequences
    hunting_sequences = get_hunting_sequences(f"dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 20,
                                              successful_captures=False, include_subsequent_action=True)
    display_all_sequences(hunting_sequences, indicate_consumption=True, save_figure=True,
                          figure_name="dqn_gamma_2-hunting_sequences_all")

    # Only successful sequences
    hunting_sequences = get_hunting_sequences(f"dqn_gamma-2", "Behavioural-Data-Free", "Naturalistic", 20,
                                              successful_captures=True, include_subsequent_action=True)
    display_all_sequences(hunting_sequences, indicate_consumption=True, save_figure=True,
                          figure_name="dqn_gamma_2-hunting_sequences_successful")

