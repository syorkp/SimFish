import numpy as np

from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import get_exploration_sequences, get_exploration_timestamps
from Analysis.Behavioural.Discrete.ActionBlocks.display_action_sequences_block import display_all_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_photogradient_sequences import get_in_light_vs_dark_steps


def get_action_display_blocks_light_vs_dark_exploration(model_name, assay_config, assay_id, n):
    """Displays action blocks for dark vs light exploration/prey capture"""
    light_vs_dark_steps = get_in_light_vs_dark_steps(model_name, assay_config, assay_id, n)
    exploration_sequences = get_exploration_sequences(model_name, assay_config, assay_id, n, flatten=False)
    exploration_timestamps = get_exploration_timestamps(model_name, assay_config, assay_id, n, flatten=False)

    dark_exploration_sequences = []
    light_exploration_sequences = []
    for t, trial_exploration_timestamps in enumerate(exploration_timestamps):
        for s, seq in enumerate(trial_exploration_timestamps):
            if len(seq) == 0:
                pass
            else:
                if np.all((light_vs_dark_steps[t][seq]) == 1):
                    light_exploration_sequences.append(exploration_sequences[t][s])
                elif np.all((light_vs_dark_steps[t][seq]) == 0):
                    dark_exploration_sequences.append(exploration_sequences[t][s])
                else:
                    pass

    if len(light_exploration_sequences) > 1:
        display_all_sequences(light_exploration_sequences, min_length=20, max_length=200, save_figure=True,
                              figure_name=f"Exploration_in_light-{model_name}")
    if len(dark_exploration_sequences) > 1:
        display_all_sequences(dark_exploration_sequences, min_length=20, max_length=200, save_figure=True,
                              figure_name=f"Exploration_in_dark-{model_name}")


if __name__ == "__main__":
    get_action_display_blocks_light_vs_dark_exploration("dqn_scaffold_26-1", "Behavioural-Data-Free", "Naturalistic", 5)


