import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_action_name import get_action_name
from Analysis.Behavioural.Tools.extract_capture_sequences import extract_consumption_action_sequences_with_positions
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_positions


def plot_action_sequences_2D_discrete(fish_positions_compiled, actions_compiled, action_sequence_timestamps,
                                      consumption_timestamps=None, predation_death_timestamps=None):
    """Given position and action choice data, plots action sequences in 2D field."""

    fish_positions_compiled_flattened = np.concatenate(fish_positions_compiled, axis=0)
    fish_positions_compiled_flattened = np.reshape(fish_positions_compiled_flattened, (-1, 2))

    actions_compiled_flattened = np.concatenate(actions_compiled)
    actions_compiled_flattened = np.reshape(actions_compiled_flattened, (-1))

    action_sequence_timestamps_flattened = np.concatenate(action_sequence_timestamps)
    action_sequence_timestamps_flattened = np.reshape(action_sequence_timestamps_flattened, (-1))

    plt.figure(figsize=(10, 10))
    action_plot = plt.scatter(fish_positions_compiled_flattened[:, 0], fish_positions_compiled_flattened[:, 1], c=actions_compiled_flattened)

    action_names = set([get_action_name(a) for a in actions_compiled_flattened])
    plt.legend(handles=action_plot.legend_elements()[0], labels=action_names)

    if consumption_timestamps is not None:
        consumption_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in consumption_timestamps]
        consumption_plot = plt.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="r", marker="x")
    if predation_death_timestamps is not None:
        predation_death_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in predation_death_timestamps]
        predation_plot = plt.scatter([p[0] for p in predation_death_positions], [p[1] for p in predation_death_positions], color="r", marker="X")

    plt.show()


data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-1")

# Capture sequences
ts, consumption_timestamps, capture_sequences, capture_fish_positions = extract_consumption_action_sequences_with_positions(data)
plot_action_sequences_2D_discrete(capture_fish_positions, capture_sequences, ts, consumption_timestamps=consumption_timestamps)

# Exploration sequences
exploration_timestamps, exploration_sequences, exploration_fish_positions = extract_exploration_action_sequences_with_positions(data)
plot_action_sequences_2D_discrete(exploration_fish_positions, exploration_sequences, exploration_timestamps)
