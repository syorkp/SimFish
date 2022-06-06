import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_action_name import get_action_name
from Analysis.Behavioural.Tools.extract_capture_sequences import extract_consumption_action_sequences_with_positions
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_positions


def plot_action_sequences_2D_discrete(fish_positions_compiled, actions_compiled, action_sequence_timestamps,
                                      consumption_timestamps=None, predation_death_timestamps=None):
    """Given position and action choice data, plots action sequences in 2D field."""

    fish_positions_compiled_flattened = np.concatenate(fish_positions_compiled, axis=0)
    fish_positions_compiled_flattened = np.reshape(fish_positions_compiled_flattened, (-1, 2))
    if len(fish_positions_compiled_flattened) == 0:
        return

    actions_compiled_flattened = np.concatenate(actions_compiled)
    actions_compiled_flattened = np.reshape(actions_compiled_flattened, (-1))

    action_sequence_timestamps_flattened = np.concatenate(action_sequence_timestamps)
    action_sequence_timestamps_flattened = np.reshape(action_sequence_timestamps_flattened, (-1))

    seen = set()
    seen_add = seen.add
    actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'y', "k", "m", "m", "black"]
    color_set = [color_set[a] for a in ordered_actions_present]
    colours = ListedColormap(color_set)

    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    plt.figure(figsize=(10, 10))
    action_plot = plt.scatter(fish_positions_compiled_flattened[:, 0], fish_positions_compiled_flattened[:, 1], c=actions_compiled_flattened, cmap=colours)
    plt.legend(handles=action_plot.legend_elements()[0], labels=associated_actions)

    if consumption_timestamps is not None:
        consumption_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in consumption_timestamps]
        consumption_plot = plt.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="black", marker="x")
    if predation_death_timestamps is not None:
        predation_death_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in predation_death_timestamps]
        predation_plot = plt.scatter([p[0] for p in predation_death_positions], [p[1] for p in predation_death_positions], color="r", marker="X")

    plt.show()


def plot_action_sequences_2D_discreteV2(fish_positions_compiled, actions_compiled, action_sequence_timestamps,
                                      consumption_timestamps=None, predation_death_timestamps=None):
    """Given position and action choice data, plots action sequences in 2D field."""

    fish_positions_compiled_flattened = np.concatenate(fish_positions_compiled, axis=0)
    fish_positions_compiled_flattened = np.reshape(fish_positions_compiled_flattened, (-1, 2))
    if len(fish_positions_compiled_flattened) == 0:
        return

    actions_compiled_flattened = np.concatenate(actions_compiled)
    actions_compiled_flattened = np.reshape(actions_compiled_flattened, (-1))

    action_sequence_timestamps_flattened = np.concatenate(action_sequence_timestamps)
    action_sequence_timestamps_flattened = np.reshape(action_sequence_timestamps_flattened, (-1))

    seen = set()
    seen_add = seen.add
    actions_present = [x for x in actions_compiled_flattened if not (x in seen or seen_add(x))]
    ordered_actions_present = sorted(actions_present)
    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    color_set = ['b', 'g', 'lightgreen', 'r', 'gold', 'y', "c", "m", "m", "black"]

    plt.figure(figsize=(10, 10))
    points = []
    for action in ordered_actions_present:
        colour = color_set[action]
        ts = np.array([i for i, a in enumerate(actions_compiled_flattened) if a==action])
        x = fish_positions_compiled_flattened[ts, 0]
        y = fish_positions_compiled_flattened[ts, 1]
        points.append(plt.scatter(x, y, c=colour))
    # plt.legend(handles=action_plot.legend_elements()[0], labels=associated_actions)
    plt.legend(points, associated_actions)

    if consumption_timestamps is not None:
        consumption_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in consumption_timestamps]
        consumption_plot = plt.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="black", marker="x")
    if predation_death_timestamps is not None:
        predation_death_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in predation_death_timestamps]
        predation_plot = plt.scatter([p[0] for p in predation_death_positions], [p[1] for p in predation_death_positions], color="r", marker="X")

    plt.show()



# Displays the exploration and capture sequences for the first trial for each model.
for i in range(1, 5):
    for j in range(1, 11):
        data = load_data(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", f"Naturalistic-{j}")

        # Capture sequences
        # ts, consumption_timestamps, capture_sequences, capture_fish_positions = extract_consumption_action_sequences_with_positions(data)
        # plot_action_sequences_2D_discrete(capture_fish_positions, capture_sequences, ts, consumption_timestamps=consumption_timestamps)

        # Exploration sequences
        exploration_timestamps, exploration_sequences, exploration_fish_positions = extract_exploration_action_sequences_with_positions(data)
        plot_action_sequences_2D_discreteV2(exploration_fish_positions, exploration_sequences, exploration_timestamps)
