import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.get_action_name import get_action_name
from Analysis.Behavioural.Tools.extract_capture_sequences import extract_consumption_action_sequences_with_positions
from Analysis.Behavioural.Tools.extract_escape_sequences import extract_escape_action_sequences_with_positions
from Analysis.Behavioural.Tools.extract_exploration_sequences import extract_exploration_action_sequences_with_positions
from Analysis.Behavioural.Tools.anchored_scale_bar import AnchoredHScaleBar


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

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "k", "m", "m", "black"]
    color_set = [color_set[a] for a in ordered_actions_present]
    colours = ListedColormap(color_set)

    associated_actions = [get_action_name(a) for a in ordered_actions_present]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    action_plot = ax.scatter(fish_positions_compiled_flattened[:, 0], fish_positions_compiled_flattened[:, 1], c=actions_compiled_flattened, cmap=colours)
    ax.legend(handles=action_plot.legend_elements()[0], labels=associated_actions)

    if consumption_timestamps is not None:
        consumption_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in consumption_timestamps]
        consumption_plot = ax.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="black", marker="x")
    if predation_death_timestamps is not None:
        predation_death_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in predation_death_timestamps]
        predation_plot = ax.scatter([p[0] for p in predation_death_positions], [p[1] for p in predation_death_positions], color="r", marker="X")

    plt.show()


def plot_action_sequences_2D_discreteV2(fish_positions_compiled, actions_compiled, action_sequence_timestamps,
                                      consumption_timestamps=None, predation_death_timestamps=None, title=None, save_figure=False):
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

    color_set = ['b', 'g', 'lightgreen', 'r', 'y', 'gold', "c", "m", "m", "black"]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    points = []
    for action in ordered_actions_present:
        colour = color_set[action]
        ts = np.array([i for i, a in enumerate(actions_compiled_flattened) if a==action])
        x = fish_positions_compiled_flattened[ts, 0]
        y = fish_positions_compiled_flattened[ts, 1]
        points.append(ax.scatter(x, y, c=colour))
    # plt.legend(handles=action_plot.legend_elements()[0], labels=associated_actions)
    # ax.legend(points, associated_actions, loc="upper right")

    if consumption_timestamps is not None:
        consumption_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in consumption_timestamps]
        consumption_plot = ax.scatter([p[0] for p in consumption_positions], [p[1] for p in consumption_positions], color="black", marker="x")
    if predation_death_timestamps is not None:
        predation_death_positions = [p for p, t in zip(fish_positions_compiled_flattened, action_sequence_timestamps_flattened) if t in predation_death_timestamps]
        predation_plot = ax.scatter([p[0] for p in predation_death_positions], [p[1] for p in predation_death_positions], color="r", marker="X")

    # ob = AnchoredHScaleBar(size=200, label="20mm", loc=4, frameon=True,
    #                        pad=0.6, sep=4, linekw=dict(color="crimson"), )
    # ax.add_artist(ob)

    scale_bar = AnchoredSizeBar(ax.transData,
                                200, '20mm', 'lower center',
                                pad=1,
                                color='black',
                                frameon=False,
                                size_vertical=1,
                                fontproperties={"size": 16})
    #                           fontproperties=fontprops )
    ax.add_artist(scale_bar)

    ax.set_facecolor('lightgrey')
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    if title is not None:
        plt.title(title)
    if save_figure:
        plt.savefig(f"../../Figures/Panels/Panel-4/{title}")
    plt.show()


# Chosen escape
# for j in [29, 36, 38]:
#     data = load_data(f"dqn_scaffold_21-2", "Behavioural-Data-Free", f"Naturalistic-{j}")
#
#     # Capture sequences
#     ts, escape_sequences, escape_fish_positions = extract_escape_action_sequences_with_positions(
#         data)
#     if len(ts) > 0:
#         plot_action_sequences_2D_discreteV2(escape_fish_positions, escape_sequences, ts, title=f"{j}-dqn_scaffold_18-1")


# Chosen escape
data = load_data(f"dqn_scaffold_21-2", "Behavioural-Data-Free", f"Naturalistic-36")
ts, escape_sequences, escape_fish_positions = extract_escape_action_sequences_with_positions(
    data)
i = 4
escape_fish_positions = escape_fish_positions[i:i+1]
escape_fish_positions = [esc + np.array([-500, 300]) for esc in escape_fish_positions]
escape_sequences = escape_sequences[i:i+1]
ts = ts[i:i+1]

# Chosen exploration
data = load_data(f"dqn_scaffold_18-1", "Behavioural-Data-Free", f"Naturalistic-4")
exploration_timestamps, exploration_sequences, exploration_fish_positions = \
    extract_exploration_action_sequences_with_positions(
    data)
# for i in range(len(exploration_timestamps)):
#     plot_action_sequences_2D_discreteV2(exploration_fish_positions[i:i+1], exploration_sequences[i:i+1],
#                                         exploration_timestamps[i:i+1],
#                                             title=f"{i}-dqn_scaffold_18-1")
i = 3
exploration_fish_positions = exploration_fish_positions[i:i+1]
exploration_sequences = exploration_sequences[i:i+1]
exploration_timestamps = exploration_timestamps[i:i+1]


# Chosen captures
data = load_data(f"dqn_scaffold_14-1", "Behavioural-Data-Free", f"Naturalistic-4")
cts, consumption_timestamps, capture_sequences, capture_fish_positions = extract_consumption_action_sequences_with_positions(
         data)
to_keep = [8, 7, 2]
cts = [cts[i] for i in to_keep]
consumption_timestamps = [consumption_timestamps[i] for i in to_keep]
capture_sequences = [capture_sequences[i] for i in to_keep]
capture_fish_positions = [capture_fish_positions[i] for i in to_keep]
capture_fish_positions = [c-450 for c in capture_fish_positions]


timestamps = cts + exploration_timestamps #+ ts
sequences = capture_sequences + exploration_sequences #+ escape_sequences
fish_positions = capture_fish_positions + exploration_fish_positions #+ escape_fish_positions


plot_action_sequences_2D_discreteV2(fish_positions, sequences, timestamps,
                                                 consumption_timestamps=consumption_timestamps,
                                    title=f"Compiled sequences", save_figure=True)


# plot_action_sequences_2D_discreteV2(capture_fish_positions, capture_sequences, ts,
#                                                  consumption_timestamps=consumption_timestamps, title=f"dqn_scaffold_14-1")
#

# for i in range(len(ts)):
#
#     plot_action_sequences_2D_discreteV2(capture_fish_positions[i:i+1], capture_sequences[i:i+1], ts[i:i+1],
#                                                  consumption_timestamps=consumption_timestamps, title=f"{i}-dqn_scaffold_14-1")

# TO use: 8, 7, 6, 2, 0

# for j in range(1, 21):
#     data = load_data(f"dqn_scaffold_18-1", "Behavioural-Data-Free", f"Naturalistic-{j}")
#
#     # Capture sequences
#     ts, consumption_timestamps, capture_sequences, capture_fish_positions = extract_consumption_action_sequences_with_positions(
#         data)
#     if len(ts) > 0:
#         plot_action_sequences_2D_discreteV2(capture_fish_positions, capture_sequences, ts,
#                                             consumption_timestamps=consumption_timestamps, title=f"{j}-dqn_scaffold_18-1")
#

# for j in range(1, 41):
#     data = load_data(f"dqn_scaffold_21-2", "Behavioural-Data-Free", f"Naturalistic-{j}")
#
#     # Capture sequences
#     ts, escape_sequences, escape_fish_positions = extract_escape_action_sequences_with_positions(
#         data)
#     if len(ts) > 0:
#         plot_action_sequences_2D_discreteV2(escape_fish_positions, escape_sequences, ts, title=f"{j}-dqn_scaffold_18-1")

# for j in range(1, 21):
#     data = load_data(f"dqn_scaffold_14-1", "Behavioural-Data-Free", f"Naturalistic-{j}")
#
#     # Capture sequences
#     ts, consumption_timestamps, capture_sequences, capture_fish_positions = extract_consumption_action_sequences_with_positions(
#         data)
#     if len(ts) > 0:
#         plot_action_sequences_2D_discreteV2(capture_fish_positions, capture_sequences, ts,
#                                             consumption_timestamps=consumption_timestamps, title=f"{j}-dqn_scaffold_18-1")

    # Exploration sequences
    # exploration_timestamps, exploration_sequences, exploration_fish_positions = \
    #     extract_exploration_action_sequences_with_positions(
    #     data)
    # if len(exploration_timestamps) > 0:
    #     plot_action_sequences_2D_discreteV2(exploration_fish_positions, exploration_sequences, exploration_timestamps,
    #                                         title=f"{j}-dqn_scaffold_18-1")





# Displays the exploration and capture sequences for the first trial for each model.
# for i in range(1, 5):
#     for j in range(1, 11):
#         data = load_data(f"dqn_scaffold_18-{i}", "Behavioural-Data-Free", f"Naturalistic-{j}")
#
#         # Capture sequences
#         ts, consumption_timestamps, capture_sequences, capture_fish_positions = extract_consumption_action_sequences_with_positions(data)
#         plot_action_sequences_2D_discreteV2(capture_fish_positions, capture_sequences, ts, consumption_timestamps=consumption_timestamps)
#
#         # Exploration sequences
#         exploration_timestamps, exploration_sequences, exploration_fish_positions = extract_exploration_action_sequences_with_positions(data)
#         plot_action_sequences_2D_discreteV2(exploration_fish_positions, exploration_sequences, exploration_timestamps)
