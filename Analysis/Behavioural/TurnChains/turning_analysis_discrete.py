import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import extract_exploration_action_sequences_with_positions
from Analysis.Behavioural.Tools.BehavLabels.extract_turn_sequences import extract_turn_sequences, extract_purely_turn_sequences
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import extract_exploration_action_sequences_with_fish_angles
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import get_no_prey_stimuli_sequences, get_exploration_sequences
from Analysis.Behavioural.TurnChains.turning_analysis_shared import model_of_action_switching, plot_turning_sequences, \
    randomly_switching_fish, randomly_switching_fish_new, cumulative_switching_probability_plot, cumulative_turn_direction_plot_multiple_models, \
    cumulative_switching_probability_plot_multiple_models
from Analysis.Behavioural.TurnChains.turning_analysis_shared import cumulative_turn_direction_plot


def get_cumulative_switching_probability_plot(action_sequences, figure_save_location):
    turn_sequences = extract_purely_turn_sequences(action_sequences)
    l, r, sl, sr = model_of_action_switching(turn_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_sequences)
    cumulative_switching_probability_plot(sl, sr, sl2, sr2, figure_save_location)


def divide_sequences(action_sequences):
    new_action_sequences = []
    for seq in action_sequences:
        while len(seq) > 11:
            new_action_sequences.append(seq[:11])
            seq = seq[11:]
        new_action_sequences.append(seq)
    return new_action_sequences


def get_frameshift_sequences(action_sequences):
    new_sequences = []
    for seq in action_sequences:
        new_sequences.append(seq)
        for i in range(1, len(seq)-8):
            if seq[i] != seq[-1]:
                new_sequences.append(seq[i:])
    return new_sequences


def plot_all_turn_analysis(model_name, assay_config, assay_id, n, use_purely_turn_sequences=True, data_cutoff=None):
    no_prey_actions, no_prey_timestamps = get_no_prey_stimuli_sequences(model_name, assay_config, assay_id, n, data_cutoff)
    otherwise_exploration_sequences = get_exploration_sequences(model_name, assay_config, assay_id, n, data_cutoff)

    if use_purely_turn_sequences:
        turn_exploration_sequences = extract_purely_turn_sequences(otherwise_exploration_sequences, 5)
        turn_no_prey_sequences = extract_purely_turn_sequences(no_prey_actions, 5)
    else:
        turn_exploration_sequences = extract_turn_sequences(otherwise_exploration_sequences)
        turn_no_prey_sequences = extract_turn_sequences(no_prey_actions)

    # Cumulative turn direction plots:
    cumulative_turn_direction_plot(turn_no_prey_sequences,
                                   label=f"Cumulative Turn Direction (no prey near, only turns) {model_name}")
    cumulative_turn_direction_plot(turn_exploration_sequences,
                                   label=f"Cumulative Turn Direction (no prey or walls, only turns) {model_name}")

    # Cumulative probability plot.
    l, r, sl, sr = model_of_action_switching(turn_exploration_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_exploration_sequences)
    cumulative_switching_probability_plot(sl, sr, sl2, sr2, save_location=f"Cumulative Switching Probability (exploration) {model_name}")

    l, r, sl, sr = model_of_action_switching(turn_no_prey_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_no_prey_sequences)
    cumulative_switching_probability_plot(sl, sr, sl2, sr2, save_location=f"Cumulative Switching Probability (no prey) {model_name}")


def plot_all_turn_analysis_multiple_models_discrete(model_names, assay_config, assay_id, n, use_purely_turn_sequences=True,
                                                    data_cutoff=None):
    compiled_l_exploration = []
    compiled_r_exploration = []
    compiled_sl_exploration = []
    compiled_sr_exploration = []

    compiled_l_no_prey = []
    compiled_r_no_prey = []
    compiled_sl_no_prey = []
    compiled_sr_no_prey = []

    turn_no_prey_sequences_list = []
    for model_name in model_names:
        no_prey_actions, no_prey_timestamps = get_no_prey_stimuli_sequences(model_name, assay_config, assay_id, n, data_cutoff=data_cutoff)
        otherwise_exploration_sequences = get_exploration_sequences(model_name, assay_config, assay_id, n, data_cutoff=data_cutoff)

        if use_purely_turn_sequences:
            turn_exploration_sequences = extract_purely_turn_sequences(otherwise_exploration_sequences, 5)
            turn_no_prey_sequences = extract_purely_turn_sequences(no_prey_actions, 5)
            turn_no_prey_sequences_list.append(turn_no_prey_sequences)
        else:
            turn_exploration_sequences = extract_turn_sequences(otherwise_exploration_sequences)
            turn_no_prey_sequences = extract_turn_sequences(no_prey_actions)
            turn_no_prey_sequences_list.append(turn_no_prey_sequences)

        l, r, sl, sr = model_of_action_switching(turn_exploration_sequences)

        compiled_l_exploration.append(l)
        compiled_r_exploration.append(r)
        compiled_sl_exploration.append(sl)
        compiled_sr_exploration.append(sr)

        l, r, sl, sr = model_of_action_switching(turn_no_prey_sequences)

        compiled_l_no_prey.append(l)
        compiled_r_no_prey.append(r)
        compiled_sl_no_prey.append(sl)
        compiled_sr_no_prey.append(sr)

    # Cumulative turn direction plots:
    cumulative_turn_direction_plot_multiple_models(turn_no_prey_sequences_list)
    cumulative_turn_direction_plot(turn_exploration_sequences,
                                   label=f"Cumulative Turn Direction (no prey or walls, only turns) {model_name}")

    # Cumulative probability plot.
    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_exploration_sequences)
    cumulative_switching_probability_plot_multiple_models(compiled_sl_exploration, compiled_sr_exploration, sl2, sr2,
                                                          label=f"Cumulative Switching Probability (exploration) {model_name}")

    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_no_prey_sequences)
    cumulative_switching_probability_plot_multiple_models(compiled_sl_no_prey, compiled_sr_no_prey,  sl2, sr2, label=f"Cumulative Switching Probability (no prey) {model_name}")


if __name__ == "__main__":
    # plot_all_turn_analysis("dqn_scaffold_14-1", "Behavioural-Data-Full-Interruptions",
    #                        f"Naturalistic", 3)

    plot_all_turn_analysis_multiple_models_discrete(["dqn_scaffold_14-1", "dqn_scaffold_14-2"], "Behavioural-Data-Free",
                                                    f"Naturalistic", 10, data_cutoff=None)

    # data = load_data("dqn_scaffold_18-1", "Behavioural-Data-Free", f"Naturalistic-18")
    # exploration_timestamps, exploration_sequences, exploration_fish_orientations = \
    #     extract_exploration_action_sequences_with_fish_angles(data)
    #
    # rt_usage = [i for i, a in enumerate(data["action"]) if a == 1 or a == 2]
    # plot_turning_sequences(exploration_fish_orientations[len(exploration_fish_orientations)-1][:-1])

    # for i in range(len(exploration_fish_orientations)):
    #     # to_keep = [o for t, o in enumerate(exploration_fish_orientations[i])
    #     #            if exploration_timestamps[i][t] in rt_usage]
    #     # if len(to_keep) > 0:
    #     #     plot_turning_sequences(to_keep)
    #     plot_turning_sequences(exploration_fish_orientations[i][:-1])


    # sl_compiled = []
    # sr_compiled = []
    # sl2_compiled = []
    # sr2_compiled = []
    # turn_exploration_sequences_compiled = []
    # for i in range(1, 5):
    #     data = load_data(f"dqn_scaffold_14-{i}", "Behavioural-Data-Free", f"Naturalistic-1")
    #
    #     # Cumulative turn direction plot
    #     exploration_timestamps, exploration_sequences, exploration_fish_positions = extract_exploration_action_sequences_with_positions(data)
    #     turn_exploration_sequences = extract_turn_sequences(exploration_sequences)
    #     cumulative_turn_direction_plot(turn_exploration_sequences)
    #
    #     # Orientation plot
    #     exploration_timestamps, exploration_sequences, exploration_fish_orientations = extract_exploration_action_sequences_with_fish_angles(data)
    #     plot_turning_sequences(exploration_fish_orientations[-2])
    #
    #     # Cumulative probability plot.
    #     l, r, sl, sr = model_of_action_switching(turn_exploration_sequences)
    #     l2, r2, sl2, sr2 = randomly_switching_fish()
    #     cumulative_switching_probability_plot(sl, sr, sl2, sr2)
    #
    #     turn_exploration_sequences_compiled += turn_exploration_sequences
    #     sl_compiled += sl
    #     sr_compiled += sr
    #     sl2_compiled += sl2
    #     sr2_compiled += sr2
    #
    # cumulative_switching_probability_plot(sl_compiled, sr_compiled, sl2_compiled, sr2_compiled)
    # cumulative_turn_direction_plot(turn_exploration_sequences_compiled)

    #
    # sl_compiled = []
    # sr_compiled = []
    # sl2_compiled = []
    # sr2_compiled = []
    # turn_exploration_sequences_compiled = []

    # Exploration sequences based on visual stimulation level
    # plot_all_turn_analysis(f"dqn_scaffold_14-1", "Behavioural-Data-Free", f"Naturalistic", 20, data_cutoff=200)
    # plot_all_turn_analysis(f"dqn_scaffold_14-2", "Behavioural-Data-Free", f"Naturalistic", 10)




        # turn_exploration_sequences_compiled += turn_exploration_sequences
        # sl_compiled += sl
        # sr_compiled += sr
        # sl2_compiled += sl2
        # sr2_compiled += sr2

    # cumulative_switching_probability_plot(sl_compiled, sr_compiled, sl2_compiled, sr2_compiled, f"Cumulative Switching Probability Plot all models")
    # cumulative_turn_direction_plot(turn_exploration_sequences_compiled, f"Cumulative turn direction plot all models")













    # VERSION 1

    # for i in range(1, 10):
    #     data = load_data("new_even_prey_ref-2", "Behavioural-Data-Free", f"Prey-{i}")
    #     colored_2d_track_turns(data["position"][100:500], data["behavioural choice"][100:500])
    #
    # orientation_log = []
    # action_sequences = []
    # for j in range(1, 4):
    #     for i in range(1, 11):
    #         data = load_data("new_differential_prey_ref-3", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
    #         new_as = get_free_swimming_sequences(data)
    #         action_sequences += [[a for a in seq if a == 1 or a == 2] for seq in new_as]
    #         orientation_changes = [data["fish_angle"][i]-data["fish_angle"][i-1] for i, angle in enumerate(data["fish_angle"]) if i!=0]
    #         orientation_log = orientation_log + orientation_changes
    #         # colored_2d_track_turns(data["position"][-200:], data["behavioural choice"][-200:], orientation_changes[-200:])
    #         plot_turning_sequences(data["fish_angle"])

    # all_action_sequences = []
    # for x in [3, 4]:
    #     action_sequences = []
    #     for j in range(1, 4):
    #         for i in range(1, 11):
    #             data = load_data(f"new_differential_prey_ref-{x}", f"Behavioural-Data-Free-{j}", f"Naturalistic-{i}")
    #             new_as = get_free_swimming_sequences(data)
    #             action_sequences += [[a for a in seq if a == 1 or a == 2] for seq in new_as]
    #             orientation_changes = [data["fish_angle"][i]-data["fish_angle"][i-1] for i, angle in enumerate(data["fish_angle"]) if i!=0]
    #             orientation_log = orientation_log + orientation_changes
    #     action_sequences = divide_sequences(action_sequences)
    #     all_action_sequences.append(action_sequences)
    #     l, r, sl, sr = model_of_action_switching(action_sequences)
    #     l2, r2, sl2, sr2 = randomly_switching_fish()
    #     plot_switching_distribution(sl, sr, sl2, sr2)
    # # action_sequences = get_frameshift_sequences(action_sequences)
    #
    # new_switching_plot2(all_action_sequences)


    # plot_turning_sequences(data["fish_angle"])
    # colored_2d_track_turns(data["position"][-200:], data["behavioural choice"][-200:])
