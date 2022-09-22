import numpy as np

from Analysis.load_data import load_data
from Analysis.Behavioural.Tools.BehavLabels.extract_turn_sequences import extract_turn_sequences, extract_purely_turn_sequences
from Analysis.Behavioural.TurnChains.turning_analysis_shared import randomly_switching_fish, model_of_action_switching, \
    cumulative_turn_direction_plot_multiple_models, cumulative_turn_direction_plot, \
    cumulative_switching_probability_plot_multiple_models
from Analysis.Behavioural.Tools.BehavLabels.extract_exploration_sequences import \
    label_exploration_sequences_free_swimming_multiple_trials, \
    label_exploration_sequences_no_prey_multiple_trials


def convert_continuous_angles_to_turn_directions(angles, threshold_for_angle):
    """Given an array of angles, converts to array of elements:
      - 0 No angle
      - 1 Left angle
      - 2 Right angle
    """
    turn_direction_array = np.zeros(angles.shape).astype(int)
    turn_direction_array[angles < -threshold_for_angle] = 1
    turn_direction_array[angles > threshold_for_angle] = 2
    return turn_direction_array


def get_all_angles(model_name, assay_config, assay_id, n):
    compiled_angles = []
    for trial in range(1, n+1):
        data = load_data(model_name, assay_config, f"{assay_id}-{trial}")
        compiled_angles.append(data["angle"][1:])
    return compiled_angles


def get_all_angle_sequences_labelled(compiled_turn_directions, compiled_labels):
    compiled_all_action_sequences = []
    for labels, turn_directions in zip(compiled_labels, compiled_turn_directions):

        timestamps = [i for i, a in enumerate(labels) if a == 1]
        all_action_sequences = []
        current_sequence = []

        for i, t in enumerate(timestamps):
            if i == 0:
                current_sequence.append(turn_directions[t])
            else:
                if t - 1 == timestamps[i - 1] or t - 2 == timestamps[i - 1]:
                    current_sequence.append(turn_directions[t])
                else:
                    all_action_sequences.append(current_sequence)
                    current_sequence = []
                    current_sequence.append(turn_directions[t])

        compiled_all_action_sequences.append(all_action_sequences)
    compiled_all_action_sequences = [item for sublist in compiled_all_action_sequences for item in sublist]
    return compiled_all_action_sequences


def plot_all_turn_analysis_multiple_models_continuous(model_names, assay_config, assay_id, n,
                                                      use_purely_turn_sequences=False, threshold_for_angle=0.1, data_cutoff=None):
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
        no_prey_exploration_labelled = label_exploration_sequences_no_prey_multiple_trials(model_name, assay_config,
                                                                                           assay_id, n)
        free_swimming_exploration_labelled = label_exploration_sequences_free_swimming_multiple_trials(model_name,
                                                                                                       assay_config,
                                                                                                       assay_id, n)
        all_angles = get_all_angles(model_name, assay_config, assay_id, n)

        if data_cutoff is not None:
            no_prey_exploration_labelled = [labels[:data_cutoff] for labels in no_prey_exploration_labelled]
            free_swimming_exploration_labelled = [labels[:data_cutoff] for labels in free_swimming_exploration_labelled]
            all_angles = [angles[:data_cutoff] for angles in all_angles]

        # Convert angle sequences to int encoding
        all_directions = [convert_continuous_angles_to_turn_directions(trial, threshold_for_angle=0.05) for trial in all_angles]

        # Get sequences of angles
        no_prey_exploration_angle_sequences = get_all_angle_sequences_labelled(all_directions, no_prey_exploration_labelled)
        free_swimming_exploration_angle_sequences = get_all_angle_sequences_labelled(all_directions, free_swimming_exploration_labelled)

        if use_purely_turn_sequences:
            turn_exploration_sequences = extract_purely_turn_sequences(free_swimming_exploration_angle_sequences, 5)
            turn_no_prey_sequences = extract_purely_turn_sequences(no_prey_exploration_angle_sequences, 5)
            turn_no_prey_sequences_list.append(turn_no_prey_sequences)
        else:
            turn_exploration_sequences = extract_turn_sequences(free_swimming_exploration_angle_sequences)
            turn_no_prey_sequences = extract_turn_sequences(no_prey_exploration_angle_sequences)
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
    l2, r2, sl2, sr2 = randomly_switching_fish()
    cumulative_switching_probability_plot_multiple_models(compiled_sl_exploration, compiled_sr_exploration, sl2, sr2,
                                                          label=f"Cumulative Switching Probability (exploration) {model_name}")

    l2, r2, sl2, sr2 = randomly_switching_fish()
    cumulative_switching_probability_plot_multiple_models(compiled_sl_no_prey, compiled_sr_no_prey,  sl2, sr2, label=f"Cumulative Switching Probability (no prey) {model_name}")


if __name__ == "__main__":
    plot_all_turn_analysis_multiple_models_continuous(["ppo_scaffold_21-1", "ppo_scaffold_21-2"], "Behavioural-Data-Free",
                                           f"Naturalistic", 10, threshold_for_angle=0.05, data_cutoff=None)
    # d = load_data("ppo_scaffold_21-1", "Behavioural-Data-Free", "Naturalistic-3")
    # exploration_np_ts = label_exploration_sequences_free_swimming(d) * 1
    # all_turns = d["angle"][1:]
    # turn_directions = convert_continuous_angles_to_turn_directions(all_turns, 0.05)
    #
    # timestamps = [i for i, a in enumerate(exploration_np_ts) if a == 1]
    #
    # all_action_sequences = []
    # current_sequence = []
    # for i, t in enumerate(timestamps):
    #     if i == 0:
    #         current_sequence.append(turn_directions[t])
    #     else:
    #         if t-1 == timestamps[i-1] or t-2 == timestamps[i-1]:
    #             current_sequence.append(turn_directions[t])
    #         else:
    #             all_action_sequences.append(current_sequence)
    #             current_sequence = []
    #             current_sequence.append(turn_directions[t])
    #
    # valid_sequences = [seq for seq in all_action_sequences if len(seq) > 1]
    #
    # l, r, sl, sr = model_of_action_switching(valid_sequences)
    # l2, r2, sl2, sr2 = randomly_switching_fish()
    # cumulative_switching_probability_plot(sl, sr, sl2, sr2, "Nowhere")
