import numpy as np

from Analysis.Behavioural.Tools.BehavLabels.extract_turn_sequences import extract_turn_sequences
from Analysis.Behavioural.Exploration.turning_analysis_discrete import model_of_action_switching, randomly_switching_fish_new
from Analysis.Behavioural.Exploration.turning_analysis_continuous import convert_continuous_angles_to_turn_directions

from Analysis.load_data import load_data
import math


def get_normalised_turn_chain_metric_discrete(actions):
    """Returns a measure of how long turn chains tend to be, normalised to random switching.
    0 means no preference for turn chains,
    0.35 means noticable preference for longer chains than happens randomly.
    and negative values mean there is a preference for shorter chains.
    """
    turn_sequences = extract_turn_sequences([actions])
    l, r, sl, sr = model_of_action_switching(turn_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_sequences)

    mean_real_l = np.mean(sl)
    mean_real_r = np.mean(sr)
    mean_random_l = np.mean(sl2)
    mean_random_r = np.mean(sr2)

    mean_random = (mean_random_l + mean_random_r)/2
    mean_real = (mean_real_l + mean_real_r)/2

    turn_chain_score = (mean_real - mean_random)/mean_random

    if math.isnan(turn_chain_score):
        return 0
    else:
        return turn_chain_score


def get_normalised_turn_chain_metric_continuous(angles, threshold_for_angle=0.05):
    """Returns a measure of how long turn chains tend to be, normalised to random switching.
    0 means no preference for turn chains,
    0.35 means noticable preference for longer chains than happens randomly.
    and negative values mean there is a preference for shorter chains.

    NOTE: This metric is based on all swims, not just exploration swims, so will always be biased by prey, predator and
    wall interactions
    """
    directions = convert_continuous_angles_to_turn_directions(angles, threshold_for_angle=threshold_for_angle)
    turn_sequences = extract_turn_sequences([directions])
    l, r, sl, sr = model_of_action_switching(turn_sequences)
    l2, r2, sl2, sr2 = randomly_switching_fish_new(turn_sequences)

    mean_real_l = np.mean(sl)
    mean_real_r = np.mean(sr)
    mean_random_l = np.mean(sl2)
    mean_random_r = np.mean(sr2)

    mean_random = (mean_random_l + mean_random_r)/2
    mean_real = (mean_real_l + mean_real_r)/2

    turn_chain_score = (mean_real - mean_random)/mean_random

    return turn_chain_score


if __name__ == "__main__":
    data = load_data("ppo_scaffold_21-1", "Behavioural-Data-Free", "Naturalistic-1")
    tsl = get_normalised_turn_chain_metric_continuous(data["angle"])
