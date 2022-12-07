from Analysis.load_data import load_data

from Analysis.Behavioural.Tools.BehavLabels.extract_sand_grain_interaction_sequences import get_sand_grain_engagement_sequences
from Analysis.Behavioural.VisTools.show_observation_sequences import display_obs_sequence


if __name__ == "__main__":
    for i in range(1, 21):
        data = load_data("dqn_scaffold_33-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        _, timestamps = get_sand_grain_engagement_sequences(data)
        if len(timestamps) > 0:
            observation = data["observation"][timestamps]
            display_obs_sequence(observation)
