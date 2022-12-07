from Analysis.load_data import load_data

from Analysis.Behavioural.Tools.BehavLabels.extract_capture_sequences import extract_consumption_action_sequences_with_positions
from Analysis.Behavioural.VisTools.show_observation_sequences import display_obs_sequence


if __name__ == "__main__":
    for i in range(1, 21):
        data = load_data("dqn_scaffold_33-2", "Behavioural-Data-Free", f"Naturalistic-{i}")
        timestamps, _, _, _ = extract_consumption_action_sequences_with_positions(data)
        if len(timestamps) > 0:
            for ts in timestamps:
                observation = data["observation"][ts]
                display_obs_sequence(observation)
    print("test...")
