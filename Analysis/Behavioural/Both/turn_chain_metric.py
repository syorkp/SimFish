from Analysis.load_data import load_data


def get_normalised_mean_turn_streak_length(actions):
    x = True



if __name__ == "__main__":
    data = load_data("dqn_scaffold_14-1", "Behavioural-Data-Free", "Naturalistic-1")
    tsl = get_normalised_mean_turn_streak_length(data["action"])
