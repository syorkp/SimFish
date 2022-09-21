import numpy as np

from Analysis.load_data import load_data


def compute_action_heterogeneity_continuous(actions):
    # Bin actions
    # Call method below.
    ...


def compute_action_heterogeneity_discrete(actions):
    """Returns score from 0 onwards. As heterogeneity increases, so does the score."""
    action_counts = np.bincount(actions)
    if action_counts.shape[0] < 10:
        action_counts = np.concatenate((action_counts, np.zeros((10 - action_counts.shape[0]))))

    differences = [np.max(np.delete(action_counts, i) - action_counts[i]) for i in range(action_counts.shape[0])]
    difference_sum = np.sum(np.absolute(differences))

    heterogeneity = actions.shape[0]/difference_sum
    heterogeneity -= 0.1
    return heterogeneity


def assess_all_dqn(model_name, n=5):
    dqn_assay_config_suffixes = ["A", "B", "C", "D", "E", "F", "G", "H"]

    for suffix in dqn_assay_config_suffixes:
        mean_heterogeneity_pre = 0
        mean_heterogeneity_post = 0
        for i in range(1, n+1):
            d = load_data(model_name, f"Interruptions-{suffix}", f"Naturalistic-{i}")

            het_pre = compute_action_heterogeneity_discrete(d["action"][:200])
            het_post = compute_action_heterogeneity_discrete(d["action"][200:])
            mean_heterogeneity_pre += het_pre
            mean_heterogeneity_post += het_post
        print("")
        print(suffix)
        print(f"Mean Heterogeneity (pre): {mean_heterogeneity_pre/n}")
        print(f"Mean Heterogeneity (post): {mean_heterogeneity_post/n}")


if __name__ == "__main__":
    assess_all_dqn("dqn_scaffold_18-1", 5)



