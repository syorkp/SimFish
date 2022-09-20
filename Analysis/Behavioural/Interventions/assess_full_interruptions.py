import numpy as np

from Analysis.load_data import load_data


def compute_action_heterogeneity_continuous(actions):
    # Bin actions
    # Call method below.
    ...


def compute_action_heterogeneity_discrete(actions):
    """Returns score from 0 onwards. As heterogeneity increases, so does the score."""
    action_counts = np.bincount(actions)
    differences = [np.max(np.delete(action_counts, i) - action_counts[i]) for i in range(action_counts.shape[0])]
    difference_sum = np.sum(np.absolute(differences))

    heterogeneity = actions.shape[0]/difference_sum
    heterogeneity -= 0.142857142857142836
    return heterogeneity


if __name__ == "__main__":
    d = load_data("dqn_scaffold_18-1", "Interruptions-A", "Naturalistic-1")
    h1 = compute_action_heterogeneity_discrete(d["action"][:20])
    h2 = compute_action_heterogeneity_discrete(d["action"][400:])


